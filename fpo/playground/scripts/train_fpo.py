import datetime
import time
from typing import Annotated
import os
os.environ['MUJOCO_GL'] = 'egl' # comment if not on gcp

import jax
import jax_dataclasses as jdc
import numpy as onp
import tyro
import wandb
from jax import numpy as jnp
from mujoco_playground import dm_control_suite, locomotion, registry
from tqdm import tqdm

from flow_policy import fpo, rollouts


def log_final_video(
    env,
    agent_state: fpo.FpoState,
    wandb_run: wandb.sdk.wandb_run.Run,
    config: fpo.FpoConfig,
    seed: int,
) -> None:
    """
    Run a short deterministic rollout with the trained policy and log a video to wandb.

    This mirrors the notebook example in `train_fpo.ipynb` but logs directly as a
    `wandb.Video` instead of writing to disk.
    """
    try:
        # Single-environment rollout.
        prng = jax.random.key(seed + 12345)
        state = env.reset(prng)
        rollout = [state]

        # Use a shorter rollout than full episode_length for quick visualization.
        n_steps = min(config.episode_length, 200)
        for _ in range(n_steps):
            prng, act_prng = jax.random.split(prng)
            action, _ = agent_state.sample_action(
                state.obs, act_prng, deterministic=True
            )
            state = env.step(state, jnp.tanh(action))
            rollout.append(state)

        # Render frames and log to wandb as a video.
        frames = env.render(rollout, height=480, width=640)
        frames_np = onp.array(frames)
        fps = float(1.0 / getattr(env, "dt", 0.05))
        wandb_run.log(
            {"eval/final_policy_video": wandb.Video(frames_np, fps=fps, format="mp4")}
        )
    except Exception as e:
        # Don't let rendering failures crash training; just report them.
        print(f"Could not render final policy video: {e}", flush=True)


def main(
    # TODO: these envs are just what i'm testing with initially
    env_name: Annotated[
        str,
        tyro.conf.arg(
            constructor=tyro.extras.literal_type_from_choices(
                dm_control_suite.ALL_ENVS + locomotion.ALL_ENVS
            )
        ),
    ],
    wandb_entity: str,
    wandb_project: str,
    config: fpo.FpoConfig,
    exp_name: str = "",
    seed: int = 0,
) -> None:
    """Main function to train PPO on a specified environment."""

    # Load environment config and PPO parameters.
    env_config = registry.get_default_config(env_name)

    # Logging.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb_run = wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        name=f"fpo_{env_name}_{exp_name}_{timestamp}",
        config={
            "env_name": env_name,
            "fpo_params": jdc.asdict(config),
            "learning_rate": config.learning_rate,
            "clipping_epsilon": config.clipping_epsilon,
            "seed": seed,
        },
    )

    # Initialize.
    env = registry.load(env_name, config=env_config)
    agent_state = fpo.FpoState.init(prng=jax.random.key(seed), env=env, config=config)
    rollout_state = rollouts.BatchedRolloutState.init(
        env,
        prng=jax.random.key(seed),
        num_envs=config.num_envs,
    )

    # Perform rollout.
    outer_iters = config.num_timesteps // (config.iterations_per_env * config.num_envs)
    eval_iters = set(onp.linspace(0, outer_iters - 1, config.num_evals, dtype=int))

    times = [time.time()]
    for i in tqdm(range(outer_iters)):
        # Evaluation. Note: this might be better done *after* the training step.
        if i in eval_iters:
            eval_outputs = rollouts.eval_policy(
                agent_state,
                prng=jax.random.fold_in(agent_state.prng, i),
                num_envs=128,
                max_episode_length=config.episode_length,
            )

            # Convert to numpy for printing.
            s_np = {k: onp.array(v) for k, v in eval_outputs.scalar_metrics.items()}

            # Print summary.
            print(f"Eval metrics at step {i}:")
            print(
                f"  Reward: mean={s_np['reward_mean']:.2f}, min={s_np['reward_min']:.2f}, max={s_np['reward_max']:.2f}, std={s_np['reward_std']:.2f}"
            )
            print(
                f"  Steps:  mean={s_np['steps_mean']:.1f}, min={s_np['steps_min']:.1f}, max={s_np['steps_max']:.1f}, std={s_np['steps_std']:.1f}"
            )

            # Log to wandb using the new API.
            eval_outputs.log_to_wandb(wandb_run, step=i)

        # Training step.
        rollout_state, transitions = rollout_state.rollout(
            agent_state,
            episode_length=config.episode_length,
            iterations_per_env=config.iterations_per_env,
        )
        agent_state, metrics = agent_state.training_step(transitions)

        # Train metric logging.
        # NOTE: wandb.Histogram will raise if all values are NaN (autodetected range is not finite),
        # so we sanitize the reward samples before constructing the histogram.
        reward_samples = onp.array(transitions.reward.flatten()[::16])  # type: ignore
        finite_mask = onp.isfinite(reward_samples)
        reward_samples = reward_samples[finite_mask]

        log_dict = {
            "train/mean_reward": onp.mean(transitions.reward),
            "train/mean_steps": (
                # Approximate the mean steps per episode.
                transitions.discount.size / jnp.sum(transitions.discount == 0.0)
            ),
            # Add all training metrics with their means
            **{f"train/{k}": onp.mean(v) for k, v in metrics.items()},
        }

        # Only log the histogram if there is at least one finite value.
        if reward_samples.size > 0:
            log_dict["train/reward_histogram"] = wandb.Histogram(reward_samples)  # type: ignore

        wandb_run.log(log_dict, step=i)

        times.append(time.time())

    print("First train step time:", times[1] - times[0])
    print("~Train time:", times[-1] - times[1])

    # After training, log a short deterministic rollout video to wandb.
    log_final_video(env=env, agent_state=agent_state, wandb_run=wandb_run, config=config, seed=seed)

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    tyro.cli(main, config=(tyro.conf.FlagConversionOff,))
