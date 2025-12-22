import datetime
import time
from typing import Annotated
import os
import jax
import jax_dataclasses as jdc
import numpy as onp
import tyro
import wandb
from jax import numpy as jnp
from tqdm import tqdm
import gymnasium as gym
from gymnasium.wrappers import RecordVideo, FilterObservation, FlattenObservation
import orbax.checkpoint
import pickle

from flow_policy import fpo, rollouts
import flow_policy.gym_utils as gym_utils

def log_final_video(
    env_name: str,
    agent_state: fpo.FpoState,
    wandb_run: wandb.sdk.wandb_run.Run,
    config: fpo.FpoConfig,
    seed: int,
    run_tag: str,
) -> None:
    """Run a short deterministic rollout with the trained policy and log a video."""
    try:
        # Create a single environment for recording
        video_folder = os.path.join("videos", run_tag)
        os.makedirs(video_folder, exist_ok=True)

        try:
            env = gym.make(env_name, render_mode="rgb_array", max_episode_steps=280)
        except Exception:
            import gymnasium_robotics

            env = gym.make(env_name, render_mode="rgb_array", max_episode_steps=280)

        env = RecordVideo(env, video_folder, episode_trigger=lambda x: True)

        obs_dict, _ = env.reset(seed=seed + 12345)
        if isinstance(obs_dict, dict) and "observation" in obs_dict:
            obs = obs_dict["observation"]
        else:
            obs = obs_dict

        done = False

        while not done:
            obs_jax = jnp.array(obs[None, :], dtype=jnp.float32)
            prng = jax.random.key(0)  # deterministic anyway

            # FpoState handles normalization internally using obs_stats
            action, _ = agent_state.sample_action(obs_jax, prng, deterministic=True)

            action_np = onp.array(action[0])

            obs_dict, reward, terminated, truncated, info = env.step(action_np)
            if isinstance(obs_dict, dict) and "observation" in obs_dict:
                obs = obs_dict["observation"]
            else:
                obs = obs_dict

            done = terminated or truncated

        env.close()

        video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
        if video_files:
            video_path = os.path.join(video_folder, video_files[0])
            print(f"Saved final video locally to: {video_path}")
        else:
            print("No video file generated.")

    except Exception as e:
        print(f"Could not render final policy video: {e}", flush=True)


def save_checkpoint(agent_state: fpo.FpoState, run_tag: str, step: int, base_dir: str = "checkpoints"):
    """Save agent state checkpoint."""
    try:
        abs_base_dir = os.path.abspath(base_dir)
        ckpt_dir = os.path.join(abs_base_dir, run_tag, str(step))
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        orbax_checkpointer.save(ckpt_dir, agent_state)
        print(f"Saved checkpoint to {ckpt_dir}")
    except Exception as e:
        print(f"Failed to save checkpoint: {e}")


def load_bc_checkpoint(agent_state: fpo.FpoState, checkpoint_path: str) -> fpo.FpoState:
    print(f"Loading BC checkpoint from {checkpoint_path}...")
    with open(checkpoint_path, "rb") as f:
        ckpt = pickle.load(f)

    # Inject policy params
    new_params = agent_state.params
    # We only want to replace the policy, not the value function
    new_params = jdc.replace(new_params, policy=ckpt["policy_params"])

    # Inject observation stats
    new_obs_stats = agent_state.obs_stats
    # Add epsilon to std to avoid division by zero
    safe_std = ckpt["obs_std"] + 1e-6
    new_obs_stats = jdc.replace(new_obs_stats, mean=ckpt["obs_mean"], std=safe_std)

    # Return updated agent state
    return jdc.replace(agent_state, params=new_params, obs_stats=new_obs_stats)


def main(
    env_name: str,
    wandb_entity: str,
    wandb_project: str,
    config: fpo.FpoConfig,
    bc_checkpoint: str,
    exp_name: str = "",
    seed: int = 0,
) -> None:
    """Train FPO in a Gym env, initialized from a BC checkpoint."""

    # Logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    method = "fpo"
    run_tag = f"{env_name}_{method}_run{timestamp}"

    # Initialize Gym Environment
    print(f"Initializing Gym environment: {env_name}")
    # Use the standard GymWrapper (works for Mujoco / D4RL pen)
    gym_env = gym_utils.GymWrapper(env_name, num_envs=config.num_envs, seed=seed)
    print(f"Obs size: {gym_env.observation_size}, Action size: {gym_env.action_size}")

    # Initialize FPO Agent
    print("Initializing FPO Agent...")
    agent_state = fpo.FpoState.init(prng=jax.random.key(seed), env=gym_env, config=config)

    # Load BC weights
    if bc_checkpoint:
        agent_state = load_bc_checkpoint(agent_state, bc_checkpoint)
        print("BC Checkpoint loaded successfully.")

    wandb_run = wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        name=f"fpo_gym_{env_name}_{exp_name}_{timestamp}",
        config={
            "env_name": env_name,
            "fpo_params": jdc.asdict(config),
            "learning_rate": config.learning_rate,
            "seed": seed,
            "bc_checkpoint": bc_checkpoint,
        },
    )

    # Optional: evaluate the pure BC policy before RL fine-tuning
    if bc_checkpoint:
        print("Running initial BC policy evaluation (before FPO fine-tuning)...")
        eval_num_envs = 16
        eval_env = gym_utils.GymWrapper(env_name, num_envs=eval_num_envs, seed=seed + 500)
        eval_outputs_init = gym_utils.eval_policy_gym(
            eval_env,
            agent_state,
            prng=jax.random.key(seed + 1),
            num_envs=eval_num_envs,
            max_episode_length=config.episode_length,
        )
        eval_env.close()

        init_metrics = {
            f"bc_init/{k}": float(v) for k, v in eval_outputs_init.scalar_metrics.items()
        }
        wandb_run.log(init_metrics, step=0)

        s_np_init = {k: onp.array(v) for k, v in eval_outputs_init.scalar_metrics.items()}
        print(
            "Initial BC policy evaluation:",
            f"Reward mean={s_np_init['reward_mean']:.2f},",
            f"min={s_np_init['reward_min']:.2f},",
            f"max={s_np_init['reward_max']:.2f}",
        )

    # Initialize Rollout State
    print("Resetting environments to initialize rollout state...")
    obs = gym_env.reset(seed=seed)
    print("Environments reset.")

    rollout_state = gym_utils.GymRolloutState(
        last_obs=jnp.array(obs, dtype=jnp.float32),
        steps=jnp.zeros((config.num_envs,), dtype=jnp.int32),
        prng=jax.random.key(seed),
    )

    # CRITIC WARMUP: Keep a copy of the BC policy parameters
    print("Starting Critic Warmup setup (Freezing Policy)...")
    initial_policy_params = agent_state.params.policy
    warmup_iters = 30
    # Strength of BC anchoring after warmup; we will decay this over time.
    # Start very conservative (strong pull toward BC).
    initial_bc_anchor = 0.99
    # Scale down actor steps after warmup so post-BC updates are small.
    step_scale = 0.05

    # Training Loop
    outer_iters = config.num_timesteps // (config.iterations_per_env * config.num_envs)
    anchor_decay_iters = int(0.5 * outer_iters)  # decay anchor over first half of training
    eval_iters = set(onp.linspace(0, outer_iters - 1, config.num_evals, dtype=int))

    print(f"Starting training for {outer_iters} iterations (Warmup: {warmup_iters})...")
    print(f"Evaluation will run at steps: {sorted(list(eval_iters))}")
    print("Entering training loop...")

    # Track previous policy params so we can shrink each update step.
    prev_policy_params = agent_state.params.policy

    for i in tqdm(range(outer_iters)):
        # Evaluation
        if i in eval_iters:
            print(f"Step {i}: Starting evaluation...")
            # Create fewer envs for eval
            eval_num_envs = 16  # Use same as training or suitable number
            eval_env = gym_utils.GymWrapper(env_name, num_envs=eval_num_envs, seed=seed + 1000)

            # Use the generic eval_policy_gym utility
            eval_outputs = gym_utils.eval_policy_gym(
                eval_env,
                agent_state,
                prng=jax.random.fold_in(agent_state.prng, i),
                num_envs=eval_num_envs,
                max_episode_length=config.episode_length,
            )

            eval_env.close()

            # Log metrics and print a brief summary
            eval_outputs.log_to_wandb(wandb_run, step=i)
            s_np = {k: onp.array(v) for k, v in eval_outputs.scalar_metrics.items()}
            tqdm.write(f"Eval metrics at step {i}:")
            tqdm.write(f"  Reward: mean={s_np['reward_mean']:.2f}")

        # Training Step
        rollout_state, transitions = gym_utils.gym_rollout(
            gym_env,
            rollout_state,
            agent_state,
            episode_length=config.episode_length,
            iterations_per_env=config.iterations_per_env,
            deterministic=False,
        )

        # FPO update
        agent_state, metrics = agent_state.training_step(transitions)

        # WARMUP LOGIC: Reset policy parameters if in warmup phase
        if i < warmup_iters:
            # Reset policy weights to initial BC weights
            new_params = jdc.replace(agent_state.params, policy=initial_policy_params)
            agent_state = jdc.replace(agent_state, params=new_params)
            metrics["is_warmup"] = 1.0

            # If this is the LAST warmup step, reset the optimizer state
            if i == warmup_iters - 1:
                print("Warmup complete. Resetting optimizer state to clear momentum...")
                # Re-initialize optimizer state. This clears momentum for both Actor and Critic.
                new_opt_state = agent_state.opt.init(agent_state.params)
                agent_state = jdc.replace(agent_state, opt_state=new_opt_state)
        else:
            # After warmup, gradually decay the strength of BC anchoring.
            metrics["is_warmup"] = 0.0

            # First, shrink the raw actor update so each RL step is small.
            # new_policy is what FPO just computed; prev_policy_params is from
            # the previous iteration.
            new_policy = agent_state.params.policy
            scaled_policy = jax.tree.map(
                lambda new, old: old + step_scale * (new - old),
                new_policy,
                prev_policy_params,
            )
            agent_state = jdc.replace(
                agent_state,
                params=jdc.replace(agent_state.params, policy=scaled_policy),
            )

            # Compute a decaying anchor strength: start at initial_bc_anchor and
            # linearly decay to 0 over anchor_decay_iters.
            if anchor_decay_iters > 0:
                progress = min(1.0, i / anchor_decay_iters)
            else:
                progress = 1.0
            bc_anchor_coef = initial_bc_anchor * (1.0 - progress)
            metrics["bc_anchor_coef"] = float(bc_anchor_coef)

            # BC ANCHORING: gently pull the policy back toward the initial BC policy.
            if bc_anchor_coef > 0.0:
                mixed_policy = jax.tree.map(
                    lambda p, p_bc: (1.0 - bc_anchor_coef) * p + bc_anchor_coef * p_bc,
                    agent_state.params.policy,
                    initial_policy_params,
                )
                new_params = jdc.replace(agent_state.params, policy=mixed_policy)
                agent_state = jdc.replace(agent_state, params=new_params)

        # Update previous policy params to the final policy at this iteration.
        prev_policy_params = agent_state.params.policy

        # Logging
        per_env_returns = onp.sum(transitions.reward, axis=0)
        log_dict = {
            "train/mean_step_reward": float(onp.mean(transitions.reward)),
            "train/mean_episode_return": float(onp.mean(per_env_returns)),
            **{f"train/{k}": float(onp.mean(v)) for k, v in metrics.items()},
        }
        wandb_run.log(log_dict, step=i)

    print("Training finished.")
    gym_env.close()

    # Save final checkpoint
    save_checkpoint(agent_state, run_tag, outer_iters)

    # Log final video
    print("Logging final video...")
    log_final_video(env_name, agent_state, wandb_run, config, seed, run_tag)

    wandb.finish()


if __name__ == "__main__":
    tyro.cli(main, config=(tyro.conf.FlagConversionOff,))
