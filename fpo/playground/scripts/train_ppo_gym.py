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
from gymnasium.wrappers import RecordVideo
import orbax.checkpoint

from flow_policy import ppo, rollouts
from flow_policy.gym_utils import GymWrapper, GymRolloutState, gym_rollout, eval_policy_gym


def log_final_video(
    env_name: str,
    agent_state: ppo.PpoState,
    wandb_run: wandb.sdk.wandb_run.Run,
    config: ppo.PpoConfig,
    seed: int,
    run_tag: str,
) -> None:
    """
    Run a short deterministic rollout with the trained policy and log a video to wandb.
    """
    try:
        # Create a single environment for recording
        # Use RecordVideo wrapper
        # Folder naming: [task]_[method]_run[YYYYMMDD_HHMMSS]
        video_folder = os.path.join("videos", run_tag)
        os.makedirs(video_folder, exist_ok=True)
        
        # Force rgb_array for rendering
        try:
             env = gym.make(env_name, render_mode="rgb_array")
        except Exception:
             # Fallback for some mujoco envs
             env = gym.make(env_name, render_mode="rgb_array", camera_id=0)

        env = RecordVideo(env, video_folder, episode_trigger=lambda x: True)
        
        # Run one episode
        obs, _ = env.reset(seed=seed + 12345)
        
        # Need to wrap obs for JAX
        # agent_state.sample_action expects (batch, obs_dim)
        
        done = False
        
        while not done:
            obs_jax = jnp.array(obs[None, :], dtype=jnp.float32)
            prng = jax.random.key(0) # deterministic anyway
            action, _ = agent_state.sample_action(obs_jax, prng, deterministic=True)
            
            # Convert back to numpy
            action_np = onp.array(action[0])
            
            obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            
        env.close()
        
        # Find the video file
        video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
        if video_files:
            video_path = os.path.join(video_folder, video_files[0])
            print(f"Saved final video locally to: {video_path}")
        else:
            print("No video file generated.")

    except Exception as e:
        # Don't let rendering failures crash training; just report them.
        print(f"Could not render final policy video: {e}", flush=True)

def save_checkpoint(agent_state: ppo.PpoState, run_tag: str, step: int, base_dir: str = "checkpoints"):
    """Save agent state checkpoint."""
    try:
        # Orbax requires absolute paths
        abs_base_dir = os.path.abspath(base_dir)
        # Folder naming: [task]_[method]_run[YYYYMMDD_HHMMSS]
        ckpt_dir = os.path.join(abs_base_dir, run_tag, str(step))
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        orbax_checkpointer.save(ckpt_dir, agent_state)
        print(f"Saved checkpoint to {ckpt_dir}")
    except Exception as e:
        print(f"Failed to save checkpoint: {e}")

def main(
    env_name: str,
    wandb_entity: str,
    wandb_project: str,
    config: ppo.PpoConfig | None = None,
    exp_name: str = "",
    seed: int = 0,
    num_timesteps: int | None = None,
    num_evals: int | None = None,
) -> None:
    """
    Train PPO using Gym environments (via GymWrapper).
    """
    
    # If no config is provided, use a generic PPO default similar to Brax.
    if config is None:
        config = ppo.PpoConfig(
            action_repeat=1,
            batch_size=256,
            discounting=0.99,
            entropy_cost=0.0,
            episode_length=1000,
            learning_rate=3e-4,
            normalize_observations=True,
            num_envs=16,
            num_evals=20,
            num_minibatches=4,
            # Default to ~300 outer iterations:
            # outer_iters = num_timesteps / (iterations_per_env * num_envs)
            # with these settings, 10_000_000 steps ≈ 305 iterations.
            num_timesteps=10_000_000,
            num_updates_per_batch=4,
            reward_scaling=1.0,
            unroll_length=32,
        )

    # Optional overrides for total timesteps / evals from CLI.
    if num_timesteps is not None:
        config = jdc.replace(config, num_timesteps=num_timesteps)
    if num_evals is not None:
        config = jdc.replace(config, num_evals=num_evals)

    # Logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Name components: [task] [method] [run][date & time]
    method = "ppo"
    run_tag = f"{env_name}_{method}_run{timestamp}"
    wandb_run = wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        name=f"ppo_gym_{env_name}_{exp_name}_{timestamp}",
        config={
            "env_name": env_name,
            "ppo_params": jdc.asdict(config),
            "learning_rate": config.learning_rate,
            "seed": seed,
        },
    )
    
    # Initialize Gym Environment
    print(f"Initializing Gym environment: {env_name}")
    gym_env = GymWrapper(env_name, num_envs=config.num_envs, seed=seed)
    print(f"Obs size: {gym_env.observation_size}, Action size: {gym_env.action_size}")
    
    # Initialize Agent State
    print("Initializing PPO Agent...")
    agent_state = ppo.PpoState.init(prng=jax.random.key(seed), env=gym_env, config=config)
    
    # Initialize Rollout State
    obs = gym_env.reset(seed=seed)
    rollout_state = GymRolloutState(
        last_obs=jnp.array(obs, dtype=jnp.float32),
        steps=jnp.zeros((config.num_envs,), dtype=jnp.int32),
        prng=jax.random.key(seed),
    )
    
    # Training Loop
    outer_iters = config.num_timesteps // (config.iterations_per_env * config.num_envs)
    eval_iters = set(onp.linspace(0, outer_iters - 1, config.num_evals, dtype=int))
    
    times = [time.time()]
    
    print(f"Starting training for {outer_iters} iterations...")
    
    for i in tqdm(range(outer_iters)):
        
        # Evaluation
        if i in eval_iters:
            # Use a separate eval env to be clean (avoid side effects on training buffers)
            # Create fewer envs for eval
            eval_num_envs = 10
            eval_env = GymWrapper(env_name, num_envs=eval_num_envs, seed=seed + 1000)
            
            eval_outputs = eval_policy_gym(
                eval_env,
                agent_state,
                prng=jax.random.fold_in(agent_state.prng, i),
                num_envs=eval_num_envs,
                max_episode_length=config.episode_length
            )
            
            eval_env.close()
            
            # Log metrics
            eval_outputs.log_to_wandb(wandb_run, step=i)
            
            # Print summary
            s_np = {k: onp.array(v) for k, v in eval_outputs.scalar_metrics.items()}
            print(f"Eval metrics at step {i}:")
            print(f"  Reward: mean={s_np['reward_mean']:.2f}")
        
        # Training Step
        # gym_rollout uses jax.pure_callback to step the gym environment
        rollout_state, transitions = gym_rollout(
            gym_env,
            rollout_state,
            agent_state,
            episode_length=config.episode_length,
            iterations_per_env=config.iterations_per_env,
            deterministic=False
        )
        
        # PPO update (Pure JAX)
        agent_state, metrics = agent_state.training_step(transitions)
        
        # Logging
        # transitions.reward has shape (T, N_envs).
        # For "total reward", we want per-env returns (sum over time) and then
        # average across parallel envs.
        per_env_returns = onp.sum(transitions.reward, axis=0)  # shape (N_envs,)
        log_dict = {
            # Average reward per step across all envs
            "train/mean_step_reward": float(onp.mean(transitions.reward)),
            # Average total return per env for this rollout window
            "train/mean_episode_return": float(onp.mean(per_env_returns)),
            **{f"train/{k}": float(onp.mean(v)) for k, v in metrics.items()},
        }
        wandb_run.log(log_dict, step=i)
        
        times.append(time.time())
        
    print("Training finished.")
    gym_env.close()
    
    # Save final checkpoint
    save_checkpoint(agent_state, run_tag, outer_iters)
    
    # Log final video to a similarly named folder
    print("Logging final video...")
    log_final_video(env_name, agent_state, wandb_run, config, seed, run_tag)
    
    # Finish wandb run to ensure all metrics are synced
    wandb.finish()

if __name__ == "__main__":
    tyro.cli(main, config=(tyro.conf.FlagConversionOff,))

