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

from flow_policy import fpo, rollouts
from flow_policy.gym_utils import GymWrapper, GymRolloutState, gym_rollout, eval_policy_gym


def log_final_video(
    env_name: str,
    agent_state: fpo.FpoState,
    wandb_run: wandb.sdk.wandb_run.Run,
    config: fpo.FpoConfig,
    seed: int,
    run_tag: str,
) -> None:
    """
    Run a short deterministic rollout with the trained policy and log a video to wandb.
    """
    try:
        # Create a single environment for recording
        # Folder naming: [task]_[method]_run[YYYYMMDD_HHMMSS]
        video_folder = os.path.join("videos", run_tag)
        os.makedirs(video_folder, exist_ok=True)
        
        # Force rgb_array for rendering
        # Try specifying camera_name if available for some envs, otherwise default
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
            # wandb_run.log(
            #     {"eval/final_policy_video": wandb.Video(video_path, fps=30, format="mp4")}
            # )
            print(f"Saved final video locally to: {video_path}")
        else:
            print("No video file generated.")

    except Exception as e:
        # Don't let rendering failures crash training; just report them.
        print(f"Could not render final policy video: {e}", flush=True)

def save_checkpoint(agent_state: fpo.FpoState, run_tag: str, step: int, base_dir: str = "checkpoints"):
    """Save agent state checkpoint."""
    try:
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
    config: fpo.FpoConfig,
    exp_name: str = "",
    seed: int = 0,
) -> None:
    """
    Train FPO Playground using Gym environments.
    """
    
    # Logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Name components: [task] [method] [run][date & time]
    method = "fpo"
    run_tag = f"{env_name}_{method}_run{timestamp}"
    wandb_run = wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        name=f"fpo_gym_{env_name}_{exp_name}_{timestamp}",
        config={
            "env_name": env_name,
            "fpo_params": jdc.asdict(config),
            "learning_rate": config.learning_rate,
            "seed": seed,
        },
    )
    
    # Initialize Gym Environment
    print(f"Initializing Gym environment: {env_name}")
    gym_env = GymWrapper(env_name, num_envs=config.num_envs, seed=seed)
    print(f"Obs size: {gym_env.observation_size}, Action size: {gym_env.action_size}")
    
    # Initialize Agent State
    # FpoState.init expects an env object with observation_size and action_size attributes
    print("Initializing FPO Agent...")
    agent_state = fpo.FpoState.init(prng=jax.random.key(seed), env=gym_env, config=config)
    
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
        
        # FPO update (Pure JAX)
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

    # Ensure Weights & Biases run is properly closed so that subsequent
    # calls (e.g., in batch scripts) start with a fresh step counter.
    wandb.finish()

if __name__ == "__main__":
    tyro.cli(main, config=(tyro.conf.FlagConversionOff,))
