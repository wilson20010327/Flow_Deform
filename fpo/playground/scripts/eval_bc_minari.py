import os
import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym
import gymnasium_robotics
import imageio
import pickle
import tyro
import jax_dataclasses as jdc
from typing import Tuple
from pathlib import Path
from tqdm import tqdm
from functools import partial

# Add src to path if needed
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from flow_policy import networks


def embed_timestep(t: jax.Array, embed_dim: int) -> jax.Array:
    """Embed (*, 1) timestep into (*, timestep_embed_dim)."""
    assert t.shape[-1] == 1
    freqs = 2 ** jnp.arange(embed_dim // 2)
    scaled_t = t * freqs
    out = jnp.concatenate([jnp.cos(scaled_t), jnp.sin(scaled_t)], axis=-1)
    return out


@partial(
    jax.jit,
    static_argnames=["action_dim", "flow_steps", "timestep_embed_dim", "policy_mlp_output_scale"],
)
def get_action(
    params,
    obs_norm,
    action_dim,
    flow_steps,
    timestep_embed_dim,
    policy_mlp_output_scale,
    rng,
):
    """Sample an action using Euler integration from t=1 (noise) to t=0 (data)."""
    batch_size = obs_norm.shape[0]
    n_samples = 1  # Deterministic evaluation, just 1 sample path per action

    # x0 ~ N(0, 1) (Noise)
    x = jax.random.normal(rng, (batch_size, n_samples, action_dim))

    dt = 1.0 / flow_steps

    for step in range(flow_steps):
        t_val = 1.0 - step / flow_steps
        t_arr = jnp.full((batch_size, n_samples, 1), t_val)
        t_embed = embed_timestep(t_arr, timestep_embed_dim)

        obs_expanded = jnp.broadcast_to(
            obs_norm[:, None, :], (batch_size, n_samples, obs_norm.shape[-1])
        )

        v = networks.flow_mlp_fwd(params, obs_expanded, x, t_embed) * policy_mlp_output_scale

        x = x - v * dt

    return jnp.squeeze(x, axis=1)  # (batch, action_dim)


def evaluate(
    checkpoint_path: str = "checkpoints/bc_pen/model_final.pkl",
    output_video_path: str = "videos/eval_video.mp4",
    seed: int = 0,
    max_steps: int = 200,  # Pen max_episode_steps
    num_episodes: int = 100,
    env_id: str = "AdroitHandPen-v1",
):
    # Resolve paths relative to project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[2]
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = str(project_root / checkpoint_path)
    if not os.path.isabs(output_video_path):
        output_video_path = str(project_root / output_video_path)

    print(f"Loading checkpoint from {checkpoint_path}...")
    with open(checkpoint_path, "rb") as f:
        ckpt = pickle.load(f)

    params = ckpt["policy_params"]
    obs_mean = ckpt["obs_mean"]
    obs_std = ckpt["obs_std"]
    config = ckpt["config"]

    print("Config loaded:", config)

    # Setup environment
    print(f"Creating environment {env_id}...")
    try:
        env = gym.make(
            env_id,
            render_mode="rgb_array",
            max_episode_steps=max_steps,
            reward_type="dense",
        )
    except gym.error.NameError:
        import gymnasium_robotics

        env = gym.make(
            env_id,
            render_mode="rgb_array",
            max_episode_steps=max_steps,
            reward_type="dense",
        )

    success_count = 0
    total_reward = 0.0

    # Store video frames only for the first episode
    frames = []

    # Per-episode stats
    returns = []
    steps_list = []

    print(f"Running evaluation for {num_episodes} episodes...")

    for ep in tqdm(range(num_episodes), desc="Evaluating"):
        # Seeding per episode
        rng = jax.random.key(seed + ep)
        obs_dict, _ = env.reset(seed=seed + ep)

        if isinstance(obs_dict, dict) and "observation" in obs_dict:
            obs = obs_dict["observation"]
        else:
            obs = obs_dict

        ep_ret = 0.0
        done = False

        for t in range(max_steps):
            # Render only first episode
            if ep == 0:
                frames.append(env.render())

            # Preprocess obs
            obs_norm = (obs - obs_mean) / (obs_std + 1e-8)
            obs_jax = jnp.array(obs_norm[None, :])  # Add batch dim

            # Sample action
            rng, step_rng = jax.random.split(rng)

            action = get_action(
                params,
                obs_jax,
                env.action_space.shape[0],
                config["flow_steps"],
                config["timestep_embed_dim"],
                config["policy_mlp_output_scale"],
                step_rng,
            )

            action_np = np.array(action[0])
            action_np = np.clip(action_np, -1.0, 1.0)

            obs_dict, reward, terminated, truncated, info = env.step(action_np)
            ep_ret += reward

            if isinstance(obs_dict, dict) and "observation" in obs_dict:
                obs = obs_dict["observation"]
            else:
                obs = obs_dict

            if terminated or truncated:
                break

        total_reward += ep_ret
        returns.append(ep_ret)
        steps_list.append(t + 1)

        # Check for success
        # Adroit environments typically have a 'success' or 'solved' key in info
        if "success" in info:
            if info["success"]:
                success_count += 1
        elif "solved" in info:
            if info["solved"]:
                success_count += 1

    returns_arr = np.array(returns, dtype=np.float32)
    steps_arr = np.array(steps_list, dtype=np.float32)

    avg_reward = float(returns_arr.mean())
    reward_min = float(returns_arr.min())
    reward_max = float(returns_arr.max())
    reward_std = float(returns_arr.std())

    steps_mean = float(steps_arr.mean())
    steps_min = float(steps_arr.min())
    steps_max = float(steps_arr.max())
    steps_std = float(steps_arr.std())

    success_rate = success_count / num_episodes

    metrics = {
        "reward_mean": avg_reward,
        "reward_min": reward_min,
        "reward_max": reward_max,
        "reward_std": reward_std,
        "steps_mean": steps_mean,
        "steps_min": steps_min,
        "steps_max": steps_max,
        "steps_std": steps_std,
        "success_rate": success_rate,
    }

    results_str = (
        f"\n{'=' * 50}\n"
        f"EVALUATION RESULTS ({num_episodes} episodes)\n"
        f"Average Reward: {avg_reward:.2f}\n"
        f"Reward Min / Max / Std: {reward_min:.2f} / {reward_max:.2f} / {reward_std:.2f}\n"
        f"Steps    Min / Max / Std: {steps_min:.1f} / {steps_max:.1f} / {steps_std:.1f}\n"
        f"Success Rate:   {success_rate * 100:.1f}%\n"
        f"{'=' * 50}\n"
    )

    # Print to console
    import sys as _sys

    print(results_str)
    _sys.stdout.flush()

    # Save to file
    with open("eval_results.txt", "w") as f:
        f.write(results_str)
    print("Saved results to eval_results.txt")

    print(f"Saving video of first episode to {output_video_path}...")
    imageio.mimsave(output_video_path, frames, fps=30)


if __name__ == "__main__":
    tyro.cli(evaluate)
