import jax_dataclasses as jdc

from mujoco_playground.config import dm_control_suite_params

from flow_policy import ppo as playground_ppo
from fpo.playground.scripts import train_ppo_gym

WANDB_ENTITY = "cz2874-columbia-university"
WANDB_PROJECT = "fpo_play_gym"


def make_brax_based_config(env_key: str) -> playground_ppo.PpoConfig:
    """
    Create a PPO config using Brax's tuned defaults for a given env key.

    This mirrors how train_ppo.py calls:
        ppo_params = dm_control_suite_params.brax_ppo_config(env_name)
        config = ppo.PpoConfig(**ppo_params)
    """
    ppo_params = dm_control_suite_params.brax_ppo_config(env_key)
    return playground_ppo.PpoConfig(**ppo_params)  # type: ignore[arg-type]


def with_outer_iters(
    config: playground_ppo.PpoConfig, target_outer_iters: int
) -> playground_ppo.PpoConfig:
    """
    Adjust num_timesteps so that:
        outer_iters = num_timesteps / (iterations_per_env * num_envs)
    equals `target_outer_iters`.
    """
    steps_per_outer = config.iterations_per_env * config.num_envs
    num_timesteps = int(steps_per_outer * target_outer_iters)
    return jdc.replace(config, num_timesteps=num_timesteps)


def run_walker_pendulum_and_reacher() -> None:
    """
    Run PPO Gym on:
      1) Pendulum-v1, using Brax's PendulumSwingup config
      2) Reacher-v4, using Brax's ReacherEasy config

    Each call passes an explicit config into train_ppo_gym.main, which
    overwrites that script's generic defaults.
    """

    # Target number of outer iterations per task.
    target_outer_iters = 150
    
    # 1) Pendulum-v1: use 'PendulumSwingup' Brax config as a good baseline,
    pendulum_config = with_outer_iters(
        make_brax_based_config("PendulumSwingup"), target_outer_iters
    )
    train_ppo_gym.main(
        env_name="Pendulum-v1",
        wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
        config=pendulum_config,
        exp_name="pendulum_brax_cfg",
        seed=42,
    )

    # 2) Reacher-v4: use 'ReacherEasy' Brax config as a good baseline,
    reacher_config = with_outer_iters(
        make_brax_based_config("ReacherEasy"), target_outer_iters
    )
    train_ppo_gym.main(
        env_name="Reacher-v4",
        wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
        config=reacher_config,
        exp_name="reacher_brax_cfg",
        seed=42,
    )

if __name__ == "__main__":
    run_walker_pendulum_and_reacher()


