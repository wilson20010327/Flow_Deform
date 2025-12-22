import jax_dataclasses as jdc

from flow_policy import fpo as playground_fpo
from fpo.playground.scripts import train_fpo_gym


WANDB_ENTITY = "cz2874-columbia-university"
WANDB_PROJECT = "fpo_play_gym"


def with_outer_iters(
    config: playground_fpo.FpoConfig, target_outer_iters: int
) -> playground_fpo.FpoConfig:
    """
    Adjust num_timesteps so that:
        outer_iters = num_timesteps / (iterations_per_env * num_envs)
    equals `target_outer_iters`.

    This keeps all other FPO hyperparameters at their defaults.
    """
    steps_per_outer = config.iterations_per_env * config.num_envs
    num_timesteps = int(steps_per_outer * target_outer_iters)
    return jdc.replace(config, num_timesteps=num_timesteps)


def run_fpo_on_gym_tasks() -> None:
    """
    Run FPO (with default config, except for num_timesteps) on three Gym tasks:
      1) BipedalWalker-v3
      2) Pendulum-v1
      3) Reacher-v4

    Each task is trained for exactly `target_outer_iters` outer iterations.
    """

    target_outer_iters = 150
    tasks = [
        "Pendulum-v1",
        "Reacher-v4",
    ]

    for env_name in tasks:
        # Start from FPO's default config and only adjust num_timesteps.
        base_config = playground_fpo.FpoConfig()
        config = with_outer_iters(base_config, target_outer_iters)

        train_fpo_gym.main(
            env_name=env_name,
            wandb_entity=WANDB_ENTITY,
            wandb_project=WANDB_PROJECT,
            config=config,
            exp_name=f"fpo_default_{env_name}_150iters",
            seed=42,
        )


if __name__ == "__main__":
    run_fpo_on_gym_tasks()


