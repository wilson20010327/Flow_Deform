"""
        This file is the executable for running PPO. It is based on this medium article: 
        https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8
"""

import gymnasium as gym
import sys,random
from typing import Optional
import torch
import wandb
import jax_dataclasses as jdc
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from wandb.integration.sb3 import WandbCallback
from datetime import datetime
import time, sys
import pathlib as Path
import numpy as np
# from fpo.gridworld.models.sac import SAC
from fpo.gridworld.models.sac_test import SAC_self
from fpo.gridworld.utils.arguments import get_args
from fpo.gridworld.models.ppo import PPO
from fpo.gridworld.models.fpo_gym import FPO
from fpo.gridworld.models.network import FeedForwardNN
from fpo.gridworld.models.diffusion_policy import DiffusionPolicy
from fpo.gridworld.utils.eval_policy import eval_policy
from fpo.gridworld.utils.gridworld import GridWorldEnv

# JAX FPO playground training 
from flow_policy import fpo as playground_fpo
from flow_policy import ppo as playground_ppo
from fpo.playground.scripts import train_fpo as playground_train_fpo
from fpo.playground.scripts import train_fpo_gym as playground_train_fpo_gym
from fpo.playground.scripts import train_ppo_gym as playground_train_ppo_gym


sys.path.append(str(Path.Path(__file__).resolve().parent))

from module_fsac.env_wrapper import NormalizeObsActWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def run_playground_fpo(
        env_name: str,
        wandb_entity: str,
        wandb_project: str,
        exp_name: str = "",
        seed: int = 0,
        config: Optional[playground_fpo.FpoConfig] = None,
) -> None:
        if config is None:
                config = playground_fpo.FpoConfig()

        playground_train_fpo.main(
                env_name=env_name,
                wandb_entity=wandb_entity,
                wandb_project=wandb_project,
                config=config,
                exp_name=exp_name,
                seed=seed,
        )


def run_playground_fpo_gym(
        env_name: str,
        wandb_entity: str,
        wandb_project: str,
        exp_name: str = "",
        seed: int = 0,
        config: Optional[playground_fpo.FpoConfig] = None,
        num_timesteps: int = None,
        num_evals: int = None,
) -> None:
        if config is None:
                config = playground_fpo.FpoConfig()
        

        if num_timesteps is not None:
             config = jdc.replace(config, num_timesteps=num_timesteps)
        if num_evals is not None:
             config = jdc.replace(config, num_evals=num_evals)

        playground_train_fpo_gym.main(
                env_name=env_name,
                wandb_entity=wandb_entity,
                wandb_project=wandb_project,
                config=config,
                exp_name=exp_name,
                seed=seed,
        )

def run_playground_ppo_gym(
        env_name: str,
        wandb_entity: str,
        wandb_project: str,
        exp_name: str = "",
        seed: int = 0,
        config: Optional[playground_ppo.PpoConfig] = None,
        num_timesteps: int = None,
        num_evals: int = None,
) -> None:
        if config is None:
                # Use brax defaults as a base if no config provided
                # We need to import params
                from mujoco_playground.config import dm_control_suite_params
                # NOTE: This might fail if env_name isn't in Brax config.
                # Fallback to manual defaults if needed.
                try:
                    ppo_params = dm_control_suite_params.brax_ppo_config(env_name)
                    config = playground_ppo.PpoConfig(**ppo_params)
                except Exception:
                    print("Warning: Could not load default PPO config for this env. Using generic defaults.")
                    # Create a generic default config if needed (omitted for brevity, assuming Brax config usually works or user provides one)
                    # For now, we'll just let it fail or rely on provided config.
                    pass

        # Override config if args provided
        if num_timesteps is not None:
             config = jdc.replace(config, num_timesteps=num_timesteps)
        if num_evals is not None:
             config = jdc.replace(config, num_evals=num_evals)

        playground_train_ppo_gym.main(
                env_name=env_name,
                wandb_entity=wandb_entity,
                wandb_project=wandb_project,
                config=config,
                exp_name=exp_name,
                seed=seed,
        )


class CustomSACWandbLogger(BaseCallback):
    """
    Custom logger for SAC training — logs episodic stats to wandb.
    """
    def __init__(self, log_interval=10000, verbose=1):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.last_time = time.time()
        self.last_timesteps = 0

        # Running buffers
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Record rewards if the env provides them
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])

        # Every N timesteps, compute stats and log
        if (self.num_timesteps - self.last_timesteps) >= self.log_interval:
            delta_t = time.time() - self.last_time
            avg_ep_rews = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0.0
            avg_ep_lens = np.mean(self.episode_lengths[-10:]) if self.episode_lengths else 0.0

            # SB3 stores losses in the logger (accessible via self.model.logger)
            # But we can’t access directly per step — so just read last known value
            try:
                last_log = self.model.logger.name_to_value
                avg_actor_loss = float(last_log.get("train/actor_loss", 0.0))
            except Exception:
                avg_actor_loss = 0.0

            i_so_far = int(self.num_timesteps // self.log_interval)
            t_so_far = int(self.num_timesteps)

            print("\n-------------------- Iteration #{} --------------------".format(i_so_far), flush=True)
            print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
            print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
            print(f"Average Actor Loss: {avg_actor_loss}", flush=True)
            print(f"Timesteps So Far: {t_so_far}", flush=True)
            print(f"Iteration took: {delta_t:.2f} secs", flush=True)
            print("------------------------------------------------------\n", flush=True)

            # Log to wandb
            wandb.log({
                "iteration": i_so_far,
                "timesteps_so_far": t_so_far,
                "avg_episode_length": float(avg_ep_lens),
                "avg_episode_return": float(avg_ep_rews),
                "avg_actor_loss": float(avg_actor_loss),
                "iteration_duration_sec": float(delta_t),
            })

            # Update timers
            self.last_timesteps = self.num_timesteps
            self.last_time = time.time()

        return True

def set_seed(env, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        try:
                env.observation_space.seed(seed)
        except Exception:
                pass
def train(env, eval_env, hyperparameters, actor_model, critic_model, method):
        """
                Trains the model.

                Parameters:
                        env - the environment to train on
                        hyperparameters - a dict of hyperparameters to use, defined in main
                        actor_model - the actor model to load in if we want to continue training
                        critic_model - the critic model to load in if we want to continue training

                Return:
                        None
        """     
        print(f"Training using {method.upper()}", flush=True)

        if method == "ppo":
                model = PPO(policy_class=FeedForwardNN, env=env, eval_env=eval_env, **hyperparameters)
        elif method == "fpo":
                model = FPO(actor_class=DiffusionPolicy, critic_class=FeedForwardNN, env=env, eval_env=eval_env, **hyperparameters)
        elif method == "sac_self":
                model = SAC_self(policy_class=FeedForwardNN, env=env, eval_env=eval_env, **hyperparameters)
        elif method == "sac":
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = f"SAC_lr{hyperparameters['lr']}_{timestamp}"
                model = SAC(
                    "MlpPolicy",
                    env,
                #     eval_env=eval_env,
                    verbose=1,
                    learning_rate=hyperparameters["lr"],
                    seed=42,
                    gamma=hyperparameters["gamma"],
                    batch_size=hyperparameters["timesteps_per_batch"],
                )
        else:
                print(f"Unsupported method: {method}")
                sys.exit(1)
        if "sac" not in method:
            model.actor.to(device)
            model.critic.to(device)
        
        # Tries to load in an existing actor/critic model to continue training on
        if actor_model != '' and critic_model != '':
                print(f"Loading in {actor_model} and {critic_model}...", flush=True)
                model.actor.load_state_dict(torch.load(actor_model, map_location=device))
                model.critic.load_state_dict(torch.load(critic_model, map_location=device))
                print(f"Successfully loaded.", flush=True)
        elif actor_model != '' or critic_model != '': # Don't train from scratch if user accidentally forgets actor/critic model
                print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
                sys.exit(0)
        else:
                print(f"Training from scratch.", flush=True)

        # Train the PPO model with a specified total timesteps
        # NOTE: You can change the total timesteps here, I put a big number just because
        # you can kill the process whenever you feel like PPO is converging
        if method != "sac":
            model.learn(total_timesteps=200_000_000)
        else:
            custom_logger = CustomSACWandbLogger(log_interval=5000)
            model.learn(total_timesteps=200_000, callback=custom_logger)
            model.save(f"{run_name}_final")
            
def test(env, actor_model, method):
        """
                Tests the model.

                Parameters:
                        env - the environment to test the policy on
                        actor_model - the actor model to load in

                Return:
                        None
        """
        print(f"Testing {actor_model}", flush=True)

        # If the actor model is not specified, then exit
        if actor_model == '':
                print(f"Didn't specify model file. Exiting.", flush=True)
                sys.exit(0)

        # Extract out dimensions of observation and action spaces
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        if method == 'ppo':
                policy = FeedForwardNN(obs_dim, act_dim).to(device)
        elif method == 'fpo':
                policy = DiffusionPolicy(obs_dim + act_dim + 1, act_dim).to(device)                
        else:
                print(f"Unsupported method: {method}")
                sys.exit(1)

        # Load in the actor model saved by the PPO algorithm
        policy.load_state_dict(torch.load(actor_model, map_location=device))

        # Evaluate our policy with a separate module, eval_policy, to demonstrate
        # that once we are done training the model/policy with ppo.py, we no longer need
        # ppo.py since it only contains the training algorithm. The model/policy itself exists
        # independently as a binary file that can be loaded in with torch.
        if method == 'fpo':
                eval_policy(policy=policy.sample_action, env=env, render=True)
        else:
                eval_policy(policy=policy, env=env, render=True)

def main(args):
        """
                The main function to run.

                Parameters:
                        args - the arguments parsed from command line

                Return:
                        None
        """
        # Run JAX FPO playground training
        if args.method == 'fpo_pg':
                if not args.wandb_entity or not args.wandb_project:
                        print("For method=fpo_pg you must provide --wandb_entity and --wandb_project.", flush=True)
                        sys.exit(1)

                print(f"Running JAX FPO playground with env='{args.pg_env_name}'", flush=True)
                run_playground_fpo(
                        env_name=args.pg_env_name,
                        wandb_entity=args.wandb_entity,
                        wandb_project=args.wandb_project,
                )
                return

        # Run JAX FPO playground training on Gym
        if args.method == 'fpo_pg_gym':
                if not args.wandb_entity or not args.wandb_project:
                        print("For method=fpo_pg_gym you must provide --wandb_entity and --wandb_project.", flush=True)
                        sys.exit(1)

                print(f"Running JAX FPO playground (Gym backend) with env='{args.pg_env_name}'", flush=True)
                run_playground_fpo_gym(
                        env_name=args.pg_env_name,
                        wandb_entity=args.wandb_entity,
                        wandb_project=args.wandb_project,
                        num_timesteps=args.num_timesteps,
                        num_evals=args.num_evals,
                )
                return

        # Run JAX PPO playground training on Gym
        if args.method == 'ppo_pg_gym':
                if not args.wandb_entity or not args.wandb_project:
                        print("For method=ppo_pg_gym you must provide --wandb_entity and --wandb_project.", flush=True)
                        sys.exit(1)

                print(f"Running JAX PPO playground (Gym backend) with env='{args.pg_env_name}'", flush=True)
                
                # Load default config here to pass to runner
                from mujoco_playground.config import dm_control_suite_params
                # We use a proxy env name if the real one isn't in Brax's list, or just try 'cartpole_swingup' as base
                try:
                    ppo_params = dm_control_suite_params.brax_ppo_config(args.pg_env_name)
                except KeyError:
                    print(f"Env {args.pg_env_name} not found in Brax defaults. Using 'cartpole_swingup' defaults as base.")
                    ppo_params = dm_control_suite_params.brax_ppo_config("cartpole_swingup")
                
                config = playground_ppo.PpoConfig(**ppo_params)

                run_playground_ppo_gym(
                        env_name=args.pg_env_name,
                        wandb_entity=args.wandb_entity,
                        wandb_project=args.wandb_project,
                        config=config,
                        num_timesteps=args.num_timesteps,
                        num_evals=args.num_evals,
                )
                return

        # NOTE: Here's where you can set default hyperparameters. I don't include them as part of
        # ArgumentParser because it's too annoying to type them every time at command line. Instead, you can change them here.
        # To see a list of hyperparameters, look in ppo.py at function _init_hyperparameters
        hyperparameters = {
                                'timesteps_per_batch': 2048,
                                'max_timesteps_per_episode': 200, 
                                'gamma': 0.99, 
                                'n_updates_per_iteration': 10,
                                'lr': 3e-4, 
                                'clip': 0.2,
                                'render': True,
                                'render_every_i': 1000,
                                # FPO specific parameters:
                                'grid_mode': 'two_walls',
                                'num_fpo_samples': 50,
                                'positive_advantage': False,
                          }

        print(hyperparameters)
        env_name = 'Pendulum-v1'
        # Creates the environment we'll be running. If you want to replace with your own
        # custom environment, note that it must inherit Gym and have both continuous
        # observation and action spaces.
        # env = gym.make('Pendulum-v1', render_mode='human' if args.mode == 'test' else 'rgb_array')
        # env = GridWorldEnv(mode=hyperparameters['grid_mode'])
        env = NormalizeObsActWrapper(gym.make(env_name, render_mode='human' if args.mode == 'test' else 'rgb_array'))
        set_seed(env, 42)
        eval_env = NormalizeObsActWrapper(gym.make(env_name, render_mode='human' if args.mode == 'test' else 'rgb_array'))
        set_seed(eval_env, 42+100)
        # Train or test, depending on the mode specified
        if args.mode == 'train':
                # Run name for wandb
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                lr = hyperparameters['lr']
                bs = hyperparameters['timesteps_per_batch']

                run_name = f"{args.method}_lr{lr}_bs{bs}_{timestamp}"
                
                # Add extra tag for FPO
                if args.method == "fpo":
                        n = hyperparameters['num_fpo_samples']
                        run_name += f"_N{n}"
                
                print(f"running {run_name}")

                hyperparameters["run_name"] = run_name
                
                wandb.init(
                        project="fpo-diffusion-grid",
                        name=run_name,
                        config=hyperparameters,
                        tags=[args.method, env_name, args.mode]
                )

                # If running under a W&B Sweep, override hyperparameters with wandb.config
                try:
                        cfg = dict(wandb.config)
                        # keep run_name stable even if config changes
                        cfg["run_name"] = run_name
                        # merge only known keys to avoid surprises
                        for k in hyperparameters.keys():
                                if k in cfg:
                                        hyperparameters[k] = cfg[k]
                except Exception:
                        pass

                train(env=env, eval_env=eval_env, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model, method=args.method)

                # Cleanly close this Weights & Biases run so that subsequent runs
                # (e.g., in batch scripts) start with a fresh step counter.
                if wandb.run is not None:
                        wandb.finish()
        else:
                test(env=env, actor_model=args.actor_model, method=args.method)

if __name__ == '__main__':
        args = get_args() # Parse arguments from command line
        main(args)
