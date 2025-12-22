import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import jax_dataclasses as jdc
from typing import Any, Tuple, Callable
from flow_policy import rollouts, fpo

class GymWrapper:
    def __init__(self, env_name: str, num_envs: int, seed: int = 0):
        def make_env():
            # Try standard Gym first; if the env is from gymnasium_robotics
            # (e.g. AdroitHandPen-v1), import it to register the envs.
            try:
                env = gym.make(env_name)
            except (gym.error.NameNotFound, gym.error.Error):
                # Importing gymnasium_robotics has side effects that register
                # robotics / Adroit environments with Gym.
                import gymnasium_robotics  # noqa: F401
                env = gym.make(env_name)

            # If observations are dicts (e.g. Adroit / robotics envs), extract
            # the low-dimensional 'observation' component so downstream code
            # always sees a flat vector.
            if isinstance(env.observation_space, gym.spaces.Dict) and "observation" in env.observation_space.spaces:

                class ExtractObsWrapper(gym.ObservationWrapper):
                    def __init__(self, env):
                        super().__init__(env)
                        self.observation_space = env.observation_space["observation"]

                    def observation(self, observation):
                        return observation["observation"]

                env = ExtractObsWrapper(env)

            env.reset(seed=seed)
            return env

        self.env = gym.vector.SyncVectorEnv([make_env for _ in range(num_envs)])
        
        self.observation_size = self.env.single_observation_space.shape[0]
        self.action_size = self.env.single_action_space.shape[0]
        self.num_envs = num_envs
        self.seed = seed

    def reset(self, seed: int = None):
        obs, info = self.env.reset(seed=seed if seed is not None else self.seed)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        buffer_obs = obs.copy()
        
        if "final_observation" in info:
            final_obs = info["final_observation"]
            for i in range(self.num_envs):
                if (terminated[i] or truncated[i]) and final_obs[i] is not None:
                    buffer_obs[i] = final_obs[i]
                    
        return obs, buffer_obs, reward, terminated, truncated

    def close(self):
        self.env.close()

@jdc.pytree_dataclass
class GymRolloutState:
    last_obs: jax.Array
    steps: jax.Array 
    prng: jax.Array

def gym_rollout(
    gym_env: GymWrapper,
    rollout_state: GymRolloutState,
    agent_state: fpo.FpoState, # or AgentState protocol
    episode_length: int,
    iterations_per_env: int,
    deterministic: bool = False,
) -> Tuple[GymRolloutState, rollouts.TransitionStruct]:
    
    def env_step_callback(action):
        obs, buffer_obs, reward, terminated, truncated = gym_env.step(np.array(action))
        return (
            obs.astype(np.float32),
            buffer_obs.astype(np.float32),
            reward.astype(np.float32),
            terminated.astype(np.bool_),
            truncated.astype(np.bool_)
        )

    def scan_step(carry: GymRolloutState, _):
        state = carry
        
        prng_act, prng_next = jax.random.split(state.prng)
        action, action_info = agent_state.sample_action(
            state.last_obs, prng_act, deterministic=deterministic
        )
        
        result_shape_dtypes = (
            jax.ShapeDtypeStruct(state.last_obs.shape, jnp.float32), 
            jax.ShapeDtypeStruct(state.last_obs.shape, jnp.float32), 
            jax.ShapeDtypeStruct((gym_env.num_envs,), jnp.float32),  
            jax.ShapeDtypeStruct((gym_env.num_envs,), jnp.bool_),    
            jax.ShapeDtypeStruct((gym_env.num_envs,), jnp.bool_),    
        )
        
        next_obs, buffer_obs, reward, terminated, truncated = jax.pure_callback(
            env_step_callback,
            result_shape_dtypes,
            action
        )
        
        discount = 1.0 - terminated.astype(jnp.float32)
        
        transition = rollouts.TransitionStruct(
            obs=state.last_obs,
            next_obs=buffer_obs,
            action=action,
            action_info=action_info,
            reward=reward,
            truncation=truncated.astype(jnp.float32),
            discount=discount,
        )
        
        new_state = jdc.replace(
            state,
            last_obs=next_obs,
            steps=state.steps + 1, 
            prng=prng_next
        )
        
        return new_state, transition

    final_state, transitions = jax.lax.scan(
        scan_step,
        rollout_state,
        (),
        length=iterations_per_env
    )
    
    return final_state, transitions

def eval_policy_gym(
    gym_env: GymWrapper,
    agent_state: fpo.FpoState,
    prng: jax.Array,
    num_envs: int,
    max_episode_length: int,
) -> rollouts.EvalOutputs:
    obs = gym_env.reset(seed=int(jax.random.randint(prng, (), 0, 100000)))
    
    rollout_state = GymRolloutState(
        last_obs=jnp.array(obs, dtype=jnp.float32),
        steps=jnp.zeros((num_envs,), dtype=jnp.int32),
        prng=prng,
    )
    
    _, transitions = gym_rollout(
        gym_env, 
        rollout_state, 
        agent_state, 
        episode_length=max_episode_length, 
        iterations_per_env=max_episode_length, 
        deterministic=True
    )
    
    is_done = jnp.logical_or(transitions.truncation > 0, transitions.discount == 0) # (T, B)
    done_cumsum = jnp.cumsum(is_done.astype(jnp.int32), axis=0)
    
    valid_mask = (done_cumsum - is_done.astype(jnp.int32)) == 0
    valid_mask = valid_mask.astype(jnp.float32)
    
    rewards = jnp.sum(transitions.reward * valid_mask, axis=0)
    steps = jnp.sum(valid_mask, axis=0)
    
    scalar_metrics = {
        "reward_mean": jnp.mean(rewards),
        "reward_min": jnp.min(rewards),
        "reward_max": jnp.max(rewards),
        "reward_std": jnp.std(rewards),
        "steps_mean": jnp.mean(steps),
        "steps_min": jnp.min(steps),
        "steps_max": jnp.max(steps),
        "steps_std": jnp.std(steps),
    }
    
    histogram_metrics = {
        "reward": rewards.flatten(),
        "steps": steps.flatten(),
    }
    
    return rollouts.EvalOutputs(
        scalar_metrics=scalar_metrics,
        histogram_metrics=histogram_metrics,
        actions=transitions.action,
        action_timestep_mask=valid_mask,
    )
