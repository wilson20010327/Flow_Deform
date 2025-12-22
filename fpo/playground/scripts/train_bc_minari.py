import os
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
import wandb
import minari
import pickle
import jax_dataclasses as jdc
from typing import Literal, Tuple
from tqdm import tqdm
from functools import partial

# Add src to path if needed, though usually installed in editable mode
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from flow_policy import networks

@jdc.pytree_dataclass
class BCConfig:
    dataset_id: str = "D4RL/pen/cloned-v2"
    
    # Flow parameters
    flow_steps: jdc.Static[int] = 20
    timestep_embed_dim: jdc.Static[int] = 8
    n_samples_per_action: jdc.Static[int] = 32
    policy_mlp_output_scale: float = 0.25
    
    # Training
    batch_size: int = 256
    learning_rate: float = 3e-4
    num_epochs: int = 500
    seed: int = 0
    
    # Normalization
    normalize_observations: bool = True
    
    # Output
    output_dir: str = "checkpoints/bc_pen"
    save_interval: int = 10
    wandb_project: str = "flow-bc-minari"
    wandb_entity: str = ""

@jdc.pytree_dataclass
class BCState:
    params: networks.MlpWeights
    opt: jdc.Static[optax.GradientTransformation]
    opt_state: optax.OptState
    rng: jax.Array
    
    # Store statistics for saving/loading compat with FPO
    obs_mean: jax.Array
    obs_std: jax.Array

def embed_timestep(t: jax.Array, embed_dim: int) -> jax.Array:
    """Embed (*, 1) timestep into (*, timestep_embed_dim)."""
    assert t.shape[-1] == 1
    freqs = 2 ** jnp.arange(embed_dim // 2)
    scaled_t = t * freqs
    out = jnp.concatenate([jnp.cos(scaled_t), jnp.sin(scaled_t)], axis=-1)
    return out

def compute_cfm_loss(
    params: networks.MlpWeights,
    obs: jax.Array,
    action: jax.Array,
    rng: jax.Array,
    config: BCConfig,
) -> jax.Array:
    """Compute Conditional Flow Matching loss."""
    
    batch_size, action_dim = action.shape
    samples_dim = config.n_samples_per_action
    obs_dim = obs.shape[-1]
    
    # Expand for samples
    # obs: (B, O) -> (B, S, O)
    obs_expanded = jnp.broadcast_to(obs[:, None, :], (batch_size, samples_dim, obs_dim))
    # action: (B, A) -> (B, S, A)
    x1 = jnp.broadcast_to(action[:, None, :], (batch_size, samples_dim, action_dim))
    
    rng_eps, rng_t = jax.random.split(rng)
    
    # Sample noise (x0) and time (t)
    x0 = jax.random.normal(rng_eps, (batch_size, samples_dim, action_dim))
    t = jax.random.uniform(rng_t, (batch_size, samples_dim, 1))
    
    eps = x0 # x0 in my notation above was noise
    x_t = t * eps + (1.0 - t) * x1
    
    # Predict vector field
    # FPO predicts velocity. 
    # network inputs: obs, x_t, embedded_t
    
    t_embed = embed_timestep(t, config.timestep_embed_dim)
    
    network_pred = (
        networks.flow_mlp_fwd(
            params,
            obs_expanded,
            x_t,
            t_embed,
        )
        * config.policy_mlp_output_scale
    )
    
    # Target velocity
    # If x_t = t * eps + (1-t) * action
    # dx_t/dt = eps - action
    velocity_target = eps - x1
    
    # Loss: MSE(pred, target)
    loss = jnp.mean((network_pred - velocity_target) ** 2)
    
    return loss

@partial(jax.jit, static_argnames=["config"])
def train_step(state: BCState, batch_obs: jax.Array, batch_actions: jax.Array, config: BCConfig):
    
    # Normalize obs
    if config.normalize_observations:
        obs_norm = (batch_obs - state.obs_mean) / (state.obs_std + 1e-8)
    else:
        obs_norm = batch_obs
        
    def loss_fn(params):
        return compute_cfm_loss(params, obs_norm, batch_actions, state.rng, config)
        
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    
    updates, new_opt_state = state.opt.update(grads, state.opt_state)
    new_params = optax.apply_updates(state.params, updates)
    
    new_rng, _ = jax.random.split(state.rng)
    
    new_state = jdc.replace(
        state,
        params=new_params,
        opt_state=new_opt_state,
        rng=new_rng
    )
    
    return new_state, loss

def load_minari_data(dataset_id: str):
    print(f"Loading Minari dataset: {dataset_id}")
    dataset = minari.load_dataset(dataset_id)
    
    observations = []
    actions = []
    
    for episode in tqdm(dataset.iterate_episodes(), desc="Loading episodes"):
        obs = episode.observations
        
        # Handle dict observations
        if isinstance(obs, dict):
             if 'observation' in obs:
                 observations.append(obs['observation'][:-1])
             else:
                 raise ValueError(f"Expected 'observation' key in dict observations, found: {obs.keys()}")
        else:
             observations.append(episode.observations[:-1])
             
        actions.append(episode.actions)
        
    observations = np.concatenate(observations, axis=0)
    actions = np.concatenate(actions, axis=0)
    
    print(f"Loaded {len(observations)} transitions.")
    return observations, actions

def main(config: BCConfig):
    # Setup WandB
    if config.wandb_project:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            config=jdc.asdict(config),
            name=f"bc_{config.dataset_id}"
        )

    # Load Data
    obs_data, action_data = load_minari_data(config.dataset_id)
    
    # Compute stats
    obs_mean = jnp.array(np.mean(obs_data, axis=0))
    obs_std = jnp.array(np.std(obs_data, axis=0))
    
    # Init State
    rng = jax.random.key(config.seed)
    rng_init, rng_state = jax.random.split(rng)
    
    obs_dim = obs_data.shape[1]
    action_dim = action_data.shape[1]
    
    # Policy architecture matching FPO
    policy_dims = (
        obs_dim + action_dim + config.timestep_embed_dim,
        256, 256, 256, 256,
        action_dim
    )
    
    policy_params = networks.mlp_init(rng_init, policy_dims)
    
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(policy_params)
    
    state = BCState(
        params=policy_params,
        opt=optimizer,
        opt_state=opt_state,
        rng=rng_state,
        obs_mean=obs_mean,
        obs_std=obs_std
    )
    
    # Training Loop
    num_samples = len(obs_data)
    steps_per_epoch = num_samples // config.batch_size
    
    obs_jax = jax.device_put(obs_data)
    action_jax = jax.device_put(action_data)
    
    for epoch in range(config.num_epochs):
        perm = np.random.permutation(num_samples)
        epoch_losses = []
        
        with tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}") as pbar:
            for i in pbar:
                idx = perm[i * config.batch_size : (i + 1) * config.batch_size]
                batch_obs = obs_jax[idx]
                batch_actions = action_jax[idx]
                
                state, loss = train_step(state, batch_obs, batch_actions, config)
                epoch_losses.append(loss)
                
                if i % 10 == 0:
                    pbar.set_postfix(loss=float(loss))
                    if config.wandb_project:
                        wandb.log({"train/loss": float(loss)})
        
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}")
        
        if config.wandb_project and (epoch % config.save_interval == 0 or epoch == config.num_epochs - 1):
            # Simple Action MSE logging at the end
            # Use the first batch for a quick check
            # Sample from flow:
            # x0 ~ N(0,1), integrate to x1
            
            # Use deterministic sampling for eval
            eval_rng = jax.random.key(0)
            batch_size_eval = 64
            eval_obs = obs_jax[:batch_size_eval]
            eval_actions_gt = action_jax[:batch_size_eval]
            
            # Normalize obs
            if config.normalize_observations:
                eval_obs_norm = (eval_obs - state.obs_mean) / (state.obs_std + 1e-8)
            else:
                eval_obs_norm = eval_obs
                
            # Define Euler solver
            def solve_flow(obs, rng):
                dt = 1.0 / config.flow_steps
                x = jax.random.normal(rng, (obs.shape[0], config.n_samples_per_action, action_dim))

                
                for step in range(config.flow_steps):
                    t_val = 1.0 - step / config.flow_steps
                    t_arr = jnp.full((obs.shape[0], config.n_samples_per_action, 1), t_val)
                    t_embed = embed_timestep(t_arr, config.timestep_embed_dim)
                    
                    obs_expanded = jnp.broadcast_to(obs[:, None, :], (obs.shape[0], config.n_samples_per_action, obs.shape[-1]))
                    
                    v = networks.flow_mlp_fwd(state.params, obs_expanded, x, t_embed) * config.policy_mlp_output_scale
                    x = x - v * dt
                
                return x

            pred_actions = solve_flow(eval_obs_norm, eval_rng)
            
            # Compare with GT (broadcast GT)
            # Take mean over samples? Or best sample? Mean is fine for MSE.
            pred_actions_mean = jnp.mean(pred_actions, axis=1)
            # Broadcast GT to match samples shape if necessary, but here we average preds
            mse = jnp.mean((pred_actions_mean - eval_actions_gt) ** 2)
            wandb.log({"eval/action_mse": float(mse)})
            print(f"Epoch {epoch} Action MSE: {float(mse)}")
            
        # Checkpointing
        if config.output_dir and epoch == config.num_epochs - 1:
            os.makedirs(config.output_dir, exist_ok=True)
            save_path = os.path.join(config.output_dir, f"model_final.pkl")
            
            # Save params and stats
            checkpoint = {
                "policy_params": state.params,
                "obs_mean": state.obs_mean,
                "obs_std": state.obs_std,
                "config": jdc.asdict(config)
            }
            with open(save_path, "wb") as f:
                pickle.dump(checkpoint, f)
            print(f"Saved checkpoint to {save_path}")

    if config.wandb_project:
        wandb.finish()

if __name__ == "__main__":
    tyro.cli(main)

