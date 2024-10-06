"""n-scale potential example."""

import os
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from examples.utils.sde import SDEIntegrator
from examples.utils.data import shrink_trajectory_len

import hydra
import logging
from omegaconf import DictConfig
from datasets import Dataset, Features, Array2D

from onsagernet.dynamics import OnsagerNetFD, SDEfromFunc
from onsagernet.models import PotentialMLP, DissipationMatrixMLP, ConservationMatrixMLP
from onsagernet.trainers import MLETrainer

# ------------------------- Typing imports ------------------------- #

from jax import Array
from jax.typing import ArrayLike


def V_homogenised(x: ArrayLike) -> Array:
    """Homogenised potential for the n-scale potential

    Args:
        x (ArrayLike): state

    Returns:
        Array: potential
    """
    effective_potential = 0.5 * jnp.log(1 + jnp.sin(jnp.pi * x) ** 2)
    return jnp.squeeze(x**2 + effective_potential)


def M_homogenised(x: ArrayLike) -> Array:
    """The homogenised dissipation matrix for the n-scale potential

    Args:
        x (ArrayLike): state

    Returns:
        Array: dissipation matrix
    """
    numerator = jnp.sqrt(1 + jnp.sin(jnp.pi * x) ** 2)
    denominator = 1 + 0.5 * jnp.sin(jnp.pi * x) ** 2
    return numerator / denominator


def build_targets(config: DictConfig) -> SDEfromFunc:
    """Build the target dynamics for the n-scale potential

    Args:
        config (DictConfig): configuration object

    Returns:
        SDEfromFunc: the target dynamics
    """

    def v_0(x):
        return x**2

    def v_1(x, y):
        x_term = jnp.sin(jnp.pi * x) ** 2
        y_term = jnp.sin(jnp.pi * y) ** 2
        return jnp.log(1 + x_term * y_term)

    def v(x):
        return v_0(x[0]) + v_1(x[0], x[1])

    def target_drift(t, x, args):
        M = jnp.array(
            [
                [1, 1 / config.data.eps],
                [1 / config.data.eps, 1 / config.data.eps**2],
            ]
        )
        dV = jax.grad(v)
        return -M @ dV(x)

    def target_diffusion(t, x, args):
        temperature = args[0]
        return jnp.array(
            [
                [jnp.sqrt(2 * temperature)],
                [jnp.sqrt(2 * temperature) / config.data.eps],
            ]
        )

    return SDEfromFunc(drift_func=target_drift, diffusion_func=target_diffusion)


def load_data(config: DictConfig) -> Dataset:
    """Load the dataset for the n-scale potential

    Args:
        config (DictConfig): configuration object

    Returns:
        Dataset: the huggingface dataset
    """

    target_SDE = build_targets(config)
    integrator = SDEIntegrator(
        model=target_SDE,
        state_dim=2,
        bm_dim=1,
        method=config.data.method,
    )

    init_key, bm_key = jr.split(jr.PRNGKey(config.data.seed), 2)
    bm_keys = jr.split(bm_key, config.data.num_runs)
    init_conditions = config.data.init_scale * jr.normal(
        key=init_key, shape=(config.data.num_runs, 2)
    )
    sol = integrator.parallel_solve(
        initial_conditions=init_conditions,
        key=bm_keys,
        t0=config.data.t0,
        t1=config.data.t1,
        dt=config.dt,
        dt_rtol=config.data.dt_rtol,
        max_steps=config.data.max_steps,
        args=[config.temperature],
    )

    traj_length = sol.ts.shape[1]
    features = Features(
        {
            "t": Array2D(shape=(traj_length, 1), dtype="float64"),
            "x": Array2D(shape=(traj_length, 1), dtype="float64"),
            "args": Array2D(shape=(traj_length, 1), dtype="float64"),
        }
    )

    dataset = Dataset.from_dict(
        {
            "t": sol.ts[:, :, None],
            "x": sol.ys[:, :, :1],
            "args": config.temperature * jnp.ones_like(sol.ts[:, :, None]),
        },
        features=features,
    )
    return dataset.with_format("jax")


def build_model(config: DictConfig) -> OnsagerNetFD:
    """Build the OnsagerNetFD model for the n-scale potential

    Args:
        config (DictConfig): configuration object

    Returns:
        OnsagerNetFD: the OnsagerNetFD model
    """

    init_keys = jax.random.PRNGKey(config.model.seed)
    v_key, m_key, w_key = jax.random.split(init_keys, 3)

    potential = PotentialMLP(
        key=v_key,
        dim=config.dim,
        units=config.model.potential.units,
        activation=config.model.potential.activation,
        alpha=config.model.potential.alpha,
    )
    dissipation = DissipationMatrixMLP(
        key=m_key,
        dim=config.dim,
        units=config.model.dissipation.units,
        activation=config.model.potential.activation,
        alpha=config.model.dissipation.alpha,
        is_bounded=config.model.dissipation.is_bounded,
    )
    conservation = ConservationMatrixMLP(
        key=w_key,
        dim=config.dim,
        units=config.model.conservation.units,
        activation=config.model.potential.activation,
        is_bounded=config.model.dissipation.is_bounded,
    )

    return OnsagerNetFD(potential, dissipation, conservation)


@hydra.main(config_path="./config", config_name="n_scale_potential", version_base=None)
def train_model(config: DictConfig) -> None:
    """Train the OnsagerNetFD model for the n-scale potential

    Args:
        config (DictConfig): configuration object
    """

    runtime_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    logger = logging.getLogger(__name__)

    logger.info(f"Loading dataset...")
    dataset = load_data(config)
    train_traj_len = config.train.get("train_traj_len", None)
    if train_traj_len is not None:
        dataset = shrink_trajectory_len(
            dataset, train_traj_len
        )  # change the trajectory length to improve GPU usage
    logger.info(f"Building model...")
    model = build_model(config)
    trainer = MLETrainer(opt_options=config.train.opt, rop_options=config.train.rop)

    logger.info(f"Training OnsagerNetFD for {config.train.num_epochs} epochs...")
    trained_model, _, _ = trainer.train(
        model=model,
        dataset=dataset,
        num_epochs=config.train.num_epochs,
        batch_size=config.train.batch_size,
        logger=logger,
        checkpoint_dir=runtime_dir,
        checkpoint_every=config.train.checkpoint_every,
    )

    logger.info(f"Saving output to {runtime_dir}")
    eqx.tree_serialise_leaves(os.path.join(runtime_dir, "model.eqx"), trained_model)


if __name__ == "__main__":
    train_model()
