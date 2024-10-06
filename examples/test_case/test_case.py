"""Test case."""

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

from onsagernet.dynamics import OnsagerNetFD
from onsagernet.models import PotentialMLP, DissipationMatrixMLP, ConservationMatrixMLP
from onsagernet.trainers import MLETrainer

# ------------------------- Typing imports ------------------------- #

from jax import Array
from jax.typing import ArrayLike


def build_targets(config: DictConfig) -> OnsagerNetFD:
    """Builds the target dynamics for the test case

    Args:
        config (DictConfig): configuration object

    Returns:
        OnsagerNetFD: target dynamics
    """

    def dissipation(x: ArrayLike) -> Array:
        def L(x):
            return jnp.array([[jnp.sin(x[0]), x[1] - x[0]], [3 * x[0], jnp.cos(x[1])]])

        L_x = L(x)
        return 0.2 * L_x @ L_x.T + 3.0 * jnp.eye(2)

    def conservation(x: ArrayLike) -> Array:
        return jnp.zeros((2, 2))

    def potential(x: ArrayLike, args: ArrayLike) -> Array:
        return 0.5 * x @ x + 0.3 * jnp.cos(x[0]) ** 2 * jnp.sin(x[0] - 2 * x[1]) ** 2

    return OnsagerNetFD(
        potential=potential, dissipation=dissipation, conservation=conservation
    )


def load_data(config: DictConfig) -> Dataset:
    """Load the dataset for the test case

    Args:
        config (DictConfig): configuration object

    Returns:
        Dataset: huggingface dataset
    """

    target_SDE = build_targets(config)
    integrator = SDEIntegrator(model=target_SDE, state_dim=2)

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
        args=[config.temperature],
    )

    traj_length = sol.ts.shape[1]
    features = Features(
        {
            "t": Array2D(shape=(traj_length, 1), dtype="float64"),
            "x": Array2D(shape=(traj_length, 2), dtype="float64"),
            "args": Array2D(shape=(traj_length, 1), dtype="float64"),
        }
    )

    dataset = Dataset.from_dict(
        {
            "t": sol.ts[:, :, None],
            "x": sol.ys,
            "args": config.temperature * jnp.ones_like(sol.ts[:, :, None]),
        },
        features=features,
    )
    return dataset.with_format("jax")


def build_model(config: DictConfig) -> OnsagerNetFD:
    """Build the OnsagerNetFD model to learn the target dynamics

    Args:
        config (DictConfig): configuration object

    Returns:
        OnsagerNetFD: OnsagerNetFD model
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
        activation=config.model.dissipation.activation,
        alpha=config.model.dissipation.alpha,
        is_bounded=config.model.dissipation.is_bounded,
    )
    conservation = ConservationMatrixMLP(
        key=w_key,
        dim=config.dim,
        units=config.model.conservation.units,
        activation=config.model.conservation.activation,
        is_bounded=config.model.dissipation.is_bounded,
    )
    return OnsagerNetFD(potential, dissipation, conservation)


@hydra.main(config_path="./config", config_name="test_case", version_base=None)
def train_model(config: DictConfig) -> None:
    """Main training routine

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
