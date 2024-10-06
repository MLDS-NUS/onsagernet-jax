"""Polymer dynamics with closure modelling."""

import os
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx

from sklearn.decomposition import PCA
from datasets import load_dataset

from examples.utils.data import shrink_trajectory_len

from onsagernet.dynamics import OnsagerNet, ReducedSDE
from onsagernet.transformations import ClosureEncoder, ClosureDecoder
from onsagernet.models import (
    PotentialResMLP,
    DissipationMatrixMLP,
    ConservationMatrixMLP,
    DiffusionDiagonalConstant,
    PCAResNetTransform,
    InversePCAResNetTransform,
)
from onsagernet.trainers import ClosureMLETrainer

import hydra
from omegaconf import DictConfig
import logging

# ------------------------- Typing imports ------------------------- #
from jax import Array
from jax.typing import ArrayLike
from datasets import Dataset
from typing import Any


class ChainExtension(eqx.Module):
    """Computes the normalised chain extension of a polymer."""

    mean: float
    std: float

    def __init__(self, x_normalisation: ArrayLike) -> None:
        """Computes the normalised chain extension of a polymer.

        Args:
            x_normalisation (ArrayLike): microscopic data used to compute normalisation
        """
        extensions = eqx.filter_jit(
            jax.vmap(self._compute_chain_extension),
            device=jax.devices("cpu")[0],  # For memory safety
        )(x_normalisation)
        self.mean, self.std = jnp.mean(extensions), jnp.std(extensions)

    def _compute_chain_extension(self, x: ArrayLike) -> Array:
        """Compute unnormalised chain extension.

        Args:
            x (ArrayLike): microscopic state

        Returns:
            Array: chain extension
        """
        return jnp.abs(jnp.max(x[::3]) - jnp.min(x[::3])).reshape(1)

    def __call__(self, x: ArrayLike) -> Array:
        extensions = self._compute_chain_extension(x)
        return (extensions - self.mean) / self.std


def build_model(config: DictConfig, dataset: Dataset) -> ReducedSDE:
    """Builds the model for polymer dynamics."""
    x_reshaped = dataset["x"].reshape(-1, config.microscopic_dim)
    pca = PCA(n_components=config.reduced_dim - config.macroscopic_dim)
    pca.fit(x_reshaped)

    init_keys = jax.random.PRNGKey(config.model.seed)
    enc_key, dec_key, v_key, m_key, w_key, d_key = jax.random.split(init_keys, 6)

    # Macorscopic coordinates of interest - chain extension
    chain_extension = ChainExtension(x_normalisation=x_reshaped)

    # Transform models
    closure_transform = PCAResNetTransform(
        mean=pca.mean_,
        components=pca.components_,
        scaling=pca.explained_variance_,
        key=enc_key,
        units=config.model.encoder.units,
        activation=config.model.encoder.activation,
        mlp_scale=config.model.encoder.mlp_scale,
        mlp_input_scale=config.model.encoder.mlp_input_scale,
    )
    inverse_closure_transform = InversePCAResNetTransform(
        mean=pca.mean_,
        components=pca.components_,
        scaling=pca.explained_variance_,
        key=dec_key,
        units=config.model.decoder.units,
        activation=config.model.decoder.activation,
        mlp_scale=config.model.decoder.mlp_scale,
    )
    encoder = ClosureEncoder(
        macroscopic_transform=chain_extension,
        closure_transform=closure_transform,
    )
    decoder = ClosureDecoder(
        inverse_closure_transform=inverse_closure_transform,
        macroscopic_dim=config.macroscopic_dim,
    )

    # SDE models
    potential = PotentialResMLP(
        key=v_key,
        dim=config.reduced_dim,
        units=config.model.potential.units,
        activation=config.model.potential.activation,
        n_pot=config.model.potential.n_pot,
        alpha=config.model.potential.alpha,
    )
    dissipation = DissipationMatrixMLP(
        key=m_key,
        dim=config.reduced_dim,
        units=config.model.dissipation.units,
        activation=config.model.potential.activation,
        alpha=config.model.dissipation.alpha,
    )
    conservation = ConservationMatrixMLP(
        key=w_key,
        dim=config.reduced_dim,
        activation=config.model.potential.activation,
        units=config.model.conservation.units,
    )
    diffusion = DiffusionDiagonalConstant(
        key=d_key,
        dim=config.reduced_dim,
        alpha=config.model.diffusion.alpha,
    )
    sde = OnsagerNet(
        potential=potential,
        dissipation=dissipation,
        conservation=conservation,
        diffusion=diffusion,
    )

    return ReducedSDE(encoder=encoder, decoder=decoder, sde=sde)


def get_filter_spec(model: ReducedSDE, filter_mode: str) -> Any:
    """Get filter logic for training.

    Two `filter_mode` options are available:

    - 'freeze': freeze the encoder and decoder transformations and only train the SDE
    - `free`: train all components of the model (except for PCA components and macrosopic transforms assumed to be fixed)

    Args:
        model (ReducedSDE): the model to be trained
        filter_mode (str): the filtering mode

    Raises:
        ValueError: if filter_mode is not 'freeze' or 'free'

    Returns:
        Any: filter_spec for training
    """

    filter_spec = jtu.tree_map(lambda _: True, model)
    if filter_mode == "freeze":
        filter_spec = eqx.tree_at(
            lambda tree: (tree.encoder, tree.decoder),
            filter_spec,
            replace=(False, False),
        )
    elif filter_mode == "free":
        filter_spec = eqx.tree_at(
            lambda tree: (
                tree.encoder.macroscopic_transform,
                tree.encoder.closure_transform.mean,
                tree.encoder.closure_transform.components,
                tree.encoder.closure_transform.scaling,
            ),
            filter_spec,
            replace=(False, False, False, False),
        )
        filter_spec = eqx.tree_at(
            lambda tree: (
                tree.decoder.inverse_closure_transform.mean,
                tree.decoder.inverse_closure_transform.components,
                tree.decoder.inverse_closure_transform.scaling,
            ),
            filter_spec,
            replace=(False, False, False),
        )
    else:
        raise ValueError(
            f"Filter mode {filter_mode} not recognised. Must be 'freeze' or 'free'."
        )
    return filter_spec


@hydra.main(config_path="./config", config_name="polymer_dynamics", version_base=None)
def train_model(config: DictConfig):
    """Training script for polymer dynamics with closure modelling."""

    logger = logging.getLogger(__name__)

    runtime_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Load data and build model
    logger.info(f"Loading data from {config.data.repo}...")
    dataset = load_dataset(config.data.repo, split="train")
    dataset = dataset.with_format("jax")
    train_traj_len = config.train.get("train_traj_len", None)
    if train_traj_len is not None:
        dataset = shrink_trajectory_len(
            dataset, train_traj_len
        )  # change the trajectory length to improve GPU usage

    logger.info("Building model...")
    model = build_model(config, dataset)

    # Train
    trainer = ClosureMLETrainer(
        opt_options=config.train.opt,
        rop_options=config.train.rop,
        loss_options=config.train.loss,
    )
    logger.info(
        f"Training transformations + OnsagerNet for {config.train.num_epochs_joint} epochs..."
    )
    model, _, opt_state = trainer.train(
        model=model,
        dataset=dataset,
        num_epochs=config.train.num_epochs_joint,
        batch_size=config.train.batch_size,
        logger=logger,
        filter_spec=get_filter_spec(model, "free"),
    )

    logger.info(f"Training OnsagerNet for {config.train.num_epochs} epochs...")
    trained_model, _, _ = trainer.train(
        model=model,
        dataset=dataset,
        num_epochs=config.train.num_epochs,
        batch_size=config.train.batch_size,
        logger=logger,
        opt_state=opt_state,
        filter_spec=get_filter_spec(model, "freeze"),
        checkpoint_dir=runtime_dir,
        checkpoint_every=config.train.checkpoint_every,
    )

    logger.info(f"Saving output to {runtime_dir}")
    eqx.tree_serialise_leaves(os.path.join(runtime_dir, "model.eqx"), trained_model)


if __name__ == "__main__":
    train_model()
