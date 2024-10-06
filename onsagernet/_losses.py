"""
# Custom loss functions

This module contains custom loss functions for training the models.

The case class `Loss` is an abstract class that defines the interface for loss functions.
Sub-classes must implement the `Loss.compute_sample_loss` method.

# """

import jax
import equinox as eqx
import jax.numpy as jnp
from abc import abstractmethod
from jax.scipy.stats import multivariate_normal
from jax.typing import ArrayLike
from jax import Array
from .dynamics import SDE, ReducedSDE


class Loss(eqx.Module):
    """Base class for losses

    Computes the loss over a batch of samples.pdoc --math my_module

    Args:
        model (eqx.Module): the model to be trained
        data_arrs (tuple[ArrayLike, ...]): input data arrays of the format (t, x, args)

    Returns:
        float: loss value
    """

    @abstractmethod
    def compute_sample_loss(
        self, model: eqx.Module, *processed_data_arrs: tuple[ArrayLike, ...]
    ) -> Array:
        """Computes the loss for a single sample.

        Args:
            model (eqx.Module): the model to be trained
            processed_data_arrs (tuple[ArrayLike, ...]): processed data arrays

        Returns:
            Array: the loss value
        """
        pass

    def _process_data_arrs(
        self, *data_arrs: tuple[ArrayLike, ...]
    ) -> tuple[ArrayLike, ...]:
        """Process the data arrays to a format that can be used by the loss function.

        The input data arrays are in the standard format of (t, x, args).
        The format of the output arrays depend on the loss.

        Args:
            data_arrs (tuple[ArrayLike, ...]): input data arrays

        Returns:
            tuple[ArrayLike, ...]: processed data arrays
        """
        return data_arrs

    @eqx.filter_jit
    def __call__(self, model: eqx.Module, *data_arrs: tuple[ArrayLike, ...]) -> float:
        data_arrs_processed = self._process_data_arrs(*data_arrs)
        in_axes = (None,) + len(data_arrs_processed) * (0,)
        compute_loss_over_time = jax.vmap(self.compute_sample_loss, in_axes=in_axes)
        compute_loss = jax.vmap(compute_loss_over_time, in_axes=in_axes)
        sample_losses = compute_loss(model, *data_arrs_processed)

        return jnp.mean(sample_losses)


class MLELoss(Loss):
    r"""Computes the maximum likelihood estimation loss for the SDE

    The loss is the negative log-likelihood of the data given the model.
    By an Euler-Maruyama discretization of the SDE
    $$
        dX(t) = f(t, X(t), \theta) dt + g(t, X(t), \theta) dW(t),
    $$
    which gives
    $$
        X(t + \Delta t) = X(t) + f(t, X(t), \theta) \Delta t + g(t, X(t), \theta) \Delta W(t),
    $$
    where $\Delta W(t) \sim N(0, \Delta t)$.
    Thus, the negative log-likelihood $X(t+\Delta t)$ is given by
    $$
        -\log p(X(t + \Delta t) | X(t), \theta) = \frac{1}{2} \log(2\pi)
        + \frac{1}{2} \log(\text{det}(\Sigma))
        + \frac{1}{2} (X(t + \Delta t) - X(t))^T \Sigma^{-1} (X(t + \Delta t) - X(t)).
    $$
    We will use `scipy.stats.multivariate_normal` to compute the log-likelihood.
    """

    def _process_data_arrs(self, t: ArrayLike, x: ArrayLike, args: ArrayLike) -> Array:
        return t[:, :-1, :], t[:, 1:, :], x[:, :-1, :], x[:, 1:, :], args[:, :-1, :]

    def compute_sample_loss(
        self,
        model: SDE,
        t: ArrayLike,
        t_plus: ArrayLike,
        x: ArrayLike,
        x_plus: ArrayLike,
        args: ArrayLike,
    ) -> float:
        """Computes the loss for a single sample.

        Args:
            model (SDE): the model to be trained
            t (ArrayLike): time
            t_plus (ArrayLike): time shifted by one step
            x (ArrayLike): state
            x_plus (ArrayLike): state shifted by one step
            args (ArrayLike): arguments for the current time step

        Returns:
            float: loss value
        """
        drift = model.drift(t, x, args)
        diffusion = model.diffusion(t, x, args)
        dt = t_plus - t
        data = (x_plus - x) / dt
        mean = drift
        cov = (1 / dt) * diffusion @ diffusion.T
        return -multivariate_normal.logpdf(data, mean, cov)


class ReconLoss(Loss):
    r"""Computes the reconstruction loss for the encoder and decoder components.

    The loss is defined by
    $$
        \| X - \text{decoder}(\text{encoder}(X)) \|^2.
    $$
    """

    def compute_sample_loss(self, model: ReducedSDE, x: ArrayLike) -> float:
        """Computes the loss for a single sample.

        Args:
            model (ReducedSDE): the model to be trained
            x (ArrayLike): state

        Returns:
            float: loss value
        """
        z = model.encoder(x)
        x_recon = model.decoder(z)
        return jnp.mean((x - x_recon) ** 2)


class CompareLoss(Loss):
    r"""Computes the comparison loss for the encoder and decoder components against PCA.

    This loss is defined by
    $$
        \max(0, \log(\text{recon_loss_model}) - \log(\text{recon_loss_pca})),
    $$
    where `recon_loss_model` is the reconstruction loss for the model
    and `recon_loss_pca` is the reconstruction loss for PCA.
    """

    def compute_sample_loss(self, model: ReducedSDE, x: ArrayLike) -> float:
        """Computes the loss for a single sample.

        Args:
            model (ReducedSDE): the model to be trained
            x (ArrayLike): state

        Returns:
            float: loss value
        """
        x_recon_model = model.decoder(model.encoder(x))
        x_recon_pca = model.decoder.inverse_closure_transform.inverse_pca_transform(
            model.encoder.closure_transform.pca_transform(x)
        )

        recon_loss_model = jnp.mean((x - x_recon_model) ** 2)
        recon_loss_pca = jnp.mean((x - x_recon_pca) ** 2)

        return jax.nn.relu(jnp.log(recon_loss_model) - jnp.log(recon_loss_pca))
