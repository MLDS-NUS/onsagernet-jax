"""
# Custom equinox modules for OnsagerNet components

This module contains custom equinox modules for the components of the OnsagerNet model,
which are used in various examples provided in the repository.

For new applications, it is suggested to try the simple models here first
and then build upon them, adapting if necessary to the specific problem at hand.





"""

import jax
import jax.numpy as jnp
import equinox as eqx

from ._activations import get_activation
from ._layers import ConstantLayer

# ------------------------- Typing imports ------------------------- #

from jax import Array
from jax.typing import ArrayLike
from typing import Callable
from jax.random import PRNGKey


# ------------------------------------------------------------------ #
#                           Template models                          #
# ------------------------------------------------------------------ #


class MLP(eqx.Module):
    """Multi-layer perceptron."""

    layers: list[eqx.nn.Linear]
    activation: Callable[[ArrayLike], Array]

    def __init__(
        self, key: PRNGKey, dim: int, units: list[int], activation: str
    ) -> None:
        r"""Multi-layer perceptron.

        Example:
        `mlp = MLP(key=jax.random.PRNGKey(0), dim=2, units=[32, 32, 1], activation='tanh')`
        gives a 2-hidden-layer MLP

        $$2 \to 32 \to 32 \to 1$$

        with tanh activation.

        Args:
            key (PRNGKey): random key
            dim (int): dimension of the input
            units (list[int]): layer sizes
            activation (str): activation function (can be any in `jax.nn` or custom ones defined in `onsagernet._activations`)
        """
        num_layers = len(units)
        units = [dim] + units
        keys = jax.random.split(key, num_layers)
        self.layers = [
            eqx.nn.Linear(units[i], units[i + 1], key=key) for i, key in enumerate(keys)
        ]
        self.activation = get_activation(activation)

    def __call__(self, x: ArrayLike) -> Array:
        h = x
        for layer in self.layers[:-1]:
            h = self.activation(layer(h))
        output = self.layers[-1](h)
        return output


# ------------------------------------------------------------------ #
#                         Potential networks                         #
# ------------------------------------------------------------------ #


class PotentialMLP(MLP):
    """Potential network based on a multi-layer perceptron."""

    alpha: float
    dim: int
    param_dim: int

    def __init__(
        self,
        key: PRNGKey,
        dim: int,
        units: list[int],
        activation: str,
        alpha: float,
        param_dim: int = 0,
    ) -> None:
        r"""Potential network based on a multi-layer perceptron.

        This implements the potential function
        $$
            V(x, args) = \alpha \|(x, args)\|^2 + \text{MLP}(x, args)
        $$
        where $x$ is the input and $u$ are additional parameters.
        The constant $\alpha \geq 0$ is a regularisation term,
        which gives a quadratic growth to ensure that the potential is integrable.
        We are tacitly assuming that MLP is of sub-quadratic growth,
        so only choose activation functions that have this property
        (most activation functions are either bounded or of linear growth).

        Args:
            key (PRNGKey): random key
            dim (int): dimension of the input
            units (list[int]): layer sizes
            activation (str): activation function (can be any in `jax.nn` or custom ones defined in `onsagernet._activations`)
            alpha (float): regulariser
            param_dim (int, optional): dimensions of the parameters. Defaults to 0.
        """
        self.dim = dim
        units = units + [1]
        self.param_dim = param_dim
        super().__init__(key, dim + param_dim, units, activation)
        self.alpha = alpha

    def __call__(self, x: ArrayLike, args: ArrayLike) -> Array:
        if self.param_dim > 0:
            x = jnp.concatenate([x, args[1:]], axis=0)
        output = super().__call__(x) + self.alpha * (x @ x)
        return jnp.squeeze(output)


class PotentialResMLP(MLP):
    r"""Potential network with a residual connection."""

    alpha: float
    gamma_layer: eqx.nn.Linear
    dim: int
    param_dim: int

    def __init__(
        self,
        key: PRNGKey,
        dim: int,
        units: list[int],
        activation: str,
        n_pot: int,
        alpha: float,
        param_dim: int = 0,
    ) -> None:
        r"""Potential network with a residual connection.

        This implements the modified potential function
        $$
            V(x, args) = \alpha \|(x, args)\|^2
            + \frac{1}{2}
            \| \text{MLP}(x, args)+ \Gamma (x, args) \|^2
        $$
        where

        - $\phi$ is a MLP of dim + param_dim -> n_pot
        - $\Gamma$ ia matrix of size [n_pot, dim + para_dim]
        - $\alpha > 0$ is a scalar regulariser

        Args:
            key (PRNGKey): random key
            dim (int): dimension of the input
            units (list[int]): layer sizes
            activation (str): activation function (can be any in `jax.nn` or custom ones defined in `onsagernet._activations`)
            n_pot (int): size of the MLP part of the potential
            alpha (float): regulariser
            param_dim (int, optional): dimension of the parameters. Defaults to 0.
        """
        self.dim = dim
        self.param_dim = param_dim
        units = units + [n_pot]
        mlp_key, gamma_key = jax.random.split(key)
        super().__init__(mlp_key, dim + param_dim, units, activation)
        self.alpha = alpha
        self.gamma_layer = eqx.nn.Linear(
            dim + param_dim, n_pot, key=gamma_key, use_bias=False
        )

    def __call__(self, x: ArrayLike, args: ArrayLike) -> Array:
        if self.param_dim > 0:
            x = jnp.concatenate([x, args[1:]], axis=0)
        output_phi = super().__call__(x)
        output_gamma = self.gamma_layer(x)
        output_combined = (output_phi + output_gamma) @ (output_phi + output_gamma)
        regularisation = self.alpha * (x @ x)
        return 0.5 * output_combined + regularisation


# ------------------------------------------------------------------ #
#                        Dissipation networks                        #
# ------------------------------------------------------------------ #


class DissipationMatrixMLP(MLP):
    """Dissipation matrix network based on a multi-layer perceptron."""

    alpha: float
    is_bounded: bool
    dim: int

    def __init__(
        self,
        key: PRNGKey,
        dim: int,
        units: list[int],
        activation: str,
        alpha: float,
        is_bounded: bool = True,
    ) -> None:
        r"""Dissipation matrix network based on a multi-layer perceptron.

        The MLP maps $x$ of dimension `dim` to a matrix $L(x)$ of size `dim` x `dim`,
        and then reshapes it to a `dim` x `dim` matrix.
        Then, the output matrix is given by
        $$
            M(x) = \alpha I + L(x) L(x)^\top.
        $$
        If `is_bounded` is set to `True`, then the output is element-wise bounded
        by applying a `jax.nn.tanh` activation to the output matrix $L$.

        Args:
            key (PRNGKey): random key
            dim (int): dimension of the input
            units (list[int]): layer sizes
            activation (str): activation function (can be any in `jax.nn` or custom ones defined in `onsagernet._activations`)
            alpha (float): regulariser
            is_bounded (bool, optional): whether to give a element-wise bounded output. Defaults to True.
        """
        self.dim = dim
        units = units + [dim * dim]
        super().__init__(key, dim, units, activation)
        self.alpha = alpha
        self.is_bounded = is_bounded

    def __call__(self, x: ArrayLike) -> Array:
        L = super().__call__(x).reshape(self.dim, self.dim)
        if self.is_bounded:
            L = jax.nn.tanh(L)
        return self.alpha * jnp.eye(self.dim) + L @ L.T


# ------------------------------------------------------------------ #
#                        Conservation networks                       #
# ------------------------------------------------------------------ #


class ConservationMatrixMLP(MLP):
    """Conservation matrix network based on a multi-layer perceptron."""

    is_bounded: bool
    dim: int

    def __init__(
        self,
        key: PRNGKey,
        dim: int,
        activation: str,
        units: list[int],
        is_bounded: bool = True,
    ) -> None:
        r"""Conservation matrix network based on a multi-layer perceptron.

        The MLP maps $x$ of dimension `dim` to a matrix $L(x)$ of size `dim` x `dim`,
        and then reshapes it to a `dim` x `dim` matrix.
        Then, the output matrix is given by
        $$
            W(x) = L(x) - L(x)^\top.
        $$
        If `is_bounded` is set to `True`, then the output is element-wise bounded
        by applying a `jax.nn.tanh` activation to the output matrix $L$.

        Args:
            key (PRNGKey): random key
            dim (int): dimension of the input
            activation (str): activation function (can be any in `jax.nn` or custom ones defined in `onsagernet._activations`)
            units (list[int]): layer sizes
            is_bounded (bool, optional): whether to give a element-wise bounded output. Defaults to True.
        """
        self.dim = dim
        units = units + [dim * dim]
        super().__init__(key, dim, units, activation)
        self.is_bounded = is_bounded

    def __call__(self, x: ArrayLike) -> Array:
        L = super().__call__(x).reshape(self.dim, self.dim)
        if self.is_bounded:
            L = jax.nn.tanh(L)
        return L - L.T


# ------------------------------------------------------------------ #
#                         Diffusion networks                         #
# ------------------------------------------------------------------ #


class DiffusionMLP(MLP):
    """Diffusion matrix network based on a multi-layer perceptron."""

    alpha: float
    dim: int
    param_dim: int

    def __init__(
        self,
        key: PRNGKey,
        dim: int,
        units: list[int],
        activation: str,
        alpha: float,
        param_dim: int = 0,
    ) -> None:
        r"""Diffusion matrix network based on a multi-layer perceptron.

        This implements the diffusion matrix function
        $$
            \sigma(x, args) = \text{Chol}(\alpha I + \text{MLP}(x, args))
        $$
        where $\text{Chol}$ is the Cholesky decomposition.
        Here, MLP maps $(x, args)$ of dimension `dim` + `param_dim` to a matrix of size `dim` x `dim`,

        Args:
            key (PRNGKey): random key
            dim (int): dimension of the input
            units (list[int]): layer sizes
            activation (str): activation function (can be any in `jax.nn` or custom ones defined in `onsagernet._activations`)
            alpha (float): regulariser
            param_dim (int, optional): dimension of the parameters. Defaults to 0.
        """
        self.dim = dim
        self.param_dim = param_dim
        units = units + [dim * dim]
        super().__init__(key, dim + param_dim, units, activation)
        self.alpha = alpha

    def __call__(self, x: ArrayLike, args: ArrayLike) -> Array:
        if self.param_dim > 0:
            x = jnp.concatenate([x, args[1:]], axis=0)
        sigma = super().__call__(x).reshape(self.dim, self.dim)
        sigma_squared_regularised = self.alpha * jnp.eye(self.dim) + sigma @ sigma.T
        return jnp.linalg.cholesky(sigma_squared_regularised)


class DiffusionDiagonalMLP(MLP):
    """Diagonal diffusion matrix network based on a multi-layer perceptron."""

    alpha: float
    dim: int
    param_dim: int

    def __init__(
        self,
        key: PRNGKey,
        dim: int,
        units: list[int],
        activation: str,
        alpha: float,
        param_dim: int = 0,
    ) -> None:
        r"""Diagonal diffusion matrix network based on a multi-layer perceptron.

        This implements the diffusion matrix function
        $$
            \sigma(x, args) = \text{diag}(\alpha + \text{MLP}(x, args)^2)^{\frac{1}{2}}.
        $$
        Here, MLP maps $(x, args)$ of dimension `dim` + `param_dim` to a vector of size `dim`.

        Args:
            key (PRNGKey): random key
            dim (int): dimension of the input
            units (list[int]): layer sizes
            activation (str): activation function (can be any in `jax.nn` or custom ones defined in `onsagernet._activations`)
            alpha (float): regulariser
            param_dim (int, optional): dimension of the parameters. Defaults to 0.
        """
        self.dim = dim
        self.param_dim = param_dim
        units = units + [dim]
        super().__init__(key, dim + param_dim, units, activation)
        self.alpha = alpha

    def __call__(self, x: ArrayLike, args: ArrayLike) -> Array:
        if self.param_dim > 0:
            x = jnp.concatenate([x, args[1:]], axis=0)
        sigma_diag = super().__call__(x)
        sigma_diag_regularised = jnp.sqrt(self.alpha + sigma_diag**2)
        return jnp.diag(sigma_diag_regularised)


class DiffusionDiagonalConstant(eqx.Module):
    """Diagonal diffusion matrix network based on a constant layer."""

    alpha: float
    constant_layer: ConstantLayer
    dim: int

    def __init__(self, key: PRNGKey, dim: int, alpha: float) -> None:
        r"""Diagonal diffusion matrix network based on a constant layer.

        This implements the diffusion matrix function that is constant
        $$
            \sigma(x, args) = \text{diag}(\alpha + \text{Constant}^2)^{\frac{1}{2}}.
        $$
        where $\text{Constant}$ is a vector of size `dim`.
        Note that by constant we mean that it does not depend on the input $x$ or the parameters $args$,
        but it can be trained.

        Args:
            key (PRNGKey): random key
            dim (int): dimension of the input
            alpha (float): regulariser
        """
        self.dim = dim
        self.alpha = alpha
        self.constant_layer = ConstantLayer(dim, key)

    def __call__(self, x: ArrayLike, args: ArrayLike) -> Array:
        sigma_diag = self.constant_layer()
        sigma_squared_regularised = jnp.sqrt(self.alpha + sigma_diag**2)
        return jnp.diag(sigma_squared_regularised)


# ------------------------------------------------------------------ #
#                      Dimensionality transforms                     #
# ------------------------------------------------------------------ #


class PCATransform(eqx.Module):
    """PCA transform."""

    mean: ArrayLike
    components: ArrayLike
    scaling: ArrayLike
    centre: bool

    def __init__(
        self,
        mean: ArrayLike,
        components: ArrayLike,
        scaling: ArrayLike,
        centre: bool = False,
    ) -> None:
        r"""PCA transform.

        Transforms the input vector $x$ via
        $$
            x \mapsto \text{components} ( x ) / \sqrt{\text{scaling}}.
        $$
        If `centre` is set to `True`, then the input is first centered by subtracting the `mean`.

        Args:
            mean (ArrayLike): mean of the data used to fit the PCA
            components (ArrayLike): PCA components
            scaling (ArrayLike): scaling of the PCA transform (e.g. explained variance)
            centre (bool, optional): whether to center the data using `mean`. Defaults to False.
        """
        self.mean = jnp.array(mean)
        self.components = jnp.array(components)
        self.scaling = jnp.array(scaling)
        self.centre = centre

    def __call__(self, x: ArrayLike) -> Array:
        if self.centre:
            x = x - self.mean
        x_reduced = self.components @ x
        return x_reduced / jnp.sqrt(self.scaling)


class InversePCATransform(PCATransform):
    """Inverse PCA transform."""

    def __init__(
        self, mean: ArrayLike, components: ArrayLike, scaling: ArrayLike
    ) -> None:
        r"""Inverse PCA transform.

        Transforms the input vector $z$ via
        $$
            z \mapsto \text{components}^\top ( z \sqrt{\text{scaling}} )
        $$
        If `centre` is set to `True`, `mean` is added to the output.

        Args:
            mean (ArrayLike): mean of the data used to fit the PCA
            components (ArrayLike): PCA components
            scaling (ArrayLike): scaling of the PCA transform (e.g. explained variance)
        """
        super().__init__(mean, components, scaling)

    def __call__(self, z: ArrayLike) -> Array:
        x_reconstructed = self.components.T @ (jnp.sqrt(self.scaling) * z)
        if self.centre:
            x_reconstructed = x_reconstructed + self.mean
        return x_reconstructed


class PCAResNetTransform(PCATransform):
    mlp: MLP
    mlp_scale: float
    mlp_input_scale: float

    def __init__(
        self,
        mean: ArrayLike,
        components: ArrayLike,
        scaling: ArrayLike,
        key: PRNGKey,
        units: list[int],
        activation: str,
        mlp_scale: float,
        mlp_input_scale: float,
    ) -> None:
        r"""PCA-ResNet transform.

        This combines the PCA transform with an MLP to give a ResNet-like architecture.
        $$
            x \mapsto \text{PCA}(x) + \text{mlp\_scale} \times \text{MLP}(\text{mlp\_input\_scale} \times x).
        $$
        The input scale adjusts the input (often not order 1) so that the MLP can learn the correct scale.

        Args:
            mean (ArrayLike): mean of the data used to fit the PCA
            components (ArrayLike): PCA components
            scaling (ArrayLike): scaling of the PCA transform (e.g. explained variance)
            key (PRNGKey): random key
            units (list[int]): layer sizes of the MLP
            activation (str): activation function (can be any in `jax.nn` or custom ones defined in `onsagernet._activations`)
            mlp_scale (float): scale of the MLP output
            mlp_input_scale (float): scale of the input to the MLP
        """
        super().__init__(mean, components, scaling)
        units = units + [components.shape[0]]
        dim = components.shape[1]
        self.mlp = MLP(key, dim, units, activation)
        self.mlp_scale = mlp_scale
        self.mlp_input_scale = mlp_input_scale

    def __call__(self, x: ArrayLike) -> Array:
        pca_features = super().__call__(x)
        mlp_features = self.mlp(self.mlp_input_scale * x)
        return pca_features + self.mlp_scale * mlp_features

    def pca_transform(self, x: ArrayLike) -> Array:
        """Perform the PCA transform.

        Args:
            x (ArrayLike): state

        Returns:
            Array: pca features
        """
        return super().__call__(x)


class InversePCAResNetTransform(InversePCATransform):
    """Inverse PCA-ResNet transform."""

    mlp: MLP
    mlp_scale: float

    def __init__(
        self,
        mean: ArrayLike,
        components: ArrayLike,
        scaling: ArrayLike,
        key: PRNGKey,
        units: list[int],
        activation: str,
        mlp_scale: float,
    ) -> None:
        r"""Inverse PCA-ResNet transform.

        This combines the inverse PCA transform with an MLP to give a ResNet-like architecture.
        $$
            z \mapsto \text{PCA}^{-1}(z) + \text{mlp\_scale} \times \text{MLP}(z).
        $$

        Args:
            mean (ArrayLike): mean of the data used to fit the PCA
            components (ArrayLike): PCA components
            scaling (ArrayLike): scaling of the PCA transform (e.g. explained variance)
            key (PRNGKey): random key
            units (list[int]): layer sizes of the MLP
            activation (str): activation function (can be any in `jax.nn` or custom ones defined in `onsagernet._activations`)
            mlp_scale (float): scale of the MLP output
        """
        super().__init__(mean, components, scaling)
        units = units + [components.shape[1]]
        dim = components.shape[0]
        self.mlp = MLP(key, dim, units, activation)
        self.mlp_scale = mlp_scale

    def __call__(self, z: ArrayLike) -> Array:
        inverse_pca_recon = super().__call__(z)
        mlp_recon = self.mlp(z)
        return inverse_pca_recon + self.mlp_scale * mlp_recon

    def inverse_pca_transform(self, z: ArrayLike) -> Array:
        """Inverse PCA transform.

        Args:
            z (ArrayLike): reduced state

        Returns:
            Array: reconstructed state
        """
        return super().__call__(z)
