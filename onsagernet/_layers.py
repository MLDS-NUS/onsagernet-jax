"""Custom layers as equinox modules"""

import jax
import equinox as eqx
from math import sqrt
from ._utils import default_floating_dtype
from jax import Array
from jax.random import PRNGKey
from typing import Optional
from jax.typing import DTypeLike


class ConstantLayer(eqx.Module):
    """Constant layer."""

    weight: Array
    dim: int

    def __init__(
        self, dim: int, key: PRNGKey, dtype: Optional[DTypeLike] = None
    ) -> None:
        """Constant layer.

        Returns a constant, trainable vector,
        similar to the bias term in a neural network,
        that is the same size as the input.

        Args:
            dim (int): dimension of the input space
            key (PRNGKey): random key
            dtype (Optional[DTypeLike], optional): data type. Defaults to None.
        """
        dtype = default_floating_dtype() if dtype is None else dtype
        lim = 1 / sqrt(dim)
        shape = (dim,)
        self.dim = dim
        self.weight = jax.random.uniform(
            key, shape, minval=-lim, maxval=lim, dtype=dtype
        )

    def __call__(self, *, key: Optional[PRNGKey] = None) -> Array:
        return self.weight
