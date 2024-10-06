"""
# Utilities
"""

import jax
import jax.numpy as jnp

# ------------------------- Typing imports ------------------------- #

from jax.typing import DTypeLike


def default_floating_dtype() -> DTypeLike:
    """Returns the default dtype for floating point operations.

    Returns:
        DTypeLike: default data type
    """
    if jax.config.jax_enable_x64:
        return jnp.float64
    else:
        return jnp.float32
