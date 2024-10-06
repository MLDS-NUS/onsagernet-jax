"""
# Custom activations

This module contains custom activation functions.
"""

import jax
import jax.numpy as jnp

# ------------------------- Typing imports ------------------------- #

from jax import Array
from jax.typing import ArrayLike
from typing import Callable


# ------------------------------------------------------------------ #
#                     Custom activation functions                    #
# ------------------------------------------------------------------ #


@jax.jit
def recu(x: ArrayLike) -> Array:
    r"""Rectified cubic unit activation function.

    Implements the activation function
    $$
        x \mapsto
        \begin{cases}
            x^3 / 3 & \text{if } x \in [0, 1), \qquad \\
            x - 2/3 & \text{if } x \in [1, \infty).
        \end{cases}
    $$

    Args:
        x (ArrayLike): inputs

    Returns:
        Array: activated inputs
    """
    cubic_part = x**3 / 3
    linear_part = x - 2 / 3
    return jnp.where(x < 0, 0, jnp.where(x < 1, cubic_part, linear_part))


@jax.jit
def srequ(x: ArrayLike) -> Array:
    r"""Shifted rectified quadratic unit activation function.

    Implements the activation function
    $$
        x \mapsto
        \max(0, x)^2 - \max(0, x - 0.5)^2.
    $$

    Args:
        x (ArrayLike): inputs

    Returns:
        Array: activated inputs
    """
    return jnp.maximum(0, x) ** 2 - jnp.maximum(0, x - 0.5) ** 2


# ------------------------------------------------------------------ #
#                               Helpers                              #
# ------------------------------------------------------------------ #

CUSTOM_ACTIVATIONS = {
    "recu": recu,
    "srequ": srequ,
}


def get_activation(name: str) -> Callable[[ArrayLike], Array]:
    """Get the activation function by name.
    First checks if the activation function is a custom activation, then tries to get it from `jax.nn`.

    Args:
        name (str): name of the activation function

    Raises:
        ValueError: If the activation function is not found in custom activations or `jax.nn`
        TypeError: If the activation function is not callable

    Returns:
        _type_: activation function
    """

    # Check custom activations first
    if name in CUSTOM_ACTIVATIONS:
        activation_function = CUSTOM_ACTIVATIONS[name]
    else:
        # Try getting the activation function from jax.nn
        try:
            activation_function = getattr(jax.nn, name)
        except AttributeError:
            raise ValueError(
                f"Activation function '{name}' not found in custom activations or jax.nn"
            )

    # Check if the result is callable (i.e., a function)
    if not callable(activation_function):
        raise TypeError(f"The activation function '{name}' is not callable")

    return activation_function
