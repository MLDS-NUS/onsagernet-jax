"""
# Dimensionality transformations

This module contains classes for dimensionality transformations, such as encoders and decoders.
These are to be used in the context where reduced dynamics is sought after,
but only microscopic data is available.

The main classes are `Encoder` and `Decoder`, which are abstract classes that should be
implemented by the user. The `ClosureEncoder` and `ClosureDecoder` classes are concrete
implementations of the `Encoder` and `Decoder` classes, respectively.
They are used to encode and decode the data into a reduced space,
after which the reduced dynamics can be trained.
In this case the reduced space consist of both known macroscopic coordinates
and learned closure coordinates.

The `Encoder` and `Decoder` objects are used in `onsagernet.dynamics.ReducedSDE`
to be used for closure modelling, for example.

```python
from onsagernet.transformations import ClosureEncoder, ClosureDecoder
from onsagernet.dynamics = ReducedSDE, OnsagerNet

encoder = ClosureEncoder(...)
decoder = ClosureDecoder(...)
sde = OnsagerNet(...)

reduced_sde = ReducedSDE(encoder, decoder, sde)  # can be used to train, predict, etc
```

As in `onsagernet.dynamics`, only the model assembly logic is provided here.
Some example implementations of the model architecture are provided
in the `onsagernet.models` module.

"""

import equinox as eqx
import jax.numpy as jnp
from abc import abstractmethod

# ------------------------- Typing imports ------------------------- #
from jax import Array
from jax.typing import ArrayLike

# ------------------------------------------------------------------ #
#                        Encoders and decoders                       #
# ------------------------------------------------------------------ #


class Encoder(eqx.Module):
    """The base class for encoders."""

    @abstractmethod
    def __call__(self, x: ArrayLike) -> Array:
        pass


class Decoder(eqx.Module):
    """The base class for decoders."""

    @abstractmethod
    def __call__(self, z: ArrayLike) -> Array:
        pass


class EncoderfromFunc(Encoder):
    """Encoder constructed from a given closure transform."""

    closure_transform: eqx.Module

    def __init__(self, closure_transform: eqx.Module) -> None:
        r"""Encoder constructed from a given closure transform.

        Takes a given closure transformation $x\mapsto z$
        to define the encoder.

        Args:
            closure_transform (eqx.Module): a given transformation $x\mapsto z$
        """
        self.closure_transform = closure_transform

    def __call__(self, x: ArrayLike) -> Array:
        return self.closure_transform(x)


class DecoderfromFunc(Decoder):
    """Decoder constructed from a given inverse closure transform."""

    inverse_closure_transform: eqx.Module

    def __init__(self, inverse_closure_transform: eqx.Module) -> None:
        r"""Decoder constructed from a given inverse closure transform.

        Takes a given inverse closure transformation $z\mapsto x$

        Args:
            inverse_closure_transform (eqx.Module): a given transformation $z\mapsto x$
        """
        self.inverse_closure_transform = inverse_closure_transform

    def __call__(self, z: ArrayLike) -> Array:
        return self.inverse_closure_transform(z)


class ClosureEncoder(EncoderfromFunc):
    """Closure encoder which combines known macroscopic coordinates
    with learned (or PCA) closure coordinates.
    """

    macroscopic_transform: eqx.Module

    def __init__(
        self, macroscopic_transform: eqx.Module, closure_transform: eqx.Module
    ) -> None:
        r"""Closure encoder which combines known macroscopic coordinates
        with learned (or PCA) closure coordinates.

        $$
            x \mapsto z = [\varphi^*(x), \hat\varphi(x)]
        $$

        where $\varphi^*$ is the known macroscopic transformation and
        $\hat\varphi$ is the learned closure transformation.

        Args:
            macroscopic_transform (eqx.Module): the known macroscopic transformation
            closure_transform (eqx.Module): the learned closure transformation
        """
        self.macroscopic_transform = macroscopic_transform
        self.closure_transform = closure_transform

    def __call__(self, x: ArrayLike) -> Array:
        """Combines the macroscopic and closure coordinates.

        Args:
            x (ArrayLike): miroscopic state

        Returns:
            Array: reduced state
        """
        macroscopic_coords = self.macroscopic_transform(x)
        closure_coords = self.closure_transform(x)
        reduced_coords = jnp.concatenate([macroscopic_coords, closure_coords])
        return reduced_coords


class ClosureDecoder(DecoderfromFunc):
    """Decodes from a closure encoder model output."""

    macroscopic_dim: int

    def __init__(
        self, inverse_closure_transform: eqx.Module, macroscopic_dim: int
    ) -> None:
        r"""Closure decoder which extracts the closure coordinates from the reduced state
        and then applies the inverse closure transformation to reconstruct the microscopic state.

        $$
            z[\text{macroscopic_dim}:] \mapsto x
        $$

        It is assuemd that the first `macroscopic_dim` coordinates are the known
        macroscopic coordinates and the rest are the learned closure coordinates.

        Args:
            inverse_closure_transform (eqx.Module): transformation from closure coordinates to microscopic state
            macroscopic_dim (int): the dimension of the known macroscopic state
        """
        self.inverse_closure_transform = inverse_closure_transform
        self.macroscopic_dim = macroscopic_dim

    def __call__(self, z: ArrayLike) -> Array:
        """Extracts the closure coordinates and applies the inverse closure transformation.

        Args:
            z (ArrayLike): reduced coordinates

        Returns:
            Array: reconstructed microscopic state
        """
        z_closure = z[self.macroscopic_dim :]
        return self.inverse_closure_transform(z_closure)
