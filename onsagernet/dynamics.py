r'''
# Dynamical models

This module defines the dynamical models used in the OnsagerNet and related architectures.
Note that the actual model components' architectures are not defined here, but rather it is the
assembly logic for the model components that is handled here.

The following is an example of how to use the `OnsagerNet` model
to create the stochastic OnsagerNet dynamics

$$
    dX(t) = -
    \left[
        M(X(t)) + W(x(t))
    \right] \nabla V(x(t), u(t)) dt
    + \sqrt{\epsilon} \sigma(x(t), u(t)) dW(t)
    \qquad
    X(t) \in \mathbb{R}^d, \quad u(t) \in \mathbb{R}^m.
$$

```python
import equinox as eqx
from onsagernet.dynamics import OnsagerNet

class MyPotential(eqx.Module):
    """Implement your V function here

    This should be a function (d + m) -> (1)
    """

class MyDissipation(eqx.Module):
    """Implement your M function here

    This should be a function (d) -> (d, d)
    """

class MyConservation(eqx.Module):
    """Implement your W function here

    This should be a function (d) -> (d, d)
    """

class MyDiffusion(eqx.Module):
    """Implement your sigma function here

    This should be a function (d + m) -> (d, d)
    """


potential = MyPotential()
dissipation = MyDissipation()
conservation = MyConservation()
diffusion = MyDiffusion()

sde = OnsagerNet(
    potential=potential,
    dissipation=dissipation,
    conservation=conservation,
    diffusion=diffusion,
)
```

The `sde` instance can then be used to simulate the dynamics
of the system or perform training.

- Some simple definition of the potential, dissipation, conservation, and diffusion functions are
  provided in `onsagernet.models`
- The `ReducedSDE` class includes both an `SDE` component and a dimensionality reduction component
  involving both an `onsagernet.transformations.Encoder` and a `onsagernet.transformations.Decoder`
- Standard training routines are provided in `onsagernet.trainers`
- The `UIDD1` and `UIDD2` classes implement the identifiable dissipative dynamics of Ref [4],
  in which the time-irreversible part of the drift comes from a Hamiltonian decomposition
  rather than from an anti-symmetric matrix. This makes the decomposition of the drift
  identifiable and gives access to the entropy production rate of the learned system.

## References

1. Chen, X. et al. Constructing custom thermodynamics using deep learning. Nature Computational Science 4, 66-85 (2024).
2. Novoselov, K. S. & Li, Q. Learning physical laws from observations of complex dynamics. Nature Computational Science 1-2 (2024).
3. Yu, H., Tian, X., E, W. & Li, Q. OnsagerNet: Learning stable and interpretable dynamics using a generalized Onsager principle. Phys. Rev. Fluids 6, 114402 (2021).
4. Zhu, A., Soh, B. W., Pavliotis, G. A. & Li, Q. Identifiable learning of dissipative dynamics.
'''

import jax
import jax.numpy as jnp
import equinox as eqx
from abc import abstractmethod

# ------------------------- Typing imports ------------------------- #

from typing import Callable
from jax.typing import ArrayLike
from jax import Array
from .transformations import Encoder, Decoder

DynamicCallable = Callable[[ArrayLike, ArrayLike, ArrayLike], Array]


# ------------------------------------------------------------------ #
#                               Helpers                              #
# ------------------------------------------------------------------ #


def _matrix_div(M: eqx.Module, x: ArrayLike) -> Array:
    r"""Computes the matrix divergence of a matrix function $M(x)$.

    This is defined in component form as
    $$
        [\nabla \cdot M(x)]_i = \sum_j \frac{\partial M_{ij}}{\partial x_j}.
    $$

    Args:
        M (eqx.Module): matrix function
        x (ArrayLike): state

    Returns:
        Array: \nabla \cdot M(x)
    """
    jac_M_x = jax.jacfwd(M)(x)
    return jnp.trace(jac_M_x, axis1=1, axis2=2)


def _rotation_generators(D: int) -> Array:
    r"""Builds the generators of the elementary two-dimensional rotations.

    Returns the stack $(J_1, \ldots, J_{d-1})$ where each $J_k$ is a
    $d \times d$ matrix whose only non-zero entries are a $1$ at position
    $(k, k+1)$ and a $-1$ at position $(k+1, k)$.
    These are used to build the divergence-free part of the drift in `UIDD1`
    and `UIDD2`.

    Args:
        D (int): dimension $d$ of the state

    Returns:
        Array: array of shape (d - 1, d, d)
    """
    J = jnp.zeros((D - 1, D, D))
    for d in range(D - 1):
        J = J.at[d, d, d + 1].set(1)
        J = J.at[d, d + 1, d].set(-1)
    return J


# ------------------------------------------------------------------ #
#                                 SDE                                #
# ------------------------------------------------------------------ #


class SDE(eqx.Module):
    """Base class for stochastic differential equations models."""

    @abstractmethod
    def drift(self, t: ArrayLike, x: ArrayLike, args: ArrayLike) -> Array:
        """Drift function

        Args:
            t (ArrayLike): time
            x (ArrayLike): state
            args (ArrayLike): additional arguments or parameters

        Returns:
            Array: drift vector field
        """
        pass

    @abstractmethod
    def diffusion(self, t: ArrayLike, x: ArrayLike, args: ArrayLike) -> Array:
        """Diffusion function

        Args:
            t (ArrayLike): time
            x (ArrayLike): state
            args (ArrayLike): additional arguments or parameters

        Returns:
            Array: diffusion matrix of size (state_dim, bm_dim)
        """
        pass


class SDEfromFunc(SDE):
    """SDE model defined by providing drift and diffusion functions."""

    _drift_func: DynamicCallable
    _diffusion_func: DynamicCallable

    def __init__(
        self, drift_func: DynamicCallable, diffusion_func: DynamicCallable
    ) -> None:
        r"""SDE model defined by providing drift and diffusion functions.

        This implements the dynamics
        $$
            dX(t) = f(t, X(t), u(t)) dt + g(t, X(t), u(t)) dW(t)
        $$
        where $f$ is the drift function and $g$ is the diffusion function.
        The `args` argument (which represents $u(t)$)
        is used to pass additional parameters to the drift and diffusion functions.

        Args:
            drift_func (Callable[[ArrayLike, ArrayLike, ArrayLike], Array]): provided drift function
            diffusion_func (Callable[[ArrayLike, ArrayLike, ArrayLike], Array]): provided diffusion function
        """

        self._drift_func = drift_func
        self._diffusion_func = diffusion_func

    def drift(self, t: ArrayLike, x: ArrayLike, args: ArrayLike) -> Array:
        """Drift function

        Args:
            t (ArrayLike): time
            x (ArrayLike): state
            args (ArrayLike): additional arguments or parameters

        Returns:
            Array: drift vector field
        """
        return self._drift_func(t, x, args)

    def diffusion(self, t: ArrayLike, x: ArrayLike, args: ArrayLike) -> Array:
        """Diffusion function

        Args:
            t (ArrayLike): time
            x (ArrayLike): state
            args (ArrayLike): addional arguments or parameters

        Returns:
            Array: diffusion matrix
        """
        return self._diffusion_func(t, x, args)


class ReducedSDE(eqx.Module):
    """SDE model with encoder and decoder with dimensionality reduction or closure modelling."""

    encoder: Encoder
    decoder: Decoder
    sde: SDE

    def __init__(self, encoder: Encoder, decoder: Decoder, sde: SDE) -> None:
        """SDE model with encoder and decoder with dimensionality reduction or closure modelling.

        The `sde` attribute can be any model of the [SDE](#SDE) class or its sub-classes.
        The `encoder` must be a (sub-)class of `onsagernet.transformations.Encoder` and
        the `decoder` must be a (sub-)class of `onsagernet.transformations.Decoder`.

        Args:
            encoder (Encoder): The encoder function mapping the microscopic state to the reduced state
            decoder (Decoder): The decoder function mapping the reduced state to the microscopic state
            sde (SDE): The stochastic dynamics for the reduced state
        """
        self.encoder = encoder
        self.decoder = decoder
        self.sde = sde


# ------------------------------------------------------------------ #
#                             OnsagerNet                             #
# ------------------------------------------------------------------ #


class OnsagerNet(SDE):
    potential: eqx.Module
    dissipation: eqx.Module
    conservation: eqx.Module
    diffusion_func: eqx.Module

    def __init__(
        self,
        potential: eqx.Module,
        dissipation: eqx.Module,
        conservation: eqx.Module,
        diffusion: eqx.Module,
    ) -> None:
        r"""Stochastic OnsagerNet model.

        Let $X(t) \in \mathbb{R}^d$. The Stochastic OnsagerNet model is defined by the SDE
        $$
            dX(t) = -
            \left[
                M(X(t)) + W(x(t))
            \right] \nabla V(x(t), u(t)) dt
            + \sqrt{\epsilon} \sigma(x(t), u(t)) dW(t)
        $$
        where

        - $M : \mathbb{R}^{d} \to \mathbb{R}^{d\times d}$ is the dissipation matrix, which is symmetric positive semi-definite for all $x$
        - $W : \mathbb{R}^{d} \to \mathbb{R}^{d\times d}$ is the conservation matrix, which is anti-symmetric for all $x$
        - $V : \mathbb{R}^{d} \to \mathbb{R}$ is the potential function
        - $\sigma: \mathbb{R}^{d} \to \mathbb{R}^{d\times d}$ is the (square) diffusion matrix
        - $u(t)$ are the additional parameters for the potential and diffusion functions, and note that **the first dimension of $u(t)$ is the temperature $\epsilon$**

        Args:
            potential (eqx.Module): potential function $V$
            dissipation (eqx.Module): dissipation matrix $M$
            conservation (eqx.Module): conservation matrix $W$
            diffusion (eqx.Module): diffusion matrix $\sigma$
        """
        self.potential = potential
        self.dissipation = dissipation
        self.conservation = conservation
        self.diffusion_func = diffusion

    def drift(self, t: ArrayLike, x: ArrayLike, args: ArrayLike) -> Array:
        """Drift function

        Args:
            t (ArrayLike): time
            x (ArrayLike): state
            args (ArrayLike): additional arguments or parameters, the first element is the temperature

        Returns:
            Array: drift vector field
        """
        dvdx = jax.grad(self.potential, argnums=0)(x, args)
        return -(self.dissipation(x) + self.conservation(x)) @ dvdx

    def diffusion(self, t: ArrayLike, x: ArrayLike, args: ArrayLike) -> Array:
        """Diffusion function

        Args:
            t (ArrayLike): time
            x (ArrayLike): state
            args (ArrayLike): additional arguments or parameters, the first element is the temperature

        Returns:
            Array: diffusion matrix
        """
        temperature = args[0]
        return jnp.sqrt(temperature) * self.diffusion_func(x, args)


# ------------------------------------------------------------------ #
#        OnsagerNet satifying fluctuation-dissipation relation       #
# ------------------------------------------------------------------ #


class OnsagerNetFD(OnsagerNet):
    shared: eqx.nn.Shared

    def __init__(
        self, potential: eqx.Module, dissipation: eqx.Module, conservation: eqx.Module
    ) -> None:
        r"""Stochastic OnsagerNet model satisfying a fluctuation-dissipation relation.

        This is a modified version of the Stochastic OnsagerNet model.
        Let $X(t) \in \mathbb{R}^d$. This model is defined by the SDE
        $$
            dX(t) = -
            \left[
                M(X(t)) + W(x(t))
            \right] \nabla V(x(t), u(t)) dt
            + \sqrt{2 \epsilon} [M(x(t)]^\frac{1}{2}dW(t)
        $$
        where

        - $M : \mathbb{R}^{d} \to \mathbb{R}^{d\times d}$ is the dissipation matrix,
          which is symmetric positive semi-definite for all $x$
        - $W : \mathbb{R}^{d} \to \mathbb{R}^{d\times d}$ is the conservation matrix,
          which is anti-symmetric for all $x$
        - $V : \mathbb{R}^{d} \to \mathbb{R}$ is the potential function
        - $u(t)$ are the additional parameters for the potential and diffusion functions,
          and note that **the first dimension of $u(t)$ is the temperature $\epsilon$**

        Notice that the main difference with `OnsagerNet` is that the
        diffusion matrix is now given by a (positive semi-definite) square root of the dissipation matrix.

        Args:
            potential (eqx.Module): potential function $V$
            dissipation (eqx.Module): dissipation matrix $M$
            conservation (eqx.Module): conservation matrix $W$
        """
        self.potential = potential
        self.conservation = conservation
        self.diffusion_func = None

        # Share the dissipation module
        dissipation_drift = dissipation
        dissipation_diffusion = dissipation
        where = lambda shared_layers: shared_layers[0]
        get = lambda shared_layers: shared_layers[1]
        self.shared = eqx.nn.Shared(
            (dissipation_drift, dissipation_diffusion), where, get
        )

    @property
    def dissipation(self) -> eqx.Module:
        """Dissipation matrix wrapper

        Returns:
            eqx.Module: dissipation matrix module
        """
        return self.shared()[0]

    def _matrix_div(self, M: eqx.Module, x: ArrayLike) -> Array:
        r"""Computes the matrix divergence of a matrix function $M(x)$.

        See `_matrix_div`.

        Args:
            M (eqx.Module): matrix function
            x (ArrayLike): state

        Returns:
            Array: \nabla \cdot M(x)
        """
        return _matrix_div(M, x)

    def drift(self, t: ArrayLike, x: ArrayLike, args: ArrayLike) -> Array:
        """Drift function

        Args:
            t (ArrayLike): time
            x (ArrayLike): state
            args (ArrayLike): additional arguments or parameters, the first element is the temperature

        Returns:
            Array: drift vector field
        """
        temperature = args[0]
        dissipation = self.shared()[0]
        f1 = super().drift(t, x, args)
        f2 = temperature * self._matrix_div(dissipation, x)
        return f1 + f2

    def diffusion(self, t: ArrayLike, x: ArrayLike, args: ArrayLike) -> Array:
        """Diffusion function

        Args:
            t (ArrayLike): time
            x (ArrayLike): state
            args (ArrayLike): additional arguments or parameters, the first element is the temperature

        Returns:
            Array: diffusion matrix
        """
        temperature = args[0]
        dissipation = self.shared()[1]
        M_x = dissipation(x)
        sqrt_M_x = jnp.linalg.cholesky(M_x)
        return jnp.sqrt(2.0 * temperature) * sqrt_M_x


# ------------------------------------------------------------------ #
#                                UIDD                                #
# ------------------------------------------------------------------ #


class UIDD1(SDE):
    r"""Identifiable dissipative dynamics, parameterised by the dissipation matrix.

    See `UIDD1.__init__` for the definition of the model, and `UIDD2` for the
    equivalent parameterisation in terms of the diffusion matrix.
    """

    potential: eqx.Module
    shared: eqx.nn.Shared
    Hamiltonian: eqx.Module
    J: Array

    def __init__(
        self,
        D: int,
        potential: eqx.Module,
        dissipation: eqx.Module,
        Hamiltonian: eqx.Module,
    ) -> None:
        r"""Identifiable dissipative dynamics, parameterised by the dissipation matrix.

        Let $Z(t) \in \mathbb{R}^d$. This model is defined by the SDE
        $$
            dZ(t) = -
            \left[
                M(Z(t)) \nabla V(Z(t))
                - \epsilon \nabla \cdot M(Z(t))
                - \gamma(Z(t))
            \right] dt
            + \sqrt{2 \epsilon} [M(Z(t))]^\frac{1}{2} dW(t)
        $$
        where

        - $M : \mathbb{R}^{d} \to \mathbb{R}^{d\times d}$ is the dissipation matrix,
          which is symmetric positive semi-definite for all $z$
        - $V : \mathbb{R}^{d} \to \mathbb{R}$ is the potential function
        - $\gamma : \mathbb{R}^{d} \to \mathbb{R}^{d}$ is the divergence-free
          (with respect to the invariant measure $e^{-V}$) part of the drift
        - $\epsilon$ is the temperature, taken as the first element of `args`

        Unlike `OnsagerNet`, where the time-irreversible part of the drift is
        modelled by an anti-symmetric matrix $W$, here it is obtained from a
        Hamiltonian decomposition. The term $\gamma e^{-V}$ is written as a sum
        of elementary two-dimensional rotations,
        $$
            \gamma(z) = \sum_{k=1}^{d-1}
            \left[
                \epsilon \, J_k \nabla H_k(z) - H_k(z) J_k \nabla V(z)
            \right],
        $$
        where $H : \mathbb{R}^{d} \to \mathbb{R}^{d-1}$ is the learned
        Hamiltonian and each $J_k$ is a $d \times d$ matrix whose only non-zero
        entries are a $1$ at position $(k, k+1)$ and a $-1$ at position
        $(k+1, k)$. This construction makes the decomposition of the drift into
        its time-reversible and time-irreversible parts identifiable, which in
        turn allows the entropy production rate of the learned system to be
        computed.

        Args:
            D (int): dimension $d$ of the state
            potential (eqx.Module): potential function $V$
            dissipation (eqx.Module): dissipation matrix $M$
            Hamiltonian (eqx.Module): Hamiltonian functions $H$
        """
        self.potential = potential
        self.Hamiltonian = Hamiltonian
        self.J = _rotation_generators(D)

        # Share the dissipation module
        dissipation_drift = dissipation
        dissipation_diffusion = dissipation
        where = lambda shared_layers: shared_layers[0]
        get = lambda shared_layers: shared_layers[1]
        self.shared = eqx.nn.Shared(
            (dissipation_drift, dissipation_diffusion), where, get
        )

    @property
    def dissipation(self) -> eqx.Module:
        """Dissipation matrix wrapper

        Returns:
            eqx.Module: dissipation matrix module
        """
        return self.shared()[0]

    def _matrix_div(self, M: eqx.Module, x: ArrayLike) -> Array:
        r"""Computes the matrix divergence of a matrix function $M(x)$.

        See `_matrix_div`.

        Args:
            M (eqx.Module): matrix function
            x (ArrayLike): state

        Returns:
            Array: \nabla \cdot M(x)
        """
        return _matrix_div(M, x)

    def _gamma(self, x: ArrayLike, dvdx: ArrayLike, temperature: ArrayLike) -> Array:
        r"""Computes the divergence-free part $\gamma$ of the drift.

        Args:
            x (ArrayLike): state
            dvdx (ArrayLike): gradient of the potential at `x`
            temperature (ArrayLike): temperature $\epsilon$

        Returns:
            Array: \gamma(x)
        """
        H = self.Hamiltonian(x)
        grad_H = jax.jacfwd(self.Hamiltonian, argnums=0)(x)
        return temperature * jnp.einsum("dab,db->a", self.J, grad_H) - jnp.einsum(
            "d,dab,b->a", H, self.J, dvdx
        )

    def drift(self, t: ArrayLike, x: ArrayLike, args: ArrayLike) -> Array:
        """Drift function

        Args:
            t (ArrayLike): time
            x (ArrayLike): state
            args (ArrayLike): additional arguments or parameters, the first element is the temperature

        Returns:
            Array: drift vector field
        """
        temperature = args[0]
        dissipation = self.shared()[0]
        dvdx = jax.grad(self.potential, argnums=0)(x, args)
        gamma = self._gamma(x, dvdx, temperature)
        return (
            -dissipation(x) @ dvdx
            + temperature * self._matrix_div(dissipation, x)
            + gamma
        )

    def diffusion(self, t: ArrayLike, x: ArrayLike, args: ArrayLike) -> Array:
        """Diffusion function

        Args:
            t (ArrayLike): time
            x (ArrayLike): state
            args (ArrayLike): additional arguments or parameters, the first element is the temperature

        Returns:
            Array: diffusion matrix
        """
        temperature = args[0]
        dissipation = self.shared()[1]
        M_x = dissipation(x)
        sqrt_M_x = jnp.linalg.cholesky(M_x)
        return jnp.sqrt(2.0 * temperature) * sqrt_M_x


class UIDD2(SDE):
    r"""Identifiable dissipative dynamics, parameterised by the diffusion matrix.

    See `UIDD2.__init__` for the definition of the model, and `UIDD1` for the
    equivalent parameterisation in terms of the dissipation matrix.
    """

    shared: eqx.nn.Shared
    potential: eqx.Module
    Hamiltonian: eqx.Module
    J: Array

    def __init__(
        self,
        D: int,
        potential: eqx.Module,
        diffusion: eqx.Module,
        Hamiltonian: eqx.Module,
    ) -> None:
        r"""Identifiable dissipative dynamics, parameterised by the diffusion matrix.

        This is the same model as `UIDD1`, but parameterised by the diffusion
        matrix $\sigma$ rather than by the dissipation matrix $M$. The two are
        related by $M = \sigma \sigma^\top$, so that
        $$
            dZ(t) = -
            \left[
                M(Z(t)) \nabla V(Z(t))
                - \epsilon \nabla \cdot M(Z(t))
                - \gamma(Z(t))
            \right] dt
            + \sqrt{2 \epsilon} \, \sigma(Z(t)) dW(t),
            \qquad
            M = \sigma \sigma^\top.
        $$
        See `UIDD1` for the meaning of $V$, $\gamma$ and $\epsilon$.

        Parameterising by $\sigma$ avoids the Cholesky factorisation that
        `UIDD1` performs at every evaluation of the diffusion, at the cost of
        an extra matrix product when the dissipation is needed. It is the
        recommended choice unless the dissipation matrix is the quantity that
        is directly available.

        Args:
            D (int): dimension $d$ of the state
            potential (eqx.Module): potential function $V$
            diffusion (eqx.Module): diffusion matrix $\sigma$
            Hamiltonian (eqx.Module): Hamiltonian functions $H$
        """
        self.potential = potential
        self.Hamiltonian = Hamiltonian
        self.J = _rotation_generators(D)

        # Share the diffusion module
        diffusion_drift = diffusion
        diffusion_diffusion = diffusion
        where = lambda shared_layers: shared_layers[0]
        get = lambda shared_layers: shared_layers[1]
        self.shared = eqx.nn.Shared((diffusion_drift, diffusion_diffusion), where, get)

    def dissipation(self, x: ArrayLike) -> Array:
        r"""Dissipation matrix $M = \sigma \sigma^\top$.

        Args:
            x (ArrayLike): state

        Returns:
            Array: dissipation matrix
        """
        sqrtdissipation = self.shared()[1]
        M_x = sqrtdissipation(x)
        return M_x @ M_x.T

    def _matrix_div(self, M: eqx.Module, x: ArrayLike) -> Array:
        r"""Computes the matrix divergence of a matrix function $M(x)$.

        See `_matrix_div`.

        Args:
            M (eqx.Module): matrix function
            x (ArrayLike): state

        Returns:
            Array: \nabla \cdot M(x)
        """
        return _matrix_div(M, x)

    def _gamma(self, x: ArrayLike, dvdx: ArrayLike, temperature: ArrayLike) -> Array:
        r"""Computes the divergence-free part $\gamma$ of the drift.

        Args:
            x (ArrayLike): state
            dvdx (ArrayLike): gradient of the potential at `x`
            temperature (ArrayLike): temperature $\epsilon$

        Returns:
            Array: \gamma(x)
        """
        H = self.Hamiltonian(x)
        grad_H = jax.jacfwd(self.Hamiltonian, argnums=0)(x)
        return temperature * jnp.einsum("dab,db->a", self.J, grad_H) - jnp.einsum(
            "d,dab,b->a", H, self.J, dvdx
        )

    def drift(self, t: ArrayLike, x: ArrayLike, args: ArrayLike) -> Array:
        """Drift function

        Args:
            t (ArrayLike): time
            x (ArrayLike): state
            args (ArrayLike): additional arguments or parameters, the first element is the temperature

        Returns:
            Array: drift vector field
        """
        temperature = args[0]
        dissipation = self.dissipation
        dvdx = jax.grad(self.potential, argnums=0)(x, args)
        gamma = self._gamma(x, dvdx, temperature)
        return (
            -dissipation(x) @ dvdx
            + temperature * self._matrix_div(dissipation, x)
            + gamma
        )

    def diffusion(self, t: ArrayLike, x: ArrayLike, args: ArrayLike) -> Array:
        """Diffusion function

        Args:
            t (ArrayLike): time
            x (ArrayLike): state
            args (ArrayLike): additional arguments or parameters, the first element is the temperature

        Returns:
            Array: diffusion matrix
        """
        temperature = args[0]
        diffusion = self.shared()[1]
        return jnp.sqrt(2.0 * temperature) * diffusion(x)
