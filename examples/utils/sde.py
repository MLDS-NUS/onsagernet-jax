"""SDE utilities"""

import jax
import jax.numpy as jnp
import equinox as eqx

from diffrax import (
    diffeqsolve,
    ControlTerm,
    Euler,
    ItoMilstein,
    MultiTerm,
    ODETerm,
    SaveAt,
    VirtualBrownianTree,
)

from onsagernet._utils import default_floating_dtype

# ------------------------- Typing imports ------------------------- #

from onsagernet.dynamics import SDE
from typing import Optional, Callable
from jax.typing import ArrayLike, DTypeLike
from jax import Array
from jax.random import PRNGKey
from diffrax._solution import Solution

METHOD_ALIASES = {
    "euler": Euler,
    "milstein": ItoMilstein,
}


class SDEIntegrator:
    """SDE solver wrapper of diffrax."""

    def __init__(
        self,
        model: SDE,
        state_dim: int,
        bm_dim: Optional[int] = None,
        method: Optional[str] = "euler",
    ):
        """SDE solver wrapper of diffrax.

        Args:
            model (SDE): SDE to be solved
            state_dim (int): dimension of the state of the SDE
            bm_dim (Optional[int], optional): dimension of the Brownian motion. Defaults to None, meaning that the dimension is the same as the state dimension.
        """
        self.model = model
        self.state_dim = state_dim
        self.bm_dim = bm_dim or state_dim
        self.method = METHOD_ALIASES.get(method)

    def _build_paralle_solver(
        self,
        t0: float,
        t1: float,
        dt: float,
        args: ArrayLike,
        dt_rtol: float,
        max_steps: int,
        dtype: Optional[DTypeLike],
    ) -> Callable[[Array, Array], Array]:
        dtype = dtype or default_floating_dtype()
        bm_shape = jax.ShapeDtypeStruct(shape=(self.bm_dim,), dtype=dtype)
        saveat = SaveAt(ts=jnp.arange(t0, t1, dt))

        @eqx.filter_jit
        @jax.vmap
        def parallel_solve(init: Array, key: Array) -> Array:
            brownian_motion = VirtualBrownianTree(
                t0, t1, tol=dt_rtol * dt, shape=bm_shape, key=key
            )
            terms = MultiTerm(
                ODETerm(self.model.drift),
                ControlTerm(self.model.diffusion, brownian_motion),
            )
            solver = Euler()
            sol = diffeqsolve(
                terms,
                solver,
                t0,
                t1,
                dt0=dt,
                y0=init,
                saveat=saveat,
                max_steps=max_steps,
                args=args,
            )
            return sol

        return parallel_solve

    def parallel_solve(
        self,
        initial_conditions: ArrayLike,
        key: PRNGKey,
        t0: float,
        t1: float,
        dt: float,
        args: ArrayLike,
        dt_rtol: float = 0.1,
        max_steps: int = 10000,
        dtype: Optional[DTypeLike] = None,
    ) -> Solution:
        """Solve the SDE in parallel using `jax.vmap`.

        Args:
            initial_conditions (ArrayLike): initial conditions of size (num_runs, state_dim)
            key (PRNGKey): random key
            t0 (float): initial time
            t1 (float): final time
            dt (float): time step size
            args (ArrayLike): arguments to pass to drift and diffusion terms
            dt_rtol (float, optional): relative tolerance of Brownian motion. Defaults to 0.1.
            max_steps (int, optional): maximum number of solver steps. Defaults to 10000.
            dtype (Optional[DTypeLike], optional): data type. Defaults to None.

        Returns:
            Solution: the solution of the SDE
        """
        solver = self._build_paralle_solver(t0, t1, dt, args, dt_rtol, max_steps, dtype)
        return solver(initial_conditions, key)
