import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import pytest

from onsagernet.dynamics import UIDD1, UIDD2
from onsagernet.models import MLP
from onsagernet.models import PotentialMLP_scale
from onsagernet.models import DissipationMatrixMLP
from onsagernet.models import DiffusionCholeskyMLP

KEY = jax.random.PRNGKey(0)
DIM = 4
UNITS = [8, 8]
ACTIVATION = "tanh"
TEMPERATURE = 0.7
T = jnp.array(0.0)
X = 0.5 * jax.random.normal(jax.random.PRNGKey(7), (DIM,))
ARGS = jnp.array([TEMPERATURE])


# ------------------------------------------------------------------ #
#                              Fixtures                              #
# ------------------------------------------------------------------ #


def _potential_and_hamiltonian(key):
    v_key, w_key = jax.random.split(key)
    potential = PotentialMLP_scale(
        key=v_key, dim=DIM, units=UNITS, activation=ACTIVATION, alpha=0.5, scale=1.0
    )
    hamiltonian = MLP(w_key, DIM, UNITS + [DIM - 1], ACTIVATION)
    return potential, hamiltonian


def _make_uidd1(key=KEY):
    potential, hamiltonian = _potential_and_hamiltonian(key)
    dissipation = DissipationMatrixMLP(
        key=jax.random.split(key, 3)[2],
        dim=DIM,
        units=UNITS,
        activation=ACTIVATION,
        alpha=0.5,
        is_bounded=True,
    )
    return UIDD1(DIM, potential, dissipation, hamiltonian)


def _make_uidd2(key=KEY):
    potential, hamiltonian = _potential_and_hamiltonian(key)
    diffusion = DiffusionCholeskyMLP(
        key=jax.random.split(key, 3)[2],
        dim=DIM,
        units=UNITS,
        activation=ACTIVATION,
        alpha=0.5,
    )
    return UIDD2(DIM, potential, diffusion, hamiltonian)


def _matrix_div(matrix_func, x):
    return jnp.trace(jax.jacfwd(matrix_func)(x), axis1=1, axis2=2)


def _gamma(model, x):
    """Recovers the divergence-free part of the drift from the public API."""
    dissipation = model.dissipation
    dvdx = jax.grad(model.potential, argnums=0)(x, ARGS)
    reversible = -dissipation(x) @ dvdx + TEMPERATURE * _matrix_div(dissipation, x)
    return model.drift(T, x, ARGS) - reversible


# ------------------------------------------------------------------ #
#                        Rotation generators                         #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("make_model", [_make_uidd1, _make_uidd2])
def test_rotation_generators(make_model):
    J = np.asarray(make_model().J)
    assert J.shape == (DIM - 1, DIM, DIM)

    for k in range(DIM - 1):
        # anti-symmetric, with the only non-zero entries at (k, k + 1)
        np.testing.assert_array_equal(J[k], -J[k].T)
        assert J[k, k, k + 1] == 1.0
        assert J[k, k + 1, k] == -1.0
        assert np.count_nonzero(J[k]) == 2


# ------------------------------------------------------------------ #
#                  Fluctuation-dissipation relation                  #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("make_model", [_make_uidd1, _make_uidd2])
def test_fluctuation_dissipation_relation(make_model):
    r"""The diffusion must satisfy $\sigma \sigma^\top = 2 \epsilon M$."""
    model = make_model()
    sigma = model.diffusion(T, X, ARGS)
    assert sigma.shape == (DIM, DIM)

    dissipation = model.dissipation
    np.testing.assert_allclose(
        sigma @ sigma.T,
        2.0 * TEMPERATURE * dissipation(X),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("make_model", [_make_uidd1, _make_uidd2])
def test_dissipation_is_positive_definite(make_model):
    dissipation = make_model().dissipation
    M = np.asarray(dissipation(X))
    np.testing.assert_allclose(M, M.T, rtol=1e-5, atol=1e-6)
    assert np.all(np.linalg.eigvalsh(M) > 0)


# ------------------------------------------------------------------ #
#                          Identifiability                           #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("make_model", [_make_uidd1, _make_uidd2])
def test_gamma_is_divergence_free(make_model):
    r"""The time-irreversible part must satisfy $\nabla \cdot (\gamma e^{-V/\epsilon}) = 0$.

    This is what makes the decomposition of the drift into its time-reversible
    and time-irreversible parts identifiable, and it holds by construction
    because each elementary rotation contributes
    $\nabla \cdot (J_k \nabla f) = 0$ for a symmetric Hessian.
    """
    model = make_model()

    def gamma_times_density(x):
        potential = model.potential(x, ARGS)
        return _gamma(model, x) * jnp.exp(-potential / TEMPERATURE)

    divergence = jnp.trace(jax.jacfwd(gamma_times_density)(X))
    scale = jnp.linalg.norm(jax.jacfwd(gamma_times_density)(X))
    np.testing.assert_allclose(divergence / scale, 0.0, atol=1e-4)


@pytest.mark.parametrize("make_model", [_make_uidd1, _make_uidd2])
def test_gamma_vanishes_without_a_hamiltonian(make_model):
    """With $H = 0$ the dynamics reduces to the reversible part alone."""
    model = make_model()
    zeroed = eqx.tree_at(
        lambda m: m.Hamiltonian,
        model,
        replace_fn=lambda h: jax.tree_util.tree_map(
            lambda leaf: jnp.zeros_like(leaf) if eqx.is_inexact_array(leaf) else leaf, h
        ),
    )
    np.testing.assert_allclose(_gamma(zeroed, X), jnp.zeros(DIM), atol=1e-6)


# ------------------------------------------------------------------ #
#                    Equivalence of the two forms                    #
# ------------------------------------------------------------------ #


class _SquaredDiffusion(eqx.Module):
    r"""Wraps a diffusion module $\sigma$ into the dissipation $M = \sigma \sigma^\top$."""

    diffusion: eqx.Module

    def __call__(self, x):
        sigma = self.diffusion(x)
        return sigma @ sigma.T


def test_uidd1_and_uidd2_agree():
    r"""Both parameterisations describe the same dynamics when $M = \sigma \sigma^\top$."""
    potential, hamiltonian = _potential_and_hamiltonian(KEY)
    diffusion = DiffusionCholeskyMLP(
        key=jax.random.split(KEY, 3)[2],
        dim=DIM,
        units=UNITS,
        activation=ACTIVATION,
        alpha=0.5,
    )
    model2 = UIDD2(DIM, potential, diffusion, hamiltonian)
    model1 = UIDD1(DIM, potential, _SquaredDiffusion(diffusion), hamiltonian)

    np.testing.assert_allclose(
        model1.drift(T, X, ARGS), model2.drift(T, X, ARGS), rtol=1e-4, atol=1e-5
    )
    # the diffusion agrees up to the Cholesky factorisation, so compare the products
    sigma1 = model1.diffusion(T, X, ARGS)
    sigma2 = model2.diffusion(T, X, ARGS)
    np.testing.assert_allclose(
        sigma1 @ sigma1.T, sigma2 @ sigma2.T, rtol=1e-4, atol=1e-5
    )


# ------------------------------------------------------------------ #
#                          Training basics                           #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("make_model", [_make_uidd1, _make_uidd2])
def test_drift_and_diffusion_under_jit_and_vmap(make_model):
    model = make_model()
    xs = 0.5 * jax.random.normal(jax.random.PRNGKey(11), (5, DIM))

    drift = jax.jit(jax.vmap(lambda x: model.drift(T, x, ARGS)))(xs)
    diffusion = jax.jit(jax.vmap(lambda x: model.diffusion(T, x, ARGS)))(xs)
    assert drift.shape == (5, DIM)
    assert diffusion.shape == (5, DIM, DIM)
    assert np.all(np.isfinite(drift))
    assert np.all(np.isfinite(diffusion))


@pytest.mark.parametrize("make_model", [_make_uidd1, _make_uidd2])
def test_gradients_flow_to_the_network_parameters(make_model):
    model = make_model()
    grads = eqx.filter_grad(lambda m: jnp.sum(m.drift(T, X, ARGS)))(model)

    for name in ("potential", "Hamiltonian"):
        leaves = jax.tree_util.tree_leaves(
            eqx.filter(getattr(grads, name), eqx.is_inexact_array)
        )
        assert leaves, f"no differentiable leaves under {name}"
        assert any(np.any(np.asarray(leaf) != 0.0) for leaf in leaves)


@pytest.mark.parametrize("make_model", [_make_uidd1, _make_uidd2])
def test_rotation_generators_are_a_constant_buffer(make_model):
    """`J` is a constant, but it is still a differentiable leaf of the model.

    It is stored as an array attribute, so `equinox` treats it as trainable and
    it does receive a gradient. It must therefore be excluded from training with
    a `filter_spec`, otherwise the optimiser will move it away from the
    anti-symmetric structure the model relies on:

    ```python
    filter_spec = jax.tree_util.tree_map(lambda _: True, model)
    filter_spec = eqx.tree_at(lambda m: m.J, filter_spec, replace=False)
    ```
    """
    model = make_model()
    grads = eqx.filter_grad(lambda m: jnp.sum(m.drift(T, X, ARGS)))(model)
    assert eqx.is_inexact_array(grads.J)
    assert np.any(np.asarray(grads.J) != 0.0)

    filter_spec = jax.tree_util.tree_map(lambda _: True, model)
    filter_spec = eqx.tree_at(lambda m: m.J, filter_spec, replace=False)
    trainable, static = eqx.partition(model, filter_spec)
    assert trainable.J is None
    np.testing.assert_array_equal(static.J, model.J)
