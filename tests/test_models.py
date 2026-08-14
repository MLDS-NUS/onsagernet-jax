import jax
import jax.numpy as jnp
import numpy as np
import pytest

from onsagernet.models import PotentialMLP
from onsagernet.models import PotentialMLP_scale
from onsagernet.models import PotentialResMLP
from onsagernet.models import PotentialResMLP_scale
from onsagernet.models import PotentialResMLP_scaleV2
from onsagernet.models import DiffusionConstant
from onsagernet.models import DiffusionCholeskyMLP

KEY = jax.random.PRNGKey(0)
DIM = 4
UNITS = [8, 8]
ACTIVATION = "tanh"
X = jax.random.normal(jax.random.PRNGKey(7), (DIM,))
ARGS = jnp.array([0.7, 1.3, -0.4])


# ------------------------------------------------------------------ #
#                         Potential networks                         #
# ------------------------------------------------------------------ #


def test_potential_mlp_scale_reduces_to_potential_mlp():
    """With a unit scale the model must agree with `PotentialMLP`."""
    common = dict(key=KEY, dim=DIM, units=UNITS, activation=ACTIVATION, alpha=0.1)
    reference = PotentialMLP(**common)
    scaled = PotentialMLP_scale(**common, scale=1.0)
    np.testing.assert_allclose(scaled(X, ARGS), reference(X, ARGS), rtol=1e-6)


def test_potential_mlp_scale_is_linear_in_the_scale():
    """Only the network branch is scaled, the quadratic regulariser is not."""
    common = dict(key=KEY, dim=DIM, units=UNITS, activation=ACTIVATION, alpha=0.1)
    quadratic = 0.1 * (X @ X)
    unit = PotentialMLP_scale(**common, scale=1.0)(X, ARGS)
    scaled = PotentialMLP_scale(**common, scale=7.0)(X, ARGS)
    np.testing.assert_allclose(scaled - quadratic, 7.0 * (unit - quadratic), rtol=1e-5)


def test_potential_res_mlp_scale_reduces_to_potential_res_mlp():
    """With a unit scale the model must agree with `PotentialResMLP`."""
    common = dict(
        key=KEY, dim=DIM, units=UNITS, activation=ACTIVATION, n_pot=5, alpha=0.1
    )
    reference = PotentialResMLP(**common)
    scaled = PotentialResMLP_scale(**common, scale=1.0)
    np.testing.assert_allclose(scaled(X, ARGS), reference(X, ARGS), rtol=1e-6)


def test_potential_res_mlp_scale_v2_recovers_the_quadratic_part():
    """As the scale grows the MLP branch vanishes and only the linear branch is left."""
    model = PotentialResMLP_scaleV2(
        key=KEY,
        dim=DIM,
        units=UNITS,
        activation=ACTIVATION,
        n_pot=5,
        alpha=0.1,
        scale=1e6,
    )
    gamma = model.gamma_layer(X)
    expected = 1e6 * (gamma @ gamma) + 0.1 * (X @ X)
    np.testing.assert_allclose(model(X, ARGS), expected, rtol=1e-3)


@pytest.mark.parametrize(
    "model_class, kwargs",
    [
        (PotentialMLP_scale, dict(alpha=0.1, scale=3.0)),
        (PotentialResMLP_scale, dict(n_pot=5, alpha=0.1, scale=3.0)),
        (PotentialResMLP_scaleV2, dict(n_pot=5, alpha=0.1, scale=3.0)),
    ],
)
def test_potential_networks_are_scalar_valued(model_class, kwargs):
    model = model_class(key=KEY, dim=DIM, units=UNITS, activation=ACTIVATION, **kwargs)
    assert model(X, ARGS).shape == ()

    # the parameters in args are appended to the state when param_dim > 0
    model = model_class(
        key=KEY, dim=DIM, units=UNITS, activation=ACTIVATION, param_dim=2, **kwargs
    )
    assert model(X, ARGS).shape == ()


# ------------------------------------------------------------------ #
#                         Diffusion networks                         #
# ------------------------------------------------------------------ #


def _assert_is_cholesky_factor(sigma):
    """A valid Cholesky factor is lower triangular with a positive diagonal."""
    sigma = np.asarray(sigma)
    np.testing.assert_array_equal(np.triu(sigma, k=1), np.zeros_like(sigma))
    assert np.all(np.diag(sigma) > 0)

    # hence the induced dissipation matrix is symmetric positive definite
    dissipation = sigma @ sigma.T
    np.testing.assert_allclose(dissipation, dissipation.T, rtol=1e-6)
    assert np.all(np.linalg.eigvalsh(dissipation) > 0)


def test_diffusion_constant():
    model = DiffusionConstant(key=KEY, dim=DIM, alpha=1.0)
    sigma = model()
    assert sigma.shape == (DIM, DIM)
    _assert_is_cholesky_factor(sigma)

    # the diffusion does not depend on the state or the parameters
    other = jax.random.normal(jax.random.PRNGKey(21), (DIM,))
    np.testing.assert_array_equal(model(X, ARGS), sigma)
    np.testing.assert_array_equal(model(other, ARGS), sigma)


def test_diffusion_cholesky_mlp():
    model = DiffusionCholeskyMLP(
        key=KEY, dim=DIM, units=UNITS, activation=ACTIVATION, alpha=0.01
    )
    sigma = model(X)
    assert sigma.shape == (DIM, DIM)
    _assert_is_cholesky_factor(sigma)

    # unlike `DiffusionConstant`, this one does depend on the state
    other = jax.random.normal(jax.random.PRNGKey(21), (DIM,))
    assert not np.array_equal(model(other), sigma)


def test_diffusion_cholesky_mlp_with_parameters():
    model = DiffusionCholeskyMLP(
        key=KEY,
        dim=DIM,
        units=UNITS,
        activation=ACTIVATION,
        alpha=0.01,
        param_dim=2,
    )
    sigma = model(X, ARGS)
    assert sigma.shape == (DIM, DIM)
    _assert_is_cholesky_factor(sigma)

    # the first element of args is the temperature and is not passed to the network
    other_args = ARGS.at[0].set(ARGS[0] + 1.0)
    np.testing.assert_array_equal(model(X, other_args), sigma)
