from onsagernet._losses import MLELoss, ReconLoss, CompareLoss
import jax.numpy as jnp
import numpy as np

from onsagernet.dynamics import SDEfromFunc, ReducedSDE
from onsagernet.transformations import EncoderfromFunc, DecoderfromFunc


def test_mleloss():
    loss_obj = MLELoss()

    sde = SDEfromFunc(
        drift_func=lambda t, x, args: -0.1 * x,
        diffusion_func=lambda t, x, args: args[0] * x[:, None],
    )

    t = np.array([[0.0]])
    t_plus = np.array([[0.1]])
    x = np.array([[0.4]])
    x_plus = np.array([[0.3]])
    args = np.array([[2.0]])

    loss_value = loss_obj.compute_sample_loss(sde, t, t_plus, x, x_plus, args)
    np.testing.assert_allclose(loss_value, 1.9190875)


def test_reconloss():
    loss_obj = ReconLoss()

    encoder = EncoderfromFunc(lambda x: x[:1])
    decoder = EncoderfromFunc(lambda z: jnp.repeat(z, 2, axis=-1))

    sde = SDEfromFunc(
        drift_func=lambda t, x, args: -0.1 * x,
        diffusion_func=lambda t, x, args: args[0] * x[:, None],
    )
    model = ReducedSDE(encoder, decoder, sde)

    x = np.array([1.0, 1.3])
    loss_value = loss_obj.compute_sample_loss(model, x)
    np.testing.assert_allclose(loss_value, 0.045)


# def test_compareloss():
#     compare_loss = CompareLoss()
