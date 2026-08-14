from onsagernet._layers import ConstantLayer
from onsagernet._layers import ConstantTensorLayer
import jax
import numpy as np


def test_constant_layer():
    key = jax.random.PRNGKey(0)
    layer = ConstantLayer(10, key)
    assert layer.dim == 10
    assert layer.weight.shape == (10,)
    output = layer()
    np.testing.assert_array_equal(output, layer.weight)


def test_constant_tensor_layer():
    key = jax.random.PRNGKey(0)
    layer = ConstantTensorLayer((3, 5), key)
    assert layer.shape == (3, 5)
    assert layer.weight.shape == (3, 5)
    output = layer()
    np.testing.assert_array_equal(output, layer.weight)

    # the initialisation is bounded by 1 / sqrt(shape[0])
    assert np.all(np.abs(np.asarray(output)) <= 1 / np.sqrt(3))
