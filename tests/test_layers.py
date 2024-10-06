from onsagernet._layers import ConstantLayer
import jax
import numpy as np


def test_constant_layer():
    key = jax.random.PRNGKey(0)
    layer = ConstantLayer(10, key)
    assert layer.dim == 10
    assert layer.weight.shape == (10,)
    output = layer()
    np.testing.assert_array_equal(output, layer.weight)
