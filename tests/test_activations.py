import jax
import numpy as np

from onsagernet._activations import recu
from onsagernet._activations import srequ
from onsagernet._activations import get_activation


def test_recu():
    x = np.array([-1.5, 0.4, 1.3])
    expected_output = np.array([0.0, 0.4**3 / 3, 1.3 - 2 / 3])
    np.testing.assert_array_almost_equal(recu(x), expected_output)


def test_srequ():
    x = np.array([-1.5, 0.4, 1.3])
    expected_output = np.maximum(0, x) ** 2 - np.maximum(0, x - 0.5) ** 2
    np.testing.assert_array_almost_equal(srequ(x), expected_output)


def test_get_activation():
    recu_activation = get_activation("recu")
    assert recu_activation == recu

    srequ_activation = get_activation("srequ")
    assert srequ_activation == srequ

    tanh_activation = get_activation("tanh")
    assert tanh_activation == jax.nn.tanh
