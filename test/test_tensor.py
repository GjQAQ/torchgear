import pytest
import torch

from torchgear.tensor import ShapeError, as1d, broadcastable


def test_shape_error_is_runtime_error():
    assert issubclass(ShapeError, RuntimeError)


def test_as1d_default():
    x = torch.arange(4)
    y = as1d(x)
    assert y.shape == (4,)
    torch.testing.assert_close(y, torch.arange(4))


def test_as1d_flattens_multidimensional_input():
    x = torch.arange(6).reshape(2, 3)
    y = as1d(x)
    assert y.shape == (6,)
    torch.testing.assert_close(y, torch.arange(6))


@pytest.mark.parametrize(
    ('ndim', 'dim', 'expected_shape'),
    [
        (2, -1, (1, 6)),
        (2, 0, (6, 1)),
        (3, 1, (1, 6, 1)),
    ],
)
def test_as1d_ndim_and_dim(ndim, dim, expected_shape):
    x = torch.arange(6).reshape(2, 3)
    y = as1d(x, ndim=ndim, dim=dim)
    assert y.shape == expected_shape
    torch.testing.assert_close(y.reshape(-1), torch.arange(6))


@pytest.mark.parametrize(
    ('shapes', 'expected'),
    [
        (((), ()), True),
        (((3, 1), (1, 4)), True),
        (((3, 4), (3, 1)), True),
        (((3, 4), (2, 4)), False),
        (((5,), (3,)), False),
    ],
)
def test_broadcastable_shapes(shapes, expected):
    assert broadcastable(*shapes) is expected


def test_broadcastable_tensors():
    a = torch.randn(3, 1)
    b = torch.randn(1, 4)
    assert broadcastable(a, b) is True
    assert broadcastable(a.shape, b.shape) is True


def test_broadcastable_incompatible_tensors():
    a = torch.randn(3, 4)
    b = torch.randn(2, 4)
    assert broadcastable(a, b) is False


def test_broadcastable_rejects_mixed_arguments():
    a = torch.randn(3, 1)
    with pytest.raises(TypeError, match='must be all tensors or all shapes'):
        broadcastable(a, (3, 1))
