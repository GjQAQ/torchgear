import pytest
import torch

from torchgear.interpolate import NdBSpline


def _poly(x):
    return x**3 + 2 * x**2 + x + 1


def _poly_deriv(x):
    return 3 * x**2 + 4 * x + 1


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_from_grid_1d_interpolates_nodes(dtype):
    x = torch.linspace(0, 3, 10, dtype=dtype)
    y = _poly(x)
    spline = NdBSpline.from_grid([x], y)
    torch.testing.assert_close(spline(x), y)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_from_grid_2d_interpolates_nodes(dtype):
    x = torch.linspace(0, 2, 6, dtype=dtype)
    y = torch.linspace(0, 3, 8, dtype=dtype)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    values = xx * yy + xx - yy
    spline = NdBSpline.from_grid([x, y], values)
    xi = torch.stack([xx, yy], dim=-1)
    torch.testing.assert_close(spline(xi), values)


def test_cubic_reproduces_cubic_polynomial():
    x = torch.linspace(0, 3, 10, dtype=torch.float64)
    y = _poly(x)
    spline = NdBSpline.from_grid([x], y, k=3)
    xq = torch.linspace(0, 3, 50, dtype=torch.float64)
    torch.testing.assert_close(spline(xq), _poly(xq), rtol=0, atol=1e-10)


def test_linear_spline_k1():
    x = torch.tensor([0.0, 1.0, 2.0, 3.0])
    y = torch.tensor([0.0, 2.0, 1.0, 3.0])
    spline = NdBSpline.from_grid([x], y, k=1)
    torch.testing.assert_close(spline(torch.tensor([0.5])), torch.tensor([1.0]))
    torch.testing.assert_close(spline(torch.tensor([1.5])), torch.tensor([1.5]))


def test_evaluate_grid_matches_call():
    x = torch.linspace(0, 1, 5)
    y = torch.linspace(0, 1, 7)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    values = torch.sin(xx) * torch.cos(yy)
    spline = NdBSpline.from_grid([x, y], values)
    xi = torch.stack([xx, yy], dim=-1)
    torch.testing.assert_close(spline.evaluate_grid([x, y]), spline(xi))


def test_trailing_dimensions():
    x = torch.linspace(0, 2, 6)
    y = torch.stack([x, x.square(), x.sin()], dim=-1)
    spline = NdBSpline.from_grid([x], y)
    torch.testing.assert_close(spline(x), y)


def test_first_derivative_1d():
    x = torch.linspace(0, 3, 10, dtype=torch.float64)
    y = _poly(x)
    spline = NdBSpline.from_grid([x], y, k=3)
    xq = torch.linspace(0.2, 2.8, 20, dtype=torch.float64)
    torch.testing.assert_close(spline(xq, nu=[1]), _poly_deriv(xq), rtol=0, atol=1e-9)


@pytest.mark.parametrize('bc_type', ['not-a-knot', 'clamped', 'natural'])
def test_boundary_conditions_interpolate_nodes(bc_type):
    x = torch.linspace(0, 2, 8)
    y = torch.exp(-x.square())
    spline = NdBSpline.from_grid([x], y, bc_type=bc_type)
    torch.testing.assert_close(spline(x), y)


def test_extrapolate_false_masks_outside_span():
    x = torch.linspace(0, 1, 5)
    y = x.square()
    spline = NdBSpline.from_grid([x], y, extrapolate=False)
    out = spline(torch.tensor([-0.1, 0.5, 1.1]))
    assert torch.isnan(out[0])
    torch.testing.assert_close(out[1], torch.tensor(0.25))
    assert torch.isnan(out[2])


def test_extrapolate_false_on_grid():
    x = torch.linspace(0, 1, 5)
    y = x.square()
    spline = NdBSpline.from_grid([x], y, extrapolate=False)
    xq = torch.tensor([-0.1, 0.0, 0.5, 1.0, 1.1])
    out = spline.evaluate_grid([xq])
    assert torch.isnan(out[0])
    torch.testing.assert_close(out[1:4], xq[1:4].square())
    assert torch.isnan(out[4])


def test_from_grid_rejects_too_few_points():
    x = torch.tensor([0.0, 1.0])
    y = torch.tensor([0.0, 1.0])
    with pytest.raises(ValueError, match='need at least'):
        NdBSpline.from_grid([x], y, k=3)


def test_from_grid_rejects_non_increasing_points():
    x = torch.tensor([0.0, 2.0, 1.0, 3.0])
    y = torch.tensor([0.0, 1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match='strictly increasing'):
        NdBSpline.from_grid([x], y)


def test_from_grid_rejects_shape_mismatch():
    x = torch.linspace(0, 1, 5)
    y = torch.linspace(0, 1, 4)
    with pytest.raises(ValueError, match='values has length'):
        NdBSpline.from_grid([x], y)


def test_call_rejects_wrong_query_dimension():
    x = torch.linspace(0, 1, 5)
    y = x.square()
    spline = NdBSpline.from_grid([x], y)
    with pytest.raises(ValueError, match='Expected last dim'):
        spline(torch.tensor([[0.0, 0.5, 1.0]]))


def test_evaluate_grid_rejects_wrong_axis_count():
    x = torch.linspace(0, 1, 5)
    y = x.square()
    spline = NdBSpline.from_grid([x], y)
    z = torch.linspace(0, 1, 3)
    with pytest.raises(ValueError, match='Expected 1 grid axes'):
        spline.evaluate_grid([x, z])
