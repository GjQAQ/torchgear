import math

import pytest
import torch

from torchgear.math import abs2, expi, polynomial, ssqrt


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_abs2_real(dtype):
    x = torch.tensor([-2.0, 0.0, 3.0], dtype=dtype)
    torch.testing.assert_close(abs2(x), x.square())


@pytest.mark.parametrize('dtype', [torch.complex64, torch.complex128])
def test_abs2_complex(dtype):
    x = torch.tensor([1 + 2j, -3 + 4j, 0j], dtype=dtype)
    torch.testing.assert_close(abs2(x), x.abs().square())


def test_expi_matches_cos_sin():
    x = torch.tensor([0.0, math.pi / 2, math.pi])
    y = expi(x)
    expected = torch.complex(torch.cos(x), torch.sin(x))
    torch.testing.assert_close(y, expected)


def test_expi_rejects_complex_input():
    x = torch.tensor([1 + 1j])
    with pytest.raises(ValueError, match='Real-valued tensor expected'):
        expi(x)


@pytest.mark.parametrize(
    ('coefficients', 'x', 'expected'),
    [
        ([1.0], 0.0, 1.0),
        ([1.0, 2.0], 3.0, 7.0),
        ([1.0, 2.0, 3.0], 2.0, 17.0),
        ([1.0, None, 3.0], 2.0, 13.0),
    ],
)
def test_polynomial_scalars(coefficients, x, expected):
    assert polynomial(x, coefficients) == expected


def test_polynomial_with_tensor():
    x = torch.tensor([0.0, 1.0, 2.0])
    y = polynomial(x, [1.0, 2.0, 3.0])
    expected = 1.0 + 2.0 * x + 3.0 * x.square()
    torch.testing.assert_close(y, expected)


def test_polynomial_empty_coefficients():
    with pytest.raises(ValueError, match='must not be empty'):
        polynomial(1.0, [])


def test_polynomial_last_coefficient_none():
    with pytest.raises(ValueError, match='last coefficient cannot be None'):
        polynomial(1.0, [1.0, None])


def test_ssqrt_non_negative():
    x = torch.tensor([0.0, 1.0, 4.0, 9.0])
    y, mask = ssqrt(x)
    torch.testing.assert_close(y, torch.sqrt(x))
    torch.testing.assert_close(mask, torch.tensor([True, True, True, True]))


def test_ssqrt_negative_clamped():
    x = torch.tensor([-4.0, -1.0, 0.0, 4.0])
    y, mask = ssqrt(x)
    torch.testing.assert_close(y, torch.tensor([0.0, 0.0, 0.0, 2.0]))
    torch.testing.assert_close(mask, torch.tensor([False, False, True, True]))
