"""N-dimensional B-spline interpolation with PyTorch."""

from __future__ import annotations

from typing import Sequence, Union

import torch
from torch import Tensor

__all__ = [
    'NdBSpline',
]

BcType = Union[
    str,
    tuple[Union[str, Sequence[tuple[int, Tensor | float]]], ...],
    None,
]


# ---------------------------------------------------------------------------
# Knot vectors & boundary-condition parsing
# ---------------------------------------------------------------------------

def _not_a_knot(x: Tensor, k: int) -> Tensor:
    """Construct a knot vector with not-a-knot boundary conditions.

    :param Tensor x: Sample abscissae, shape ``(n,)``.
    :param int k: Spline degree.
    :return: Knot vector, shape ``(n + k + 1,)``.
    :rtype: Tensor
    """
    if k % 2 == 1:
        k2 = (k + 1) // 2
        interior = x
    else:
        k2 = k // 2
        interior = (x[1:] + x[:-1]) / 2
    interior = interior[k2:-k2] if k2 > 0 and interior.numel() > 2 * k2 else interior.new_empty(0)
    return torch.cat([
        x[0].expand(k + 1),
        interior,
        x[-1].expand(k + 1),
    ])


def _augknt(x: Tensor, k: int) -> Tensor:
    """Clamped-style open knot vector: ``k`` repeated boundary knots.

    :param Tensor x: Sample abscissae, shape ``(n,)``.
    :param int k: Spline degree.
    :return: Knot vector, shape ``(n + 2 * k,)``.
    :rtype: Tensor
    """
    return torch.cat([x[0].expand(k), x, x[-1].expand(k)])


def _normalize_bc(bc: BcType, ndim: int, trailing_shape: tuple[int, ...] = ()) -> list[tuple]:
    """Return a per-dimension ``(deriv_l, deriv_r)`` specification."""
    if bc is None or bc == 'not-a-knot':
        return [(None, None)] * ndim
    if isinstance(bc, str):
        side = _parse_side(bc, trailing_shape)
        return [(side, side)] * ndim
    if len(bc) == ndim and all(isinstance(b, (str, type(None))) or _is_deriv_spec(b) for b in bc):
        out = []
        for b in bc:
            if isinstance(b, str):
                side = _parse_side(b, trailing_shape)
                out.append((side, side))
            elif b is None:
                out.append((None, None))
            else:
                out.append((b, b))
        return out
    left, right = bc  # type: ignore[misc]
    return [(_parse_side(left, trailing_shape), _parse_side(right, trailing_shape))]


def _is_deriv_spec(spec) -> bool:
    if spec is None:
        return True
    try:
        for item in spec:
            if not (isinstance(item, (tuple, list)) and len(item) == 2):
                return False
        return True
    except TypeError:
        return False


def _parse_side(side: Union[str, None, Sequence], trailing_shape: tuple[int, ...]):
    if side is None:
        return None
    if isinstance(side, str):
        if side == 'clamped':
            return [(1, torch.zeros(trailing_shape))]
        if side == 'natural':
            return [(2, torch.zeros(trailing_shape))]
        raise ValueError(f'Unknown boundary condition alias: {side!r}')
    return list(side)


def _broadcast_bc_value(val, yv: Tensor) -> Tensor:
    v = torch.as_tensor(val, dtype=yv.dtype, device=yv.device)
    if v.shape == yv.shape:
        return v.reshape(-1)
    if v.numel() == 1:
        return v.expand_as(yv).reshape(-1)
    return v.reshape(-1)


def _normalize_k(k: int | Sequence[int], ndim: int) -> list[int]:
    if isinstance(k, int):
        return [k] * ndim
    k = [int(ki) for ki in k]
    if len(k) != ndim:
        raise ValueError(f'Expected {ndim} degrees, got {len(k)}.')
    return k


def _num_coeffs(t: Tensor, k: int) -> int:
    """Number of B-spline coefficients for knot vector ``t`` and degree ``k``."""
    return t.numel() - k - 1


def _validate_strictly_increasing(x: Tensor, name: str = 'x') -> None:
    if x.numel() > 1 and not torch.all(x[1:] > x[:-1]):
        raise ValueError(f'{name} must be strictly increasing.')


def _in_knot_span(x: Tensor, t: Tensor, k: int) -> Tensor:
    return (x >= t[k]) & (x <= t[-k - 1])


def _mask_outside_span(
    values: Tensor,
    coords: Sequence[Tensor],
    knots: Sequence[Tensor],
    degrees: Sequence[int],
) -> Tensor:
    """Replace grid values outside the knot span with NaN."""
    valid: Tensor | None = None
    for d, (coord, td, kd) in enumerate(zip(coords, knots, degrees)):
        in_span = _in_knot_span(coord.reshape(-1), td, kd)
        shape = [1] * values.ndim
        shape[d] = in_span.numel()
        in_span = in_span.reshape(shape)
        valid = in_span if valid is None else valid & in_span
    assert valid is not None
    return values.masked_fill(~valid, float('nan'))


def _mask_query_points(
    values: Tensor,
    pts: Tensor,
    knots: Sequence[Tensor],
    degrees: Sequence[int],
    batch_shape: torch.Size,
) -> Tensor:
    """Replace scattered query values outside the knot span with NaN."""
    valid = torch.ones(pts.shape[0], dtype=torch.bool, device=pts.device)
    for d, (td, kd) in enumerate(zip(knots, degrees)):
        valid &= _in_knot_span(pts[:, d], td, kd)
    mask = valid.reshape(batch_shape)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    return values.masked_fill(~mask, float('nan'))


# ---------------------------------------------------------------------------
# Core B-spline algorithms
# ---------------------------------------------------------------------------

def _find_span_batch(x: Tensor, t: Tensor, k: int) -> Tensor:
    """Span index for each query point.

    :return: Span indices with the same shape as ``x``.
    :rtype: Tensor
    """
    nt = _num_coeffs(t, k)
    x = x.to(dtype=t.dtype)
    span = torch.searchsorted(t, x.contiguous(), right=True) - 1
    return span.clamp(k, nt - 1).long()


def _bspline_derivative(t: Tensor, c: Tensor, k: int, nu: int = 1) -> tuple[Tensor, Tensor, int]:
    """Return ``(t, c, k)`` of the ``nu``-th derivative.

    :param Tensor t: Knot vector, shape ``(m,)``.
    :param Tensor c: B-spline coefficients, shape ``(nt,)`` or ``(nt, B)``
        where ``nt = m - k - 1``.
    :param int k: Spline degree.
    :param int nu: Derivative order. Default: ``1``.
    :return: Knot vector, coefficients, and degree of the derivative spline.
        ``c`` has shape ``(nt',)`` or ``(nt', B)`` with ``nt' = nt - nu``.
    :rtype: tuple[Tensor, Tensor, int]
    """
    for _ in range(nu):
        if k == 0:
            return t, torch.zeros_like(c), 0
        n = c.shape[0]
        denom = t[k + 1: k + n] - t[1: n]
        delta = c[1:] - c[:-1]
        scale = k * delta / denom.unsqueeze(-1) if delta.ndim > 1 else k * delta / denom
        c, t, k = scale, t[1:-1], k - 1
    return t, c, k


def _de_boor_batch(x: Tensor, t: Tensor, c: Tensor, k: int) -> Tensor:
    """Evaluate B-spline(s) at batch of points.

    :param Tensor x: Query points, shape ``(N,)``.
    :param Tensor t: Knot vector.
    :param Tensor c: Coefficients, shape ``(nt,)`` or ``(nt, B)``.
    :param int k: Spline degree.
    :return: Spline values, shape ``(N,)`` or ``(N, B)``.
    :rtype: Tensor
    """
    x = x.reshape(-1).to(dtype=t.dtype)
    c = c.to(dtype=t.dtype)
    spans = _find_span_batch(x, t, k)
    idx = (spans - k).unsqueeze(-1) + torch.arange(k + 1, device=t.device, dtype=spans.dtype)
    d = c[idx]
    if d.ndim == 2:
        d = d.unsqueeze(-1)
    for r in range(1, k + 1):
        for j in range(k, r - 1, -1):
            t_left = t[spans - k + j]
            t_right = t[spans + j - r + 1]
            denom = t_right - t_left
            alpha = torch.where(denom != 0, (x - t_left) / denom, torch.zeros_like(x))
            a = alpha.unsqueeze(-1)
            d[:, j, :] = (1.0 - a) * d[:, j - 1, :] + a * d[:, j, :]
    return d[:, k, :].squeeze(-1)

def _evaluate_all_bspl(
    t: Tensor, k: int, x: float, nu: int = 0, left: int | None = None,
) -> tuple[Tensor, slice]:
    """Non-zero basis values (or derivatives) at *x*."""
    span = left if left is not None else _find_span_batch(t.new_tensor(x), t, k).item()
    sl = slice(span - k, span + 1)
    xv = t.new_tensor([x])
    if nu == 0:
        return _design_matrix(xv, t, k, 0)[0, sl], sl

    nt = _num_coeffs(t, k)
    cols = torch.arange(span - k, span + 1, device=t.device)
    c = torch.zeros(nt, k + 1, dtype=t.dtype, device=t.device)
    j = torch.arange(k + 1, device=t.device)
    valid = (cols >= 0) & (cols < nt)
    c[cols[valid], j[valid]] = 1.0
    t_d, c_d, k_d = _bspline_derivative(t, c, k, nu)
    return _de_boor_batch(xv, t_d, c_d, k_d)[0], sl


def _design_matrix(x: Tensor, t: Tensor, k: int, nu: int = 0) -> Tensor:
    """Dense design matrix.

    :return: Matrix of shape ``(len(x), len(t) - k - 1)``.
    :rtype: Tensor
    """
    nt = _num_coeffs(t, k)
    x = x.reshape(-1).to(dtype=t.dtype, device=t.device)
    eye = torch.eye(nt, dtype=t.dtype, device=t.device)
    if nu == 0:
        return _de_boor_batch(x, t, eye, k)
    t_d, c_d, k_d = _bspline_derivative(t, eye, k, nu)
    return _de_boor_batch(x, t_d, c_d.T.contiguous(), k_d)


# ---------------------------------------------------------------------------
# 1D interpolation (internal)
# ---------------------------------------------------------------------------

def _make_knots_1d(x: Tensor, k: int, bc_left, bc_right, t: Tensor | None) -> Tensor:
    if t is not None:
        return torch.as_tensor(t, dtype=x.dtype, device=x.device)
    if bc_left is None and bc_right is None:
        return _not_a_knot(x, k)
    return _augknt(x, k)


def _bc_constraint_rows(
    t: Tensor,
    k: int,
    x_end: float,
    nt: int,
    derivs: list,
    y_ref: Tensor,
) -> tuple[list[Tensor], list[Tensor]]:
    rows, rhss = [], []
    for order, val in derivs:
        bb, sl = _evaluate_all_bspl(t, k, x_end, int(order))
        row = t.new_zeros(nt)
        row[sl] = bb
        rows.append(row.unsqueeze(0))
        rhss.append(_broadcast_bc_value(val, y_ref).unsqueeze(0))
    return rows, rhss


def _make_interp_spline_1d(
    x: Tensor,
    y: Tensor,
    k: int = 3,
    t: Tensor | None = None,
    bc_type: BcType = None,
) -> tuple[Tensor, Tensor]:
    """Return knot vector and coefficients for a 1D interpolating spline.

    :param Tensor x: Sample abscissae, shape ``(n,)``.
    :param Tensor y: Sample ordinates, shape ``(n,)`` or ``(n, *trailing)``.
    :param int k: Spline degree. Default: ``3``.
    :param Tensor t: Custom knot vector. Default: ``None``.
    :param bc_type: Boundary conditions. Default: ``None``.
    :return: Knot vector and B-spline coefficients with shape
        ``(nt, *trailing)``, where ``nt = t.numel() - k - 1``.
    :rtype: tuple[Tensor, Tensor]
    """
    x = torch.as_tensor(x, dtype=y.dtype, device=y.device).reshape(-1)
    y = torch.as_tensor(y)
    if y.shape[0] != x.numel():
        raise ValueError(f'x has {x.numel()} points but y has shape {tuple(y.shape)}.')
    _validate_strictly_increasing(x)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError('x and y must be finite.')
    if bc_type == 'periodic':
        raise ValueError("Boundary condition 'periodic' is not supported.")

    trailing = tuple(y.shape[1:])

    if k == 0:
        if t is not None or (bc_type not in (None, 'not-a-knot')):
            raise ValueError('For k=0 only not-a-knot without custom knots is supported.')
        return torch.cat([x, x[-1:]]), y.clone()

    if k == 1 and t is None and bc_type in (None, 'not-a-knot'):
        return torch.cat([x[:1], x, x[-1:]]), y.clone()

    bc_left, bc_right = _normalize_bc(bc_type, 1, trailing)[0]
    t = _make_knots_1d(x, k, bc_left, bc_right, t)
    n, nt = x.numel(), _num_coeffs(t, k)

    deriv_l, deriv_r = bc_left or [], bc_right or []
    if nt - n != len(deriv_l) + len(deriv_r):
        raise ValueError(
            f'Knot/coefficient count mismatch: nt={nt}, n={n}, '
            f'expected {len(deriv_l)}+{len(deriv_r)} boundary conditions.'
        )

    y_flat = y.reshape(n, -1)
    rows_l, rhs_l = _bc_constraint_rows(t, k, float(x[0]), nt, deriv_l, y_flat[0])
    rows_r, rhs_r = _bc_constraint_rows(t, k, float(x[-1]), nt, deriv_r, y_flat[0])
    mat = torch.cat(rows_l + [_design_matrix(x, t, k)] + rows_r)
    rhs = torch.cat(rhs_l + [y_flat] + rhs_r)
    coef = torch.linalg.solve(mat, rhs)
    return t, coef.reshape((nt,) + trailing)


def _contract_bases(c: Tensor, bases: list[Tensor], grid: bool = True) -> Tensor:
    """Contract ``c`` with one design matrix per dimension."""
    out = c
    for d, B in enumerate(bases):
        if grid or d == 0:
            out = torch.movedim(out, d, 0)
            n_d = out.shape[0]
            out = (B @ out.reshape(n_d, -1)).reshape(B.shape[0], *out.shape[1:])
            if grid:
                out = torch.movedim(out, 0, d)
        else:
            b_shape = [1] * out.ndim
            b_shape[0] = B.shape[0]
            b_shape[d] = B.shape[1]
            out = (out * B.reshape(b_shape)).sum(dim=d)
    return out


def _apply_derivatives(
    c: Tensor,
    t: Sequence[Tensor],
    k: Sequence[int],
    nu: Sequence[int],
) -> tuple[Tensor, list[Tensor], list[int]]:
    """Differentiate B-spline coefficients along each dimension."""
    c_out = c
    t_out = list(t)
    k_out = list(k)
    for d, nud in enumerate(nu):
        if nud == 0:
            continue
        c_out = c_out.movedim(d, 0)
        m = c_out.shape[0]
        rest = c_out.shape[1:]
        flat = c_out.reshape(m, -1)
        t_out[d], flat, k_out[d] = _bspline_derivative(t_out[d], flat, k_out[d], int(nud))
        c_out = flat.reshape((flat.shape[0],) + rest).movedim(0, d)
    return c_out, t_out, k_out


def _bc_type_for_dim(bc_type: BcType, dim: int, ndim: int) -> BcType:
    """Select the boundary-condition spec for one dimension."""
    if bc_type is None or isinstance(bc_type, str) or _is_deriv_spec(bc_type):
        return bc_type
    return bc_type[dim] if len(bc_type) == ndim else bc_type


def _separable_axes(xi: Tensor) -> list[Tensor] | None:
    """If ``xi`` is a rectilinear grid, return the 1D axis coordinates."""
    ndim = xi.shape[-1]
    grid_shape = xi.shape[:-1]
    if len(grid_shape) == 0:
        return None
    axes: list[Tensor] = []
    for d in range(ndim):
        v = xi[..., d]
        if v.shape != grid_shape:
            return None
        for ax in range(len(grid_shape)):
            if ax == d:
                continue
            if not torch.all(v == v.select(ax, 0).unsqueeze(ax)):
                return None
        sl = [slice(None) if ax == d else 0 for ax in range(len(grid_shape))]
        axes.append(v[tuple(sl)].reshape(-1))
    return axes


# ---------------------------------------------------------------------------
# N-d B-spline class
# ---------------------------------------------------------------------------


class NdBSpline:
    r"""
    N-dimensional tensor-product B-spline.

    In two dimensions, with :math:`\mathbf{x} = (x, y)` and degrees
    :math:`k_x`, :math:`k_y`:

    .. math::

        S(x, y) = \sum_{i=0}^{n_x - 1} \sum_{j=0}^{n_y - 1}
            c_{ij}\, B_{i, k_x}(x)\, B_{j, k_y}(y)

    Each 1D basis function is defined by the Cox--de Boor recursion

    .. math::

        B_{i, k}(x) =
            \frac{x - t_i}{t_{i+k} - t_i}\, B_{i, k-1}(x)
            + \frac{t_{i+k+1} - x}{t_{i+k+1} - t_{i+1}}\, B_{i+1, k-1}(x)

    with :math:`B_{i, 0}(x) = 1` on :math:`[t_i, t_{i+1})` and zero elsewhere.
    Higher dimensions are defined analogously as products of 1D basis functions.

    :param t: Knot vectors, one per dimension; ``t[d]`` has shape ``(m_d,)``.
    :type t: Sequence[Tensor]
    :param Tensor c: B-spline coefficients, shape
        ``(n_1, n_2, ..., n_N, *trailing)`` where ``n_d = t[d].numel() - k[d] - 1``.
    :param k: Spline degree per dimension (scalar or length-``N`` list).
    :type k: int | Sequence[int]
    :param bool extrapolate: Whether to extrapolate outside the knot span.
        Default: ``True``.
    """

    def __init__(
        self,
        t: Sequence[Tensor],
        c: Tensor,
        k: int | Sequence[int],
        extrapolate: bool = True,
    ):
        self._t = [torch.as_tensor(ti) for ti in t]
        self._k = _normalize_k(k, len(self._t))
        self.c = torch.as_tensor(c)  #: B-spline coefficients; see class docstring.
        self.extrapolate = bool(extrapolate)  #: Whether to extrapolate outside the knot span.
        for d, (td, kd) in enumerate(zip(self._t, self._k)):
            n = _num_coeffs(td, kd)
            if self.c.shape[d] != n:
                raise ValueError(
                    f'Dimension {d}: expected {n} coefficients, got {self.c.shape[d]}.'

                )

    @property
    def t(self) -> list[Tensor]:
        """Knot vectors, one per dimension."""
        return self._t

    @property
    def k(self) -> list[int]:
        """Spline degree per dimension."""
        return self._k

    def _design_matrices(
        self,
        coords: Sequence[Tensor],
        nu: Sequence[int] | None = None,
    ) -> list[Tensor]:
        nu = nu or (0,) * len(self._t)
        bases = []
        for coord, td, kd, nud in zip(coords, self._t, self._k, nu):
            x = torch.as_tensor(coord, dtype=td.dtype, device=self.c.device).reshape(-1)
            bases.append(_design_matrix(x, td, kd, int(nud)).to(self.c.dtype))
        return bases

    @classmethod
    def from_grid(
        cls,
        points: Sequence[Tensor],
        values: Tensor,
        k: int | Sequence[int] = 3,
        bc_type: BcType = 'not-a-knot',
        extrapolate: bool = True,
    ) -> NdBSpline:
        """
        Build an interpolating spline from rectilinear grid samples.

        :param points: Strictly increasing 1D coordinate vectors;
            ``points[d]`` has shape ``(n_d,)``.
        :type points: Sequence[Tensor]
        :param Tensor values: Sample values on the full grid, shape
            ``(n_1, n_2, ..., n_N, *trailing)``.
        :param k: Spline degree per dimension (scalar or length-``N`` list).
            Default: ``3``.
        :type k: int | Sequence[int]
        :param bc_type: Boundary condition for all dimensions, per dimension, or
            ``(left, right)`` applied to every dimension. Supports
            ``'not-a-knot'``, ``'clamped'``, ``'natural'``, and explicit
            derivative specs ``[(order, value), ...]``. Default: ``'not-a-knot'``.
        :param bool extrapolate: Whether to extrapolate outside the knot span.
            Default: ``True``.
        :return: Interpolating spline.
        :rtype: NdBSpline
        """
        vals = torch.as_tensor(values)
        pts = [torch.as_tensor(p, dtype=vals.dtype, device=vals.device).reshape(-1) for p in points]
        ndim = len(pts)
        if vals.ndim < ndim:
            raise ValueError(
                f'values must have at least {ndim} dimensions, got shape {tuple(vals.shape)}.'
            )
        ks = _normalize_k(k, ndim)

        for d, (p, kd) in enumerate(zip(pts, ks)):
            if p.numel() <= kd:
                raise ValueError(
                    f'Dimension {d}: need at least {kd + 1} points for degree {kd}.'
                )
            if vals.shape[d] != p.numel():
                raise ValueError(
                    f'Dimension {d}: values has length {vals.shape[d]}, '
                    f'but points has {p.numel()}.'
                )
            _validate_strictly_increasing(p, f'points[{d}]')

        coef = vals.clone()
        knots: list[Tensor] = []
        for d in range(ndim):
            coef = coef.movedim(d, 0)
            m = coef.shape[0]
            rest = coef.shape[1:]
            flat = coef.reshape(m, -1)
            t, c = _make_interp_spline_1d(
                pts[d], flat, k=ks[d], bc_type=_bc_type_for_dim(bc_type, d, ndim),
            )
            knots.append(t)
            coef = c.reshape((c.shape[0],) + rest).movedim(0, d)
        return cls(knots, coef, ks, extrapolate=extrapolate)

    def __call__(
        self,
        xi: Tensor,
        nu: Sequence[int] | None = None,
        extrapolate: bool | None = None,
    ) -> Tensor:
        """Evaluate at query points.

        :param Tensor xi: Query coordinates, shape ``(..., N)`` where ``N`` is
            the number of dimensions.
        :param nu: Derivative order per dimension, length ``N``.
            Default: all zeros.
        :type nu: Sequence[int] | None
        :param bool extrapolate: Override :attr:`.extrapolate`. Default: ``None``.
        :return: Interpolated values, shape ``(..., *trailing)`` where
            ``trailing`` denotes extra dimensions appended to :attr:`.c`.
        :rtype: Tensor
        """
        extrapolate = self.extrapolate if extrapolate is None else extrapolate
        xi = torch.as_tensor(xi, dtype=self._t[0].dtype, device=self.c.device)
        if len(self._t) == 1 and xi.ndim == 1:
            xi = xi.unsqueeze(-1)
        if xi.shape[-1] != len(self._t):
            raise ValueError(
                f'Expected last dim {len(self._t)}, got {xi.shape[-1]}. '
                f'For 1D splines, pass shape ``(..., 1)`` or a 1D tensor.'
            )
        nu = nu or (0,) * len(self._t)
        if len(nu) != len(self._t):
            raise ValueError('nu must have one entry per dimension.')

        axes = _separable_axes(xi)
        if axes is not None and all(n == 0 for n in nu):
            out = self.evaluate_grid(axes, extrapolate=extrapolate)
            return out.reshape(xi.shape[:-1] + self.c.shape[len(self._t):])

        pts = xi.reshape(-1, xi.shape[-1])
        c, t, k = _apply_derivatives(self.c, self._t, self._k, nu)
        out = _contract_bases(
            c,
            [
                _design_matrix(
                    torch.as_tensor(pts[:, d], dtype=t[d].dtype, device=self.c.device).reshape(-1),
                    t[d],
                    k[d],
                    0,
                ).to(self.c.dtype)
                for d in range(len(self._t))
            ],
            grid=False,
        )
        out = out.reshape(xi.shape[:-1] + self.c.shape[len(self._t):])
        if not extrapolate:
            out = _mask_query_points(out, pts, self._t, self._k, xi.shape[:-1])
        return out

    def evaluate_grid(
        self,
        grid_points: Sequence[Tensor],
        extrapolate: bool | None = None,
    ) -> Tensor:
        """Evaluate on a rectilinear grid.

        :param grid_points: 1D coordinate vectors per dimension;
            ``grid_points[d]`` has shape ``(m_d,)``.
        :type grid_points: Sequence[Tensor]
        :param bool extrapolate: Override :attr:`.extrapolate`. Default: ``None``.
        :return: Values on the rectilinear grid, shape
            ``(m_1, m_2, ..., m_N, *trailing)``.
        :rtype: Tensor
        """
        extrapolate = self.extrapolate if extrapolate is None else extrapolate
        if len(grid_points) != len(self._t):
            raise ValueError(
                f'Expected {len(self._t)} grid axes, got {len(grid_points)}.'
            )
        axes = [
            torch.as_tensor(g, dtype=td.dtype, device=self.c.device).reshape(-1)
            for g, td in zip(grid_points, self._t)
        ]
        out = _contract_bases(self.c, self._design_matrices(axes))
        if not extrapolate:
            out = _mask_outside_span(out, axes, self._t, self._k)
        return out
