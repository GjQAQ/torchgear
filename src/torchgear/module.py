from functools import partial
import typing

import torch
from torch import nn

__all__ = [
    'DeviceMixIn',
    'DtypeMixIn',
    'FreezeParamMixIn',
    'TensorAsDelegate',
    'TensorContainerMixIn',
    'WrapperModule',
]

_Ts = torch.Tensor


class WrapperModule(nn.Module):
    """
    A class to wrap a function as a :py:class:`torch.nn.Module`.

    .. doctest::
        :skipif: True

        >>> s = WrapperModule(torch.sum, dim=(-2, -1))
        >>> x = torch.rand(4)
        >>> s(x)  # equivalent to torch.sum(x, dim=(-2, -1))

    :param Callable func: The function to be wrapped.
    :param args: Positional arguments to be passed to ``func`` when this module is called.
    :param kwargs: Keyword arguments to be passed to ``func`` when this module is called.
    """

    def __init__(self, func: typing.Callable, *args, **kwargs):
        super().__init__()
        self._impl = partial(func, *args, **kwargs)

    def forward(self, *args, **kwargs):
        """
        Call the wrapped function ``func``.

        :param args: Additional positional arguments to be passed to ``func``.
        :param kwargs: Additional keyword arguments to be passed to ``func``.
        :return: The returned value of the wrapped function.
        :rtype: Any
        """
        return self._impl(*args, **kwargs)


def _check_consistency(attr: str, obj, ts: _Ts, error: bool) -> bool:
    v1, v2 = getattr(obj, attr), getattr(ts, attr)
    if v1 != v2:
        if error:
            raise RuntimeError(f'{attr.capitalize()} mismatch: {v1} for an instance of '
                               f'{obj.__class__.__name__} while {v2} for an incoming tensor')
        else:
            return False
    return True


class TensorAsDelegate:
    """
    A mixin that creates :py:class:`torch.Tensor` s on the same device and dtype as
    a delegate tensor associated with ``self``.

    When mixed into a :py:class:`torch.nn.Module` subclass, the delegate is taken from
    the first registered parameter or buffer. If none exists, a stub buffer is
    registered lazily on first use. Subclasses that are not derived from
    :py:class:`torch.nn.Module` must implement :meth:`._delegate`.
    """

    def new_tensor(self, data, **kwargs) -> _Ts:
        """Call ``torch.Tensor.new_tensor`` on the delegate tensor."""
        return self._delegate().new_tensor(data, **kwargs)

    def new_full(self, size, fill_value, **kwargs) -> _Ts:
        """Call ``torch.Tensor.new_full`` on the delegate tensor."""
        return self._delegate().new_full(size, fill_value, **kwargs)

    def new_empty(self, size, **kwargs) -> _Ts:
        """Call ``torch.Tensor.new_empty`` on the delegate tensor."""
        return self._delegate().new_empty(size, **kwargs)

    def new_ones(self, size, **kwargs) -> _Ts:
        """Call ``torch.Tensor.new_ones`` on the delegate tensor."""
        return self._delegate().new_ones(size, **kwargs)

    def new_zeros(self, size, **kwargs) -> _Ts:
        """Call ``torch.Tensor.new_zeros`` on the delegate tensor."""
        return self._delegate().new_zeros(size, **kwargs)

    def arange(self, *args, **kwargs) -> _Ts:
        """Call ``torch.arange`` with the device and dtype of the delegate tensor."""
        d = self._delegate()
        return torch.arange(*args, **kwargs, device=d.device, dtype=d.dtype)

    def linspace(self, *args, **kwargs) -> _Ts:
        """Call ``torch.linspace`` with the device and dtype of the delegate tensor."""
        d = self._delegate()
        return torch.linspace(*args, **kwargs, device=d.device, dtype=d.dtype)

    def rand(self, *args, **kwargs) -> _Ts:
        """Call ``torch.rand`` with the device and dtype of the delegate tensor."""
        d = self._delegate()
        return torch.rand(*args, **kwargs, device=d.device, dtype=d.dtype)

    def randn(self, *args, **kwargs) -> _Ts:
        """Call ``torch.randn`` with the device and dtype of the delegate tensor."""
        d = self._delegate()
        return torch.randn(*args, **kwargs, device=d.device, dtype=d.dtype)

    def _delegate(self) -> _Ts:
        if not isinstance(self, nn.Module):
            raise NotImplementedError(f'A subclass of {TensorAsDelegate.__name__} that is not derived from '
                                      f'torch.nn.Module must implement {self._delegate.__name__} method')
        # register stub dynamically to avoid calling __init__
        if not hasattr(self, '_delegate_tensor'):
            t = None
            for p in self.parameters():
                t = p
                break
            if t is None:
                for b in self.buffers():
                    t = b
                    break
            if t is None:
                t = torch.tensor([])
            self.register_buffer('_delegate_tensor', t.new_tensor([]), False)
        return self._delegate_tensor


class DeviceMixIn(TensorAsDelegate):
    """
    Some :py:class:`torch.Tensor` s may be associated to objects of the class
    (e.g. buffers and parameters of :py:class:`torch.nn.Module`)
    derived from this class. They are assumed to be on the same device,
    which is the value of :attr:`device`.
    """

    def _check_consistency(self, ts: _Ts, error: bool = True) -> bool:
        return _check_consistency('device', self, ts, error)

    def _cast(self, ts: _Ts) -> _Ts:
        return ts.to(device=self.device)

    @property
    def device(self) -> torch.device:
        """
        Device of this object.

        :type: :py:class:`torch.device`
        """
        dlg = self._delegate()
        # torch.get_default_device() is not available for old versions
        return torch.tensor(0.).device if dlg is None else dlg.device


class DtypeMixIn(TensorAsDelegate):
    """
    Some :py:class:`torch.Tensor` s may be associated to objects of the class
    (e.g. buffers and parameters of :py:class:`torch.nn.Module`)
    derived from this class. They are assumed to have same data type,
    which is the value of :attr:`dtype`.
    """

    def _check_consistency(self, ts: _Ts, error: bool = True) -> bool:
        return _check_consistency('dtype', self, ts, error)

    def _cast(self, ts: _Ts) -> _Ts:
        return ts.to(dtype=self.dtype)

    @property
    def dtype(self) -> torch.dtype:
        """
        Data type of this object.

        :type: :py:class:`torch.dtype`
        """
        dlg = self._delegate()
        return torch.get_default_dtype() if dlg is None else dlg.dtype


class TensorContainerMixIn(DeviceMixIn, DtypeMixIn):
    """
    A mixin combining :class:`DeviceMixIn` and :class:`DtypeMixIn`.

    Tensors associated with objects of a class derived from this mixin (e.g. parameters
    and buffers of a :py:class:`torch.nn.Module`) are assumed to share the same
    :attr:`device` and :attr:`dtype`. :meth:`._check_consistency` and :meth:`._cast`
    validate and align incoming tensors against both attributes.
    """
    def _check_consistency(self, ts: _Ts, error: bool = True) -> bool:
        return (_check_consistency('device', self, ts, error) and
                _check_consistency('dtype', self, ts, error))

    def _cast(self, ts: _Ts) -> _Ts:
        return ts.to(device=self.device, dtype=self.dtype)


class FreezeParamMixIn:
    def freeze(self, name: str | typing.Sequence[str] = None):
        """Equivalent to ``self.set_optimizable(name, False)``. See :meth:`.set_optimizable`."""
        self.set_optimizable(name, False)

    def unfreeze(self, name: str | typing.Sequence[str] = None):
        """Equivalent to ``self.set_optimizable(name, True)``. See :meth:`.set_optimizable`."""
        self.set_optimizable(name, True)

    def set_optimizable(self, name: str | typing.Sequence[str] = None, optimizable: bool = True):
        """
        Specify whether a parameter is optimizable.

        :param str name: Name of the parameter. If ``None``, all parameters will be set.
            It follows the same convention as :meth:`torch.nn.Module.get_parameter`.
        :param bool optimizable: Whether the specified parameter is optimizable. Default: ``True``.
        """
        if not isinstance(self, nn.Module):
            raise RuntimeError(
                f'{FreezeParamMixIn.__name__} must be used with a subclass of '
                f'torch.nn.Module, got {self.__class__.__name__}'
            )
        if name is None:
            for p in self.parameters():
                p.requires_grad = optimizable
        elif isinstance(name, str):
            param = self.get_parameter(name)
            param.requires_grad = optimizable
        else:
            for n in name:
                param = self.get_parameter(n)
                param.requires_grad = optimizable


class TorchgearModule(
    nn.Module,
    TensorContainerMixIn,
    FreezeParamMixIn,
):
    """
    Base class for torchgear models.
    """
    pass
