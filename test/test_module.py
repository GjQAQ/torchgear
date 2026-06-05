import pytest
import torch
import torch.nn as nn

from torchgear.module import (
    DeviceMixIn,
    DtypeMixIn,
    FreezeParamMixIn,
    TensorAsDelegate,
    TensorContainerMixIn,
    TorchgearModule,
    WrapperModule,
)


class _DeviceModel(nn.Module, DeviceMixIn):
    def __init__(self, tensor: torch.Tensor):
        super().__init__()
        self.register_parameter('w', nn.Parameter(tensor))


class _DtypeModel(nn.Module, DtypeMixIn):
    def __init__(self, tensor: torch.Tensor):
        super().__init__()
        self.register_parameter('w', nn.Parameter(tensor))


class _ContainerModel(nn.Module, TensorContainerMixIn):
    def __init__(self, tensor: torch.Tensor):
        super().__init__()
        self.register_parameter('w', nn.Parameter(tensor))


class _FreezeModel(nn.Module, FreezeParamMixIn):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(2))
        self.b = nn.Parameter(torch.randn(2))


class _TorchgearModel(TorchgearModule):
    def __init__(self, tensor: torch.Tensor):
        super().__init__()
        self.w = nn.Parameter(tensor)


class _BareDelegate(TensorAsDelegate):
    pass


def test_wrapper_module_wraps_function():
    module = WrapperModule(torch.sum, dim=0)
    x = torch.tensor([1.0, 2.0, 3.0])
    torch.testing.assert_close(module(x), torch.tensor(6.0))


def test_tensor_as_delegate_uses_parameter_device_and_dtype():
    param = torch.randn(2, dtype=torch.float64)
    model = _ContainerModel(param)
    y = model.arange(3)
    assert y.device == param.device
    assert y.dtype == param.dtype
    torch.testing.assert_close(y, torch.arange(3, dtype=torch.float64))


def test_tensor_as_delegate_uses_buffer_when_no_parameters():
    class BufferModel(nn.Module, DtypeMixIn):
        def __init__(self):
            super().__init__()
            self.register_buffer('b', torch.zeros(1, dtype=torch.float16))

    model = BufferModel()
    assert model.dtype == torch.float16


def test_tensor_as_delegate_requires_delegate_without_module():
    with pytest.raises(NotImplementedError, match='must implement _delegate'):
        _BareDelegate().new_tensor([1.0])


def test_device_mixin_check_consistency():
    model = _DeviceModel(torch.randn(2))
    matching = torch.randn(2, device=model.device)
    other = torch.randn(2, device=torch.device('cpu'))
    if model.device != other.device:
        assert model._check_consistency(matching) is True
        assert model._check_consistency(other, error=False) is False
        with pytest.raises(RuntimeError, match='Device mismatch'):
            model._check_consistency(other)


def test_device_mixin_cast():
    model = _DeviceModel(torch.randn(2))
    ts = torch.randn(2)
    casted = model._cast(ts)
    assert casted.device == model.device


def test_dtype_mixin_check_consistency():
    model = _DtypeModel(torch.randn(2, dtype=torch.float64))
    matching = torch.randn(2, dtype=torch.float64)
    other = torch.randn(2, dtype=torch.float32)
    assert model._check_consistency(matching) is True
    assert model._check_consistency(other, error=False) is False
    with pytest.raises(RuntimeError, match='Dtype mismatch'):
        model._check_consistency(other)


def test_dtype_mixin_cast():
    model = _DtypeModel(torch.randn(2, dtype=torch.float64))
    casted = model._cast(torch.randn(2, dtype=torch.float32))
    assert casted.dtype == torch.float64


def test_tensor_container_mixin_check_consistency_and_cast():
    model = _ContainerModel(torch.randn(2, dtype=torch.float64))
    matching = torch.randn(2, dtype=torch.float64)
    wrong_dtype = torch.randn(2, dtype=torch.float32)
    assert model._check_consistency(matching) is True
    assert model._check_consistency(wrong_dtype, error=False) is False
    casted = model._cast(wrong_dtype)
    assert casted.dtype == torch.float64
    assert casted.device == model.device


def test_freeze_param_mixin_all_parameters():
    model = _FreezeModel()
    model.freeze()
    assert all(not p.requires_grad for p in model.parameters())
    model.unfreeze()
    assert all(p.requires_grad for p in model.parameters())


def test_freeze_param_mixin_by_name():
    model = _FreezeModel()
    model.freeze('a')
    assert not model.a.requires_grad
    assert model.b.requires_grad
    model.unfreeze(['a'])
    assert model.a.requires_grad


def test_freeze_param_mixin_requires_module_subclass():
    class Bad(FreezeParamMixIn):
        pass

    with pytest.raises(RuntimeError, match='must be used with a subclass of'):
        Bad().freeze()


def test_torchgear_module_combines_mixins():
    param = torch.randn(2, dtype=torch.float64)
    model = _TorchgearModel(param)
    assert model.dtype == torch.float64
    model.freeze()
    assert not model.w.requires_grad
    y = model.new_zeros(2)
    assert y.dtype == torch.float64
    assert y.device == param.device
