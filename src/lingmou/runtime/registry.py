"""后端注册表：按名字创建推理后端。

用法::

    from lingmou.runtime import registry

    backend = registry.create("ort", providers=["CPUExecutionProvider"])
    backend.load("artifacts/model.onnx")
    outputs = backend.infer({"images": x})

新增芯片后端时，在 ``runtime/`` 下新增一个模块，用 ``@register`` 装饰类即可，
**不需要修改任何业务代码**。这正是"换芯片 = 换执行后端"的落地方式。
"""

from __future__ import annotations

from typing import Any

from .base import InferenceBackend

_REGISTRY: dict[str, type[InferenceBackend]] = {}


def register(cls: type[InferenceBackend]) -> type[InferenceBackend]:
    """类装饰器：将后端登记到注册表。"""
    name = getattr(cls, "name", None)
    if not name or name == "base":
        raise ValueError(f"{cls.__name__} 必须声明唯一的类属性 name")
    if name in _REGISTRY:
        raise ValueError(f"后端名冲突：{name!r} 已被 {_REGISTRY[name].__name__} 注册")
    _REGISTRY[name] = cls
    return cls


def available() -> list[str]:
    """返回已注册的后端名（排序后）。"""
    return sorted(_REGISTRY)


def create(name: str, **options: Any) -> InferenceBackend:
    """按名字创建后端实例。"""
    if name not in _REGISTRY:
        raise KeyError(f"未注册的后端 {name!r}；当前可用：{available()}")
    return _REGISTRY[name](**options)
