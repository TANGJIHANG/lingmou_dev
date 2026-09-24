"""国产算力适配层。

统一推理接口 + 后端注册表。业务代码只应通过本包使用推理能力::

    from lingmou.runtime import registry

    backend = registry.create("ort", providers=["CPUExecutionProvider"])
    backend.load("artifacts/model.onnx")
    backend.warmup({"images": x})
    outputs = backend.infer({"images": x})
    print(backend.describe())        # → 《国产化适配测试报告》环境清单表
"""

from .base import BackendError, InferenceBackend
from .registry import available, create, register

# 导入具体后端以触发 @register 注册。新增芯片后端时在此追加一行即可。
from . import ort_backend  # noqa: F401  (import 副作用：注册)

__all__ = [
    "InferenceBackend",
    "BackendError",
    "register",
    "create",
    "available",
]
