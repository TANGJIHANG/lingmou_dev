"""ONNX Runtime 后端 —— 通过替换 providers 实现跨芯片。

这是"换芯片 = 换执行后端"最直接的落点::

    ORTBackend(providers=["CPUExecutionProvider"])    # 开发机 / 精度对齐
    ORTBackend(providers=["CUDAExecutionProvider"])   # 本机 NVIDIA 卡（仅开发期）
    ORTBackend(providers=["CANNExecutionProvider"])   # 昇腾板卡

**换芯片只改这一行参数，业务代码完全不动。**

注意
----
- ``onnxruntime`` 是延迟导入的：未安装本模块也能被 import，只有在 ``load()`` 时才报错。
- CANN 后端需要安装 onnxruntime 的昇腾构建版本，且**仅有 Linux 版本**。
- 若某个算子在 CANN 后端不受支持，退路是昇腾 ATC 转 OM + AscendCL 原生推理，
  届时应在本目录新增一个 ``cann_backend.py``，而不是去改业务代码。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .base import BackendError, InferenceBackend
from .registry import register


@register
class ORTBackend(InferenceBackend):
    """基于 ONNX Runtime 的推理后端。"""

    name = "ort"

    def __init__(
        self,
        providers: Sequence[str] | None = None,
        session_options: Any = None,
        **options: Any,
    ) -> None:
        super().__init__(**options)
        self.providers = list(providers) if providers else ["CPUExecutionProvider"]
        self._session_options = session_options
        self._sess = None
        self._model_path: str | None = None

    def load(self, model_path: str, **kwargs: Any) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - 取决于环境
            raise BackendError(
                "未安装 onnxruntime：pip install onnxruntime"
            ) from exc

        installed = ort.get_available_providers()
        missing = [p for p in self.providers if p not in installed]
        if missing:
            raise BackendError(
                f"请求的执行后端不可用：{missing}\n"
                f"当前 onnxruntime 已安装的后端：{installed}\n"
                "提示：CANN 后端需安装 onnxruntime 的昇腾构建版本，且仅支持 Linux。"
            )

        try:
            self._sess = ort.InferenceSession(
                model_path,
                sess_options=self._session_options,
                providers=self.providers,
                **kwargs,
            )
        except Exception as exc:  # pragma: no cover - 取决于模型
            raise BackendError(f"加载模型失败：{model_path}") from exc

        self._model_path = model_path
        self._loaded = True

    def infer(self, inputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        if not self._loaded or self._sess is None:
            raise BackendError("后端尚未 load()，无法推理")
        # inputs 的键须与模型输入名一致；ONNX Runtime 返回 list，用输出名重组为 dict
        values = self._sess.run(None, dict(inputs))
        names = [o.name for o in self._sess.get_outputs()]
        return dict(zip(names, values))

    def describe(self) -> dict[str, Any]:
        import onnxruntime as ort

        active = self._sess.get_providers() if self._loaded else []
        return {
            "backend": self.name,
            "runtime": f"onnxruntime {ort.__version__}",
            "requested_providers": self.providers,
            "active_providers": active,
            "installed_providers": ort.get_available_providers(),
            # 国产化自证字段：False 才符合命题"全面脱离 CUDA 生态"的要求
            "has_nvidia_components": any(
                ("CUDA" in p) or ("TensorRT" in p) for p in active
            ),
            "model_path": self._model_path,
        }
