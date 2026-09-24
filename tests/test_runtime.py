"""runtime 层注册表与接口契约的单元测试。

跑法::

    .\\.venv\\Scripts\\python.exe -m pip install -e .
    .\\.venv\\Scripts\\python.exe -m pytest -q
"""

from __future__ import annotations

import pytest

from lingmou.runtime import registry
from lingmou.runtime.base import BackendError, InferenceBackend


# ── 注册表 ────────────────────────────────────────────────

def test_ort_backend_is_registered() -> None:
    """导入 runtime 包后，ORT 后端应自动注册。"""
    assert "ort" in registry.available()


def test_create_returns_backend_instance() -> None:
    backend = registry.create("ort", providers=["CPUExecutionProvider"])
    assert isinstance(backend, InferenceBackend)
    assert backend.ready is False          # 未 load() 前不算就绪
    assert backend.providers == ["CPUExecutionProvider"]


def test_create_unknown_backend_raises() -> None:
    with pytest.raises(KeyError):
        registry.create("no-such-backend")


def test_default_provider_is_cpu() -> None:
    """不指定 provider 时必须默认 CPU —— 避免开发机无意中依赖 CUDA。"""
    backend = registry.create("ort")
    assert backend.providers == ["CPUExecutionProvider"]


# ── @register 校验 ────────────────────────────────────────

def test_register_rejects_missing_name() -> None:
    class NoName(InferenceBackend):
        def load(self, model_path, **kwargs): ...
        def infer(self, inputs): ...
        def describe(self): ...

    with pytest.raises(ValueError, match="name"):
        registry.register(NoName)


def test_register_rejects_duplicate_name() -> None:
    class Dup(InferenceBackend):
        name = "ort"                       # 与已注册的 ORTBackend 冲突

        def load(self, model_path, **kwargs): ...
        def infer(self, inputs): ...
        def describe(self): ...

    with pytest.raises(ValueError, match="冲突"):
        registry.register(Dup)


# ── 后端行为（需要 onnxruntime） ──────────────────────────

def test_unsupported_provider_raises_backend_error() -> None:
    pytest.importorskip("onnxruntime")
    backend = registry.create("ort", providers=["NotARealProvider"])
    with pytest.raises(BackendError, match="不可用"):
        backend.load("dummy.onnx")


def test_infer_before_load_raises() -> None:
    backend = registry.create("ort")
    with pytest.raises(BackendError, match="load"):
        backend.infer({"images": None})
