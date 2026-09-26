"""后端冒烟测试工具的回归测试。

守的是什么
----------
**ONNX IR version 错配**。本项目实测：``onnx 1.23.0`` 默认写出 IR version **14**，
而 ``onnxruntime 1.23.2`` 最高只支持 **11**，报错为::

    Unsupported model IR version: 14, max supported IR version: 11

这是 D5（导出 ONNX）→ D6（ORT 推理）最典型的卡点，而且会在**换环境时突然出现**
（Windows 上 onnxruntime 1.23.2、WSL 上 1.30.0，支持的 IR 上限并不相同）。
所以必须有一条测试钉住"我们产出的模型确实能被当前 onnxruntime 读进去"。
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_TOOL = PROJECT_ROOT / "tools" / "backend_smoke.py"


def _load_smoke_module():
    """按路径加载 tools/backend_smoke.py（它不在包内，无法直接 import）。"""
    spec = importlib.util.spec_from_file_location("backend_smoke", SMOKE_TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_tool_exists() -> None:
    assert SMOKE_TOOL.is_file()


def test_built_model_ir_version_is_ort_loadable(tmp_path: Path) -> None:
    """产出的 ONNX 必须能被本机 onnxruntime 加载。

    直接钉住 IR version 这个坑：若有人把 ``model.ir_version`` 那行删掉，
    在 onnx 较新的环境上这条测试就会失败。
    """
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    smoke = _load_smoke_module()
    model_path = tmp_path / "smoke.onnx"
    x, w, b = smoke.build_tiny_model(model_path)

    # 先确认模型里的 IR 版本确实不高于本机 ORT 的上限（不依赖报错信息）
    import onnx

    ir = onnx.load(str(model_path)).ir_version
    assert ir <= 11, f"IR version {ir} 过高，旧版 onnxruntime 会拒绝加载"

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    out = session.run(None, {"X": x})[0]
    assert out.shape == (1, w.shape[1])
    assert abs(float(out[0, 0]) - float((x @ w + b)[0, 0])) < 1e-5


def test_smoke_script_runs_and_passes() -> None:
    """端到端跑一次脚本，退出码必须是 0。"""
    pytest.importorskip("onnxruntime")
    proc = subprocess.run(
        [sys.executable, str(SMOKE_TOOL)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    assert proc.returncode == 0, f"冒烟测试未通过：\n{proc.stdout}\n{proc.stderr}"
    assert "PASS" in proc.stdout
    assert "has_nvidia_components" in proc.stdout
