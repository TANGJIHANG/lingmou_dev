"""D5 导出模块测试。

重点守两件事
------------
1. **导出切面**：``DetectorRawOutput`` 只暴露 backbone+FPN+head 的三个张量，
   形状与 torchvision 原生 forward 的中间结果一致。切面错了，后面 D6 的对齐就无从谈起。
2. **IR version 压制**：``onnx`` 与 ``onnxruntime`` 独立发版、IR 上限不同步。
   实测 onnx 1.23 默认写 IR 14，而 onnxruntime 1.23.2 只支持到 11。
   这条必须在**每次导出**时都生效，不能靠"我记得手动改过"。

完整导出（写出上百 MB 的 .onnx）耗时长，默认跳过；设 ``LINGMOU_SLOW_TESTS=1`` 才跑。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from lingmou.engine.export import (
    DEFAULT_IR_VERSION,
    DEFAULT_OPSET,
    OUTPUT_NAMES,
    DetectorRawOutput,
    ExportSpec,
    export_onnx,
    inspect_onnx,
)
from lingmou.engine.export import _force_ir_version
from lingmou.engine.model import build_detector

CLASSES = ["airplane", "ship", "vehicle"]


@pytest.fixture(scope="module")
def raw_wrapper():
    """小尺寸未预训练 FCOS 的可导出包装（模块级缓存，构建较慢）。"""
    model = build_detector("fcos", len(CLASSES), pretrained=False, min_size=64, max_size=128)
    return DetectorRawOutput(model).eval()


# ── 导出切面 ──────────────────────────────────────────────

def test_raw_output_returns_three_named_tensors(raw_wrapper) -> None:
    with torch.no_grad():
        out = raw_wrapper(torch.rand(1, 3, 128, 128))
    assert len(out) == 3
    cls_logits, bbox_reg, ctrness = out
    # 最后一维 = 类别数（含背景）；位置数由 H/W 决定
    assert cls_logits.shape == (1, cls_logits.shape[1], len(CLASSES) + 1)
    assert bbox_reg.shape == (1, cls_logits.shape[1], 4)
    assert ctrness.shape == (1, cls_logits.shape[1], 1)


def test_raw_output_location_count_scales_with_input(raw_wrapper) -> None:
    """位置数必须随输入尺寸变化 —— 这是"动态 H/W 能成立"的前提。"""
    with torch.no_grad():
        small = raw_wrapper(torch.rand(1, 3, 64, 64))[0]
        large = raw_wrapper(torch.rand(1, 3, 128, 128))[0]
    assert small.shape[1] < large.shape[1]
    assert OUTPUT_NAMES == ("cls_logits", "bbox_regression", "bbox_ctrness")


def test_raw_output_rejects_non_detector() -> None:
    with pytest.raises(TypeError, match="backbone"):
        DetectorRawOutput(torch.nn.Linear(3, 3))


# ── IR version 压制 ───────────────────────────────────────

def _tiny_onnx(path: Path, ir_version: int) -> Path:
    from onnx import TensorProto, helper

    node = helper.make_node("Add", ["a", "b"], ["c"])
    graph = helper.make_graph(
        [node], "tiny",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [1]),
        ],
        outputs=[helper.make_tensor_value_info("c", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = ir_version
    import onnx

    onnx.save(model, str(path))
    return path


def test_force_ir_version_downgrades(tmp_path: Path) -> None:
    """onnx 默认可能写出高于 onnxruntime 上限的 IR，导出流程必须压下来。"""
    import onnx

    p = _tiny_onnx(tmp_path / "hi.onnx", ir_version=14)
    result = _force_ir_version(p, 8)
    assert result["ir_version_before"] == 14
    assert result["ir_version_after"] == 8
    assert onnx.load(str(p)).ir_version == 8


def test_force_ir_version_never_raises_it(tmp_path: Path) -> None:
    """已低于目标时不得反而抬高 —— 抬高会破坏与旧运行时的兼容。"""
    p = _tiny_onnx(tmp_path / "lo.onnx", ir_version=7)
    _force_ir_version(p, 11)
    import onnx

    assert onnx.load(str(p)).ir_version == 7


def test_defaults_are_conservative() -> None:
    """默认值要保守：产物得能在**可能更旧的**板卡运行时上加载。"""
    assert DEFAULT_IR_VERSION <= 8
    assert DEFAULT_OPSET <= 17


# ── 自检与规格 ────────────────────────────────────────────

def test_inspect_onnx_reports_graph_facts(tmp_path: Path) -> None:
    p = _tiny_onnx(tmp_path / "t.onnx", ir_version=8)
    info = inspect_onnx(p)
    assert info["ir_version"] == 8
    assert info["inputs"] == ["a", "b"]
    assert info["outputs"] == ["c"]
    assert info["node_count"] == 1
    assert info["op_types"] == {"Add": 1}
    assert info["size_bytes"] > 0


def test_export_spec_json_roundtrip(tmp_path: Path) -> None:
    spec = ExportSpec(detector="fcos", num_classes=3, classes=("a", "b", "c"), dynamic_hw=True)
    p = spec.to_json(tmp_path / "s.json")
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["opset"] == DEFAULT_OPSET
    assert data["ir_version"] == DEFAULT_IR_VERSION
    assert data["classes"] == ["a", "b", "c"]
    assert data["dynamic_hw"] is True


# ── 完整导出（慢，默认跳过） ──────────────────────────────

@pytest.mark.skipif(
    not os.environ.get("LINGMOU_SLOW_TESTS"),
    reason="完整导出会写出上百 MB 的 .onnx；设 LINGMOU_SLOW_TESTS=1 才跑",
)
def test_full_export_loads_in_ort_and_keeps_dynamic_hw(tmp_path: Path) -> None:
    """端到端：导出 -> onnxruntime 加载 -> 用与 trace 不同的尺寸推理。

    这是 D5 验收 2、3 的自动化版本。动态轴若被某些算子固化，
    这条会在"换尺寸"那一步失败。
    """
    ort = pytest.importorskip("onnxruntime")
    model = build_detector("fcos", len(CLASSES), pretrained=False, min_size=64, max_size=128)
    out = tmp_path / "m.onnx"
    path, spec = export_onnx(model, out, spec=ExportSpec(sample_height=128, sample_width=128))

    info = inspect_onnx(path)
    assert info["ir_version"] == spec.ir_version

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    wrapper = DetectorRawOutput(model).eval()
    for h, w in [(128, 128), (96, 160)]:          # 第二个尺寸与 trace 不同
        x = torch.rand(1, 3, h, w)
        with torch.no_grad():
            torch_out = wrapper(x)
        ort_out = sess.run(None, {"images": x.numpy()})
        assert all(t.shape == o.shape for t, o in zip(torch_out, ort_out))
        diffs = [float(np.abs(t.numpy() - o).max()) for t, o in zip(torch_out, ort_out)]
        assert max(diffs) < 1e-4, f"{h}x{w} 数值不一致：{max(diffs):.3e}"
