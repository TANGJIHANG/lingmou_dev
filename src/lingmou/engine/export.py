"""ONNX 导出（D5）。

核心设计：**只导出模型，不导出后处理**
--------------------------------------
torchvision 检测模型的 ``forward`` 返回 ``List[Dict[str, Tensor]]``
（每张图一个 dict，含 boxes / scores / labels），这是 **Python 对象**，
既有变长输出、又含 NMS 这类自定义逻辑，无法直接 trace 成计算图。

所以本模块把"导出什么"切在模型与后处理之间：

    导出：backbone + FPN + head   ->  cls_logits / bbox_regression / bbox_ctrness
    图外：anchor 生成 + 解码 + centerness + NMS  ->  boxes / scores / labels

这个切法有个**决定性好处**：D6 要求"逐框与 PyTorch 对齐"。
两侧共用同一套后处理代码后，差异就只可能来自模型图本身，
而不是"NMS 在两边实现得不一样"这类噪声。**先排除后处理变量，再谈精度对齐。**

⚠️ 代价（必须记账）：导出的 ``.onnx`` **不是**一个可以直接喂图片、
直接吐框的成品。谁要用它，谁就得自己实现后处理。
终态（真上板）必须把后处理也做进去或重写一份无 torch 依赖的实现——
那是 D6/D7 的事，不是 D5。

两个已经踩过的坑
----------------
1. **导出器**：``torch.onnx.export`` 在本机 torch 2.14 上默认 ``dynamo=True``，
   而它依赖 ``onnxscript``——**本环境没装**，不显式传 ``dynamo=False`` 会直接失败。
2. **IR version**：``onnx`` 与 ``onnxruntime`` 独立发版，IR 上限不同步。
   实测 onnx 1.23 默认写 IR 14，而 onnxruntime 1.23.2 只支持到 11。
   因此 :func:`export_onnx` **导出后强制压低** ``ir_version``。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

#: IR version 上限。取 8（opset 17 要求 IR >= 8），远低于当前 onnxruntime 的上限，
#: 目的是让产物在**板子上可能更旧的** onnxruntime 上也能加载。
DEFAULT_IR_VERSION = 8
#: opset 17：Conv/Resize/Concat/Reshape/Sigmoid/Exp 等 FCOS 用到的算子全都成熟支持，
#: 且远低于各后端的支持上限。不要盲目追新 opset——板卡后端往往滞后。
DEFAULT_OPSET = 17

INPUT_NAME = "images"
OUTPUT_NAMES = ("cls_logits", "bbox_regression", "bbox_ctrness")


@dataclass
class ExportSpec:
    """导出规格。**D6 必须用同一份规格**，否则两侧输入不同、对齐无意义。"""

    opset: int = DEFAULT_OPSET
    ir_version: int = DEFAULT_IR_VERSION
    dynamic_hw: bool = True
    sample_height: int = 600
    sample_width: int = 800
    detector: str = ""
    num_classes: int = 0          # 前景类别数
    classes: tuple[str, ...] = ()

    def to_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        return p


class DetectorRawOutput(nn.Module):
    """把 torchvision 检测模型切成"可导出的前半段"。

    ``backbone + head`` 输出三个原始张量，形状为 ``[N, 位置数, C]``；
    位置数与输入 H/W 相关，故动态轴导出时它是动态的。
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        if not (hasattr(model, "backbone") and hasattr(model, "head")):
            raise TypeError("传入的模型没有 backbone/head，本包装只支持 torchvision 检测模型")
        self.backbone = model.backbone
        self.head = model.head

    def forward(self, images: torch.Tensor):
        features = self.backbone(images)
        # BackboneWithFPN 返回 OrderedDict；按 level 顺序取 value 即可（head 不关心键名）
        feats = list(features.values()) if isinstance(features, dict) else features
        head_out = self.head(feats)
        return (
            head_out["cls_logits"],
            head_out["bbox_regression"],
            head_out["bbox_ctrness"],
        )


def _dynamic_axes(dynamic_hw: bool) -> dict[str, dict[int, str]]:
    axes: dict[str, dict[int, str]] = {INPUT_NAME: {0: "batch"}}
    if dynamic_hw:
        # 字母序必须是 height < width（onnx 对动态轴命名无要求，但保持可读）
        axes[INPUT_NAME][2] = "height"
        axes[INPUT_NAME][3] = "width"
    for name in OUTPUT_NAMES:
        axes[name] = {0: "batch", 1: "num_locations"}
    return axes


def export_onnx(
    model: nn.Module,
    path: str | Path,
    *,
    spec: ExportSpec | None = None,
) -> tuple[Path, ExportSpec]:
    """把检测模型的前半段导出为 ONNX。

    :param model: 已 ``load_state_dict`` 的 torchvision 检测模型（本函数会切出 backbone+head）
    :param path: 输出 ``.onnx`` 路径
    :return: ``(onnx 路径, 实际使用的 ExportSpec)``
    """
    spec = spec or ExportSpec()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    wrapper = DetectorRawOutput(model).eval()
    dummy = torch.randn(1, 3, spec.sample_height, spec.sample_width)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(path),
            # 必须显式 False：本机 torch 2.14 默认 dynamo=True，而它需要 onnxscript（未安装）
            dynamo=False,
            opset_version=spec.opset,
            input_names=[INPUT_NAME],
            output_names=list(OUTPUT_NAMES),
            dynamic_axes=_dynamic_axes(spec.dynamic_hw),
            do_constant_folding=True,
        )

    _force_ir_version(path, spec.ir_version)
    return path, spec


def _force_ir_version(path: Path, ir_version: int) -> dict[str, Any]:
    """把 IR version 压到指定值并存回。

    为什么必须做：``onnx`` 写出的 IR 版本取决于它自己的版本，
    可能高于目标 ``onnxruntime`` 的上限。实测 onnx 1.23 写 IR 14、
    onnxruntime 1.23.2 上限 11，报 ``Unsupported model IR version``。
    版本错配在"换一台机器/换一块板子"时才暴露，属于最难查的一类问题。
    """
    import onnx

    model = onnx.load(str(path))
    original = model.ir_version
    if original > ir_version:
        model.ir_version = ir_version
        onnx.save(model, str(path))
    return {"ir_version_before": original, "ir_version_after": model.ir_version}


def inspect_onnx(path: str | Path) -> dict[str, Any]:
    """读回产物做自检：IR/opset/算子清单/规模。用于验收与报告取证。"""
    import onnx
    from collections import Counter

    model = onnx.load(str(path))
    ops = Counter(node.op_type for node in model.graph.node)
    initializers = sum(int(t.data_type != 0) for t in model.graph.initializer)
    return {
        "path": str(path),
        "ir_version": model.ir_version,
        "opset": [(o.domain or "ai.onnx", o.version) for o in model.opset_import],
        "producer": f"{model.producer_name} {model.producer_version}".strip(),
        "inputs": [i.name for i in model.graph.input],
        "outputs": [o.name for o in model.graph.output],
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "op_types": dict(ops.most_common()),
        "size_bytes": Path(path).stat().st_size,
    }
