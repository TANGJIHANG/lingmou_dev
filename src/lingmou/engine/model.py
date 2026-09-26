"""基线检测器构建。

选型：**FCOS + ResNet50-FPN（COCO 预训练）**，理由有三条，都是为后面几步铺路
-----------------------------------------------------------------------------
1. **无锚框（anchor-free）**：后处理只有"解码 + NMS"，没有一堆 anchor 尺寸/长宽比
   需要解释。D6 要求"逐框与 PyTorch 对齐"，后处理越简单，两侧不一致的可能性越小
   ——差异只可能来自模型本身，而不是后处理里的魔法数字。
2. **无 RoIAlign / 两阶段结构**：Faster R-CNN 的 RoIAlign 与 RPN 在 ONNX 导出和
   国产芯片算子支持上都是已知难点。FCOS 是单阶段纯卷积，算子面窄得多。
3. **满足"不引入 Ultralytics"**：torchvision 是 BSD-3，无 AGPL 传染风险。

⚠️ **部署视角的已知风险**（写在这里免得 D5 才发现）
--------------------------------------------------
torchvision 的 FCOS 头用了 ``GroupNorm``。部分推理后端对 GroupNorm 的支持不如
BatchNorm 普遍，昇腾 CANN 上的算子覆盖**必须实测**。若届时不支持，
退路是：① 导出前把 GroupNorm 融合/替换；② 换 RetinaNet（头是 Conv+ReLU）。
这属于"为部署而设计"的必要记录，不是杞人忧天。

torchvision 的 API 坑
---------------------
::

    fcos_resnet50_fpn(weights=COCO_V1, num_classes=3)
    -> ValueError: The parameter 'num_classes' expected value 91 but got 3 instead.

torchvision 不允许在加载 COCO 权重的同时改类别数。正确做法是**先按 COCO 建模型，
再手工替换检测头**（见 :func:`_replace_fcos_head`），这样骨干与 FPN 的预训练权重
都保留下来。
"""

from __future__ import annotations

from typing import Any, Callable

import torch.nn as nn


class UnknownDetectorError(ValueError):
    """请求了未支持的检测器名。"""


def _replace_fcos_head(model: nn.Module, num_classes: int) -> nn.Module:
    """把 COCO 的 80 类检测头换成 ``num_classes`` 类。

    从**现有头**里读 ``in_channels`` / ``num_anchors``，而不是把 256 / 1 写死：
    将来换骨干（如 ResNet50 -> MobileNet）时这里不用改。
    """
    from torchvision.models.detection.fcos import FCOSClassificationHead, FCOSRegressionHead

    old_cls = model.head.classification_head
    in_channels = old_cls.conv[0].in_channels
    num_anchors = old_cls.num_anchors

    model.head.classification_head = FCOSClassificationHead(in_channels, num_anchors, num_classes)
    model.head.regression_head = FCOSRegressionHead(in_channels, num_anchors)
    return model


def _build_fcos(num_classes: int, *, pretrained: bool, **kw: Any) -> nn.Module:
    from torchvision.models.detection import FCOS_ResNet50_FPN_Weights, fcos_resnet50_fpn

    if pretrained:
        model = fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.COCO_V1)
        return _replace_fcos_head(model, num_classes)
    return fcos_resnet50_fpn(weights=None, weights_backbone=None, num_classes=num_classes)


def _build_retinanet(num_classes: int, *, pretrained: bool, **kw: Any) -> nn.Module:
    from torchvision.models import ResNet50_Weights
    from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights, retinanet_resnet50_fpn_v2
    from torchvision.models.detection.retinanet import RetinaNetClassificationHead

    if pretrained:
        model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1)
        old = model.head.classification_head
        model.head.classification_head = RetinaNetClassificationHead(
            old.conv[0].in_channels, old.num_anchors, num_classes
        )
        return model
    return retinanet_resnet50_fpn_v2(
        weights=None, weights_backbone=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None,
        num_classes=num_classes,
    )


def _build_fasterrcnn(num_classes: int, *, pretrained: bool, **kw: Any) -> nn.Module:
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_V2_Weights,
        fasterrcnn_resnet50_fpn_v2,
    )

    if pretrained:
        model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model
    return fasterrcnn_resnet50_fpn_v2(
        weights=None, weights_backbone=None, num_classes=num_classes
    )


#: 检测器名 -> 构造函数。新增模型只加一行，不改调用方。
_BUILDERS: dict[str, Callable[..., nn.Module]] = {
    "fcos": _build_fcos,
    "retinanet": _build_retinanet,
    "fasterrcnn": _build_fasterrcnn,
}


def build_detector(
    name: str,
    num_classes: int,
    *,
    pretrained: bool = True,
    min_size: int = 600,
    max_size: int = 800,
    **kwargs: Any,
) -> nn.Module:
    """构建检测器。

    :param name: ``fcos`` / ``retinanet`` / ``fasterrcnn``
    :param num_classes: **前景类别数**（不含背景）。例如 3 类传 3。
        本函数负责转成 torchvision 要的"含背景"口径（见下）。
    :param pretrained: 是否加载 COCO 预训练权重。
        仅 230 张正样本时，从零训练基本不可能收敛，必须用迁移学习。
    :param min_size: 输入短边目标尺寸。torchvision 默认 800/1333 对本项目的
        小数据集偏重；调小可显著提速，但**会改变特征尺度**，一旦定下就不要中途改，
        否则前后两次实验的数字不可比。该值同时决定 D5 导出时的输入规格。
    :param max_size: 输入长边上限

    ⚠️ ``num_classes`` 的口径陷阱（本项目已实际踩到，CUDA 报
    ``index out of bounds / device-side assert``）
    ------------------------------------------------------------------
    torchvision 检测模型需要的是**含背景**的类别数，标签从 1 开始（0 是背景）。
    但它的两处 docstring **自相矛盾**：

    - ``fcos_resnet50_fpn``：*"number of output classes of the model
      (including the background)"* —— 含背景
    - ``FCOSClassificationHead``：*"number of classes to be predicted"* —— 不提背景

    而 FCOS 的 loss 实现把**标签值直接当作分类头最后一维的下标**::

        gt_classes_targets = torch.zeros_like(cls_logits)      # 最后一维 = num_classes
        gt_classes_targets[foreground_mask, gt_classes[foreground_mask]] = 1.0

    所以传 3（标签为 1/2/3）时，`gt_classes_targets[..., 3]` 在长度为 3 的维度上越界。
    本函数统一在内部 ``+1``，调用方只需传前景类别数——避免每个调用点各错一次。
    """
    if name not in _BUILDERS:
        raise UnknownDetectorError(f"未知检测器 {name!r}；可选：{sorted(_BUILDERS)}")
    if num_classes < 1:
        raise ValueError(f"num_classes（前景类别数）必须 >= 1，实际 {num_classes}")

    # 前景 -> 含背景。标签仍从 1 开始，0 留给背景。
    n_classes_with_bg = num_classes + 1
    model = _BUILDERS[name](n_classes_with_bg, pretrained=pretrained, **kwargs)

    # 输入尺寸写在模型自带的 transform 上，导出 ONNX 时以它为准
    model.transform.min_size = (min_size,)
    model.transform.max_size = max_size
    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """返回 ``(总参数量, 可训练参数量)``，进评测报告的"模型规模"一节。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
