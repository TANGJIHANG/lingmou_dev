"""engine 层（检测引擎）单元测试。

重点守两类东西
--------------
1. **``num_classes`` 口径**——torchvision 检测模型要"含背景"的类别数，标签从 1 开始。
   本项目实际踩过：传 3 而标签为 1/2/3，CUDA 报
   ``index out of bounds`` + ``device-side assert``，且报错位置（GIoU loss）
   离真正的原因（分类头最后一维）很远，很难查。故此处用**真实前向**钉死。
2. **划分的可复现性**——验证集一变，历史实验数字就不可比。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from lingmou.engine import (
    DetectionDataset,
    TrainConfig,
    build_detector,
    count_parameters,
    detection_collate,
    plot_loss_curve,
    predictions_to_boxes,
    split_samples,
)
from lingmou.engine.model import UnknownDetectorError
from lingmou.engine.train import History
from lingmou.io.imageio import save_image
from lingmou.io.schema import Box, Sample, SourceType

CLASSES = ["airplane", "ship", "vehicle"]


def _sample(tmp_path: Path, name: str, labels: list[str], *, size: int = 64) -> Sample:
    path = save_image(tmp_path / f"{name}.png", np.zeros((size, size, 3), dtype=np.uint8))
    boxes = tuple(
        Box(4, 4, 20, 20, label) if i % 2 == 0 else Box(24, 24, 40, 40, label)
        for i, label in enumerate(labels)
    )
    return Sample(
        image_id=name,
        source_type=SourceType.OPTICAL,
        payload=path,
        width=size,
        height=size,
        boxes=boxes,
    )


@pytest.fixture(scope="module")
def tiny_detector():
    """一个小输入尺寸的未预训练 FCOS，供前向测试复用（构建较慢，模块级缓存）。"""
    return build_detector("fcos", len(CLASSES), pretrained=False, min_size=64, max_size=128)


# ── num_classes 口径（核心回归） ──────────────────────────

def test_detector_head_includes_background_channel(tiny_detector) -> None:
    """3 个前景类 -> 分类头必须有 4 个通道（含背景），否则标签 3 会越界。"""
    head = tiny_detector.head.classification_head
    assert head.cls_logits.out_channels // head.num_anchors == len(CLASSES) + 1


def test_detector_forward_accepts_labels_starting_at_one(tiny_detector) -> None:
    """标签 1..N 必须能直接训练。

    修复前这条会触发 CUDA ``device-side assert``：
    ``gt_classes_targets`` 最后一维是 num_classes，而标签值被当作下标使用。
    """
    tiny_detector.train()
    images = [torch.rand(3, 128, 128)]
    targets = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0], [60.0, 60.0, 100.0, 110.0]]),
            "labels": torch.tensor([len(CLASSES), 1]),   # 用最大标签值探边界
        }
    ]
    losses = tiny_detector(images, targets)
    assert {"classification", "bbox_regression", "bbox_ctrness"} <= set(losses)
    for name, value in losses.items():
        assert torch.isfinite(value), f"{name} 不是有限值：{value}"


def test_detector_forward_accepts_empty_targets(tiny_detector) -> None:
    """纯背景图（无任何框）必须能训练——D2 特意保留了 150 张真背景图。"""
    tiny_detector.train()
    targets = [
        {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64)}
    ]
    losses = tiny_detector([torch.rand(3, 128, 128)], targets)
    assert torch.isfinite(losses["classification"])


def test_build_detector_rejects_unknown_name() -> None:
    with pytest.raises(UnknownDetectorError, match="未知检测器"):
        build_detector("yolov99", 3, pretrained=False)


def test_build_detector_rejects_zero_classes() -> None:
    with pytest.raises(ValueError, match="必须 >= 1"):
        build_detector("fcos", 0, pretrained=False)


def test_count_parameters_positive(tiny_detector) -> None:
    total, trainable = count_parameters(tiny_detector)
    assert total > 0
    assert 0 < trainable <= total


# ── 数据集 ────────────────────────────────────────────────

def test_label_mapping_starts_at_one(tmp_path: Path) -> None:
    """标签从 1 开始，0 留给背景 —— torchvision 检测模型的硬约定。"""
    ds = DetectionDataset([_sample(tmp_path, "a", ["airplane", "ship"])], CLASSES)
    assert ds.class_to_idx == {"airplane": 1, "ship": 2, "vehicle": 3}
    _, target = ds[0]
    assert target["labels"].tolist() == [1, 2]
    assert target["boxes"].shape == (2, 4)


def test_dataset_image_is_float_zero_to_one(tmp_path: Path) -> None:
    """模型内部才做 mean/std 归一化，Dataset 必须给 0–1 的 float。"""
    ds = DetectionDataset([_sample(tmp_path, "a", ["ship"])], CLASSES)
    image, _ = ds[0]
    assert image.dtype == torch.float32
    assert image.shape[0] == 3
    assert 0.0 <= float(image.min()) and float(image.max()) <= 1.0


def test_dataset_rejects_label_outside_classes(tmp_path: Path) -> None:
    """io 层已过滤过类别；出现未知类别说明两处配置不一致，必须报错而不是静默丢框。"""
    s = _sample(tmp_path, "a", ["ship"])
    polluted = Sample(
        s.image_id, s.source_type, s.payload, s.width, s.height,
        boxes=(Box(1, 1, 9, 9, "harbor"),),
    )
    ds = DetectionDataset([polluted], CLASSES)
    with pytest.raises(KeyError, match="未知类别"):
        _ = ds[0]


def test_dataset_handles_background_image(tmp_path: Path) -> None:
    ds = DetectionDataset([_sample(tmp_path, "bg", [])], CLASSES)
    _, target = ds[0]
    assert target["boxes"].shape == (0, 4)
    assert target["labels"].shape == (0,)


def test_detection_collate_returns_parallel_lists(tmp_path: Path) -> None:
    ds = DetectionDataset(
        [_sample(tmp_path, "a", ["ship"]), _sample(tmp_path, "b", ["airplane", "vehicle"])],
        CLASSES,
    )
    images, targets = detection_collate([ds[0], ds[1]])
    assert len(images) == len(targets) == 2


# ── 划分 ──────────────────────────────────────────────────

def test_split_is_deterministic_and_order_independent(tmp_path: Path) -> None:
    """同种子、同集合 -> 同划分；输入顺序不同也必须得到同一划分。"""
    samples = [_sample(tmp_path, f"s{i:02d}", ["ship"]) for i in range(20)]
    a_train, a_val = split_samples(samples, val_ratio=0.25, seed=7)
    b_train, b_val = split_samples(list(reversed(samples)), val_ratio=0.25, seed=7)
    assert [s.image_id for s in a_val] == [s.image_id for s in b_val]
    assert [s.image_id for s in a_train] == [s.image_id for s in b_train]
    assert len(a_val) == 5


def test_split_different_seeds_differ(tmp_path: Path) -> None:
    samples = [_sample(tmp_path, f"s{i:02d}", ["ship"]) for i in range(20)]
    _, v1 = split_samples(samples, val_ratio=0.25, seed=1)
    _, v2 = split_samples(samples, val_ratio=0.25, seed=2)
    assert [s.image_id for s in v1] != [s.image_id for s in v2]


def test_split_rejects_bad_ratio(tmp_path: Path) -> None:
    samples = [_sample(tmp_path, "s0", ["ship"])]
    with pytest.raises(ValueError, match="val_ratio"):
        split_samples(samples, val_ratio=1.0)


# ── 配置 ──────────────────────────────────────────────────

def test_train_config_rejects_unknown_field() -> None:
    """拼错的键必须报错：静默忽略 = 你以为改了参数其实没改。"""
    with pytest.raises(ValueError, match="未知字段"):
        TrainConfig.from_dict({"classes": ["ship"], "epoch": 10})


def test_train_config_rejects_empty_classes() -> None:
    with pytest.raises(ValueError, match="classes"):
        TrainConfig(classes=[])


def test_train_config_from_yaml(tmp_path: Path) -> None:
    import yaml

    p = tmp_path / "t.yaml"
    p.write_text(
        yaml.safe_dump({"train": {"classes": ["ship"], "epochs": 3, "detector": "fcos"}}),
        encoding="utf-8",
    )
    cfg = TrainConfig.from_yaml(p)
    assert cfg.classes == ["ship"]
    assert cfg.epochs == 3
    assert cfg.dataset == "nwpu_vhr10"     # 未指定的取默认值


# ── 推理后处理 ────────────────────────────────────────────

def test_predictions_to_boxes_maps_label_and_score() -> None:
    pred = {
        "boxes": np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32),
        "scores": np.array([0.87], dtype=np.float32),
        "labels": np.array([2], dtype=np.int64),          # 2 -> classes[1] == "ship"
    }
    boxes = predictions_to_boxes(pred, CLASSES)
    assert len(boxes) == 1
    assert boxes[0].label == "ship 0.87"


def test_predictions_to_boxes_skips_degenerate() -> None:
    """检测模型在阈值边缘会吐出零面积框；Box 会拒绝，必须先过滤。"""
    pred = {
        "boxes": np.array([[5.0, 5.0, 5.0, 5.0], [0.0, 0.0, 8.0, 8.0]], dtype=np.float32),
        "scores": np.array([0.9, 0.8], dtype=np.float32),
        "labels": np.array([1, 1], dtype=np.int64),
    }
    boxes = predictions_to_boxes(pred, CLASSES)
    assert len(boxes) == 1


# ── 画图 ──────────────────────────────────────────────────

def test_plot_loss_curve_writes_file(tmp_path: Path) -> None:
    history = History()
    history.add(0, {"total": 3.0, "classification": 2.0, "_seconds": 1.0}, {"total": 2.5}, 0.005, 1.0)
    history.add(1, {"total": 2.0, "classification": 1.2, "_seconds": 1.0}, {"total": 1.8}, 0.005, 1.0)
    out = plot_loss_curve(history, tmp_path / "curve.png")
    assert out.is_file() and out.stat().st_size > 0


def test_plot_loss_curve_rejects_empty_history(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="history 为空"):
        plot_loss_curve(History(), tmp_path / "curve.png")


def test_history_save_roundtrip(tmp_path: Path) -> None:
    import json

    history = History()
    history.add(0, {"total": 1.5, "_seconds": 2.0}, {}, 0.01, 2.0)
    p = history.save(tmp_path / "h.json")
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["epochs"][0]["train"]["total"] == 1.5
