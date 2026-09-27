"""评测模块测试。

重点
----
1. **IoU 与匹配**：这是所有召回/精度数字的地基，算错一处全盘皆错。
2. **"这不是 mAP" 的口径警告必须出现在输出里** —— 报告里把贪心匹配说成 mAP
   属于不实陈述，所以警告不能靠人记得加，得由格式化函数自己带上。
3. **有目标图与真背景图必须分开统计** —— 虚警信号藏在背景图的预测数里，
   混在一起算平均就把它稀释掉了。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from lingmou.engine.evaluate import EvalResult, evaluate, format_results_table, iou_xyxy
from lingmou.engine.model import build_detector
from lingmou.io.imageio import save_image
from lingmou.io.schema import Box, Sample, SourceType

CLASSES = ["airplane", "ship", "vehicle"]


# ── IoU ───────────────────────────────────────────────────

def test_iou_identical_boxes() -> None:
    assert iou_xyxy((0, 0, 10, 10), np.array([0, 0, 10, 10], dtype=np.float32)) == pytest.approx(1.0)


def test_iou_disjoint_boxes() -> None:
    assert iou_xyxy((0, 0, 10, 10), np.array([20, 20, 30, 30], dtype=np.float32)) == 0.0


def test_iou_half_overlap() -> None:
    # 交集 5x10=50，并集 100+100-50=150 -> 1/3
    assert iou_xyxy((0, 0, 10, 10), np.array([5, 0, 15, 10], dtype=np.float32)) == pytest.approx(50 / 150)


def test_iou_degenerate_box_is_zero() -> None:
    """零面积框不能除零，也不能返回 nan —— 检测模型在阈值边缘会吐这种框。"""
    v = iou_xyxy((5, 5, 5, 5), np.array([5, 5, 5, 5], dtype=np.float32))
    assert v == 0.0 and not np.isnan(v)


# ── EvalResult 派生指标 ────────────────────────────────────

def test_eval_result_metrics() -> None:
    r = EvalResult(label="m", checkpoint="", score_threshold=0.5, iou_threshold=0.5,
                   n_gt=100, n_pred_pos=80, n_matched=70, n_pred_bg=6, n_bg_images=30,
                   n_bg_images_with_fp=3)
    assert r.recall == pytest.approx(0.7)
    assert r.precision == pytest.approx(70 / 80)
    assert r.bg_fp_per_image == pytest.approx(0.2)
    assert r.bg_fp_image_rate == pytest.approx(0.1)


def test_bg_fp_image_rate_differs_from_box_rate() -> None:
    """框数与图数必须分开看：7 个框集中在一张图，和分散在 7 张图，含义完全不同。"""
    concentrated = EvalResult(label="c", checkpoint="", score_threshold=0.5, iou_threshold=0.5,
                              n_pred_bg=7, n_bg_images=10, n_bg_images_with_fp=1)
    spread = EvalResult(label="s", checkpoint="", score_threshold=0.5, iou_threshold=0.5,
                        n_pred_bg=7, n_bg_images=10, n_bg_images_with_fp=7)
    assert concentrated.bg_fp_per_image == spread.bg_fp_per_image == pytest.approx(0.7)
    assert concentrated.bg_fp_image_rate == pytest.approx(0.1)
    assert spread.bg_fp_image_rate == pytest.approx(0.7)


def test_eval_result_metrics_are_nan_not_crash_when_empty() -> None:
    """评测集里没有目标图时不能崩，也不能给出 0（0 会被误读成"全漏"）。"""
    r = EvalResult(label="m", checkpoint="", score_threshold=0.5, iou_threshold=0.5)
    assert np.isnan(r.recall)
    assert np.isnan(r.precision)
    assert np.isnan(r.bg_fp_per_image)


# ── 输出格式 ───────────────────────────────────────────────

def test_table_carries_not_map_warning() -> None:
    """口径警告必须由格式化函数自带，不能靠人记得加。"""
    r = EvalResult(label="A", checkpoint="a.pth", score_threshold=0.5, iou_threshold=0.5,
                   n_images=10, n_pos_images=8, n_bg_images=2, n_gt=40, n_pred_pos=35, n_matched=30)
    text = format_results_table([r], title="对照")
    assert "不是 mAP" in text
    assert "贪心一对一" in text
    assert "A" in text


def test_table_lists_per_class_recall() -> None:
    r = EvalResult(label="A", checkpoint="", score_threshold=0.5, iou_threshold=0.5,
                   per_class_gt={"airplane": 10, "ship": 4}, per_class_hit={"airplane": 9, "ship": 1})
    text = format_results_table([r])
    assert "airplane" in text and "ship" in text
    assert "9/10" in text and "1/4" in text


# ── 端到端（小模型，只验计数与不崩） ───────────────────────

@pytest.fixture(scope="module")
def tiny_detector():
    return build_detector("fcos", len(CLASSES), pretrained=False, min_size=64, max_size=128)


def _samples(tmp_path: Path) -> list[Sample]:
    img = save_image(tmp_path / "x.png", np.zeros((64, 64, 3), dtype=np.uint8))

    def mk(name: str, boxes: tuple[Box, ...]) -> Sample:
        return Sample(name, SourceType.OPTICAL, img, 64, 64, boxes=boxes)

    return [
        mk("pos1", (Box(4, 4, 24, 24, "airplane"),)),
        mk("pos2", (Box(4, 4, 24, 24, "ship"), Box(30, 30, 50, 50, "vehicle"))),
        mk("bg1", ()),
        mk("bg2", ()),
    ]


def test_evaluate_counts_pos_and_bg_separately(tmp_path: Path, tiny_detector) -> None:
    """有目标图与真背景图必须分开计数 —— 虚警信号在背景图里。"""
    r = evaluate(tiny_detector, _samples(tmp_path), CLASSES, torch.device("cpu"),
                 label="t", score_threshold=0.99)
    assert r.n_images == 4
    assert r.n_pos_images == 2
    assert r.n_bg_images == 2
    assert r.n_gt == 3
    assert r.per_class_gt == {"airplane": 1, "ship": 1, "vehicle": 1}


def test_evaluate_high_threshold_yields_no_boxes(tmp_path: Path, tiny_detector) -> None:
    """阈值拉到 0.99 时未训练模型不该出框；命中数与背景虚警都应为 0。"""
    r = evaluate(tiny_detector, _samples(tmp_path), CLASSES, torch.device("cpu"),
                 label="t", score_threshold=0.99)
    assert r.n_matched == 0
    assert r.n_pred_bg == 0
    assert r.n_pred_pos == 0
