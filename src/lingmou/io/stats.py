"""数据集统计。

为什么单独一层
--------------
本项目的开发纪律写着"**所有指标都要有口径**"。数据集规模、类别分布、
正负样本比这些数字会反复出现在答辩 PPT、评测报告和训练日志里，
口径必须唯一。所以统计只在本模块算一次，别处在需要时调用它，
而不是各处 ``len()`` 现算——那迟早会出现"PPT 上一个数、报告里另一个数"。
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from lingmou.io.schema import Sample


def label_counts(samples: Iterable[Sample]) -> dict[str, int]:
    """各类别的框数，按数量降序。"""
    counter: Counter[str] = Counter()
    for s in samples:
        counter.update(s.labels)
    return dict(counter.most_common())


def summarize(samples: list[Sample]) -> dict:
    """汇总一批样本的关键统计量。

    ``boxes_per_image`` 只在**有目标的图**上统计：把负样本算进平均值
    会把"每图目标数"稀释成一个没有物理意义的数。
    """
    if not samples:
        return {
            "images": 0, "images_with_objects": 0, "images_without_objects": 0,
            "boxes": 0, "boxes_per_positive_image_mean": 0.0, "boxes_per_image_max": 0,
            "labels": {}, "source_types": {}, "size_min": None, "size_max": None,
        }

    positives = [s for s in samples if s.has_objects]
    n_boxes = sum(len(s.boxes) for s in samples)
    sizes = [(s.width, s.height) for s in samples]

    return {
        "images": len(samples),
        "images_with_objects": len(positives),
        "images_without_objects": len(samples) - len(positives),
        "boxes": n_boxes,
        "boxes_per_positive_image_mean": (n_boxes / len(positives)) if positives else 0.0,
        "boxes_per_image_max": max(len(s.boxes) for s in samples),
        "labels": label_counts(samples),
        "source_types": dict(Counter(s.source_type.value for s in samples).most_common()),
        "size_min": (min(w for w, _ in sizes), min(h for _, h in sizes)),
        "size_max": (max(w for w, _ in sizes), max(h for _, h in sizes)),
    }


def by_source(samples: Iterable[Sample]) -> dict[str, list[Sample]]:
    """按数据源分组，键为 ``source_type`` 值（``optical`` / ``infrared`` / ``sar``）。"""
    grouped: dict[str, list[Sample]] = {}
    for s in samples:
        grouped.setdefault(s.source_type.value, []).append(s)
    return grouped


def summary_markdown(title: str, summary: dict) -> str:
    """把汇总渲染成 Markdown 表格，供 ``DATA_SOURCES.md`` / 评测报告直接引用。"""
    lines = [
        f"### {title}",
        "",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| 图像总数 | {summary['images']} |",
        f"| 含目标图像 | {summary['images_with_objects']} |",
        f"| 纯背景图像 | {summary['images_without_objects']} |",
        f"| 标注框总数 | {summary['boxes']} |",
        f"| 平均每张含目标图的框数 | {summary['boxes_per_positive_image_mean']:.2f} |",
        f"| 单图最大框数 | {summary['boxes_per_image_max']} |",
        f"| 数据源 | {', '.join(f'{k}×{v}' for k, v in summary['source_types'].items()) or '—'} |",
        f"| 图像尺寸范围 | {summary['size_min']} – {summary['size_max']} |",
        "",
        "| 类别 | 框数 |",
        "|---|---|",
    ]
    for label, n in summary["labels"].items():
        lines.append(f"| {label} | {n} |")
    if not summary["labels"]:
        lines.append("| （无） | 0 |")
    return "\n".join(lines)
