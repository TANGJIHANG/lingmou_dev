"""检测评测：IoU 贪心匹配的召回/精度 + 虚警统计。

⚠️ **这不是 mAP**
----------------
本模块做的是"按 IoU 阈值做一对一贪心匹配"，然后算召回与精度。
它**不是** COCO 的 mAP@[.5:.95]，也**不是** pycocotools 口径。
报告里必须写明这一点——把贪心匹配的数字说成 mAP 是**不实陈述**。

为什么现在只做到这个程度：D3–D4 之后紧接着要做的是**两组训练的对照**，
需要的是"同一把尺子量两个模型"，而不是"与文献比绝对分数"。
贪心匹配足以支撑内部对照，且没有额外依赖。正式 mAP 属 D6+ 的评测工作。

为什么评价要能分组
------------------
"虚警"必须在**真背景图**上单独看。把有目标图和背景图混在一起算平均，
会把两组差异里最关键的信号（模型是否把空地当成了目标）稀释掉。
所以 :class:`EvalResult` 把前景图与背景图的预测数分开统计。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from lingmou.engine.dataset import DetectionDataset
from lingmou.engine.infer import predict
from lingmou.io.schema import Box, Sample


def iou_xyxy(a: tuple[float, float, float, float], b: np.ndarray) -> float:
    """两个 ``xyxy`` 框的 IoU。``b`` 允许是长度为 4 的数组。"""
    ix1, iy1 = max(a[0], float(b[0])), max(a[1], float(b[1]))
    ix2, iy2 = min(a[2], float(b[2])), min(a[3], float(b[3]))
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1])) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class EvalResult:
    """一个模型在一个评测集上的结果。"""

    label: str
    checkpoint: str
    score_threshold: float
    iou_threshold: float

    n_images: int = 0
    n_pos_images: int = 0
    n_bg_images: int = 0
    n_gt: int = 0
    #: 有目标图上的预测框数（精度的分母）
    n_pred_pos: int = 0
    #: **真背景图**上的预测框数 —— 这就是虚警
    n_pred_bg: int = 0
    n_matched: int = 0
    per_class_gt: dict[str, int] = field(default_factory=dict)
    per_class_hit: dict[str, int] = field(default_factory=dict)

    @property
    def recall(self) -> float:
        return self.n_matched / self.n_gt if self.n_gt else float("nan")

    @property
    def precision(self) -> float:
        return self.n_matched / self.n_pred_pos if self.n_pred_pos else float("nan")

    @property
    def bg_fp_per_image(self) -> float:
        return self.n_pred_bg / self.n_bg_images if self.n_bg_images else float("nan")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    samples: list[Sample],
    classes: list[str],
    device: torch.device,
    *,
    label: str,
    checkpoint: str = "",
    score_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> EvalResult:
    """在给定样本集上评测一个模型。

    匹配规则：每张图内对标注框与预测框做**贪心一对一**匹配
    （每个标注框找当前 IoU 最大的未占用预测框，IoU ≥ 阈值为命中）。
    同一预测框不会被两个标注框重复计入。
    """
    dataset = DetectionDataset(samples, classes)
    images = [dataset[i][0] for i in range(len(dataset))]
    preds = predict(model, images, device, score_threshold=score_threshold)

    r = EvalResult(
        label=label, checkpoint=checkpoint,
        score_threshold=score_threshold, iou_threshold=iou_threshold,
        n_images=len(samples),
    )

    for sample, pred in zip(samples, preds):
        boxes = pred["boxes"]
        if not sample.has_objects:
            r.n_bg_images += 1
            r.n_pred_bg += len(boxes)
            continue

        r.n_pos_images += 1
        r.n_pred_pos += len(boxes)
        used: set[int] = set()
        for gt in sample.boxes:
            r.n_gt += 1
            r.per_class_gt[gt.label] = r.per_class_gt.get(gt.label, 0) + 1

            best_iou, best_j = iou_threshold, -1
            g = (gt.x1, gt.y1, gt.x2, gt.y2)
            for j, pb in enumerate(boxes):
                if j in used:
                    continue
                v = iou_xyxy(g, pb)
                if v >= best_iou:
                    best_iou, best_j = v, j
            if best_j >= 0:
                used.add(best_j)
                r.n_matched += 1
                r.per_class_hit[gt.label] = r.per_class_hit.get(gt.label, 0) + 1

    return r


def format_results_table(results: list[EvalResult], *, title: str = "") -> str:
    """把若干结果排成对照表。"""
    lines: list[str] = []
    if title:
        lines += [f"### {title}", ""]

    header = (
        "| 模型 | 评测图 | 有目标图 | 背景图 | 标注框 | 命中 | 召回 | 精度 | 背景图虚警 | 虚警/图 |"
    )
    lines += [
        header,
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.label} | {r.n_images} | {r.n_pos_images} | {r.n_bg_images} | {r.n_gt} | "
            f"{r.n_matched} | {r.recall:.3f} | {r.precision:.3f} | {r.n_pred_bg} | "
            f"{r.bg_fp_per_image:.2f} |"
        )

    classes = sorted({c for r in results for c in r.per_class_gt})
    if classes:
        lines += ["", "各类别召回：", "", "| 模型 | " + " | ".join(classes) + " |", "|---" * (len(classes) + 1) + "|"]
        for r in results:
            cells = []
            for c in classes:
                gt = r.per_class_gt.get(c, 0)
                hit = r.per_class_hit.get(c, 0)
                cells.append(f"{hit}/{gt}" if gt else "—")
            lines.append(f"| {r.label} | " + " | ".join(cells) + " |")

    lines += [
        "",
        f"> 匹配口径：IoU ≥ {results[0].iou_threshold if results else 0.5} 的**贪心一对一**匹配，"
        f"置信度阈值 {results[0].score_threshold if results else 0.5}。",
        "> ⚠️ **这不是 mAP**（非 COCO mAP@[.5:.95] 口径），只用于内部对照，不可与文献数字比较。",
    ]
    return "\n".join(lines)
