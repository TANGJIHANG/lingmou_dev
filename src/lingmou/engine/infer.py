"""推理与结果可视化（D3–D4 的"带框可视化图"）。

一个必须知道的坐标约定
----------------------
torchvision 检测模型内部会把图像 resize 到 ``min_size``/``max_size``，
但它的 ``transform.postprocess`` **会把预测框映射回原图尺寸**。
所以 ``model(images)`` 在 eval 模式下返回的 ``boxes`` 已经是**原图像素坐标**，
可以直接和标注框叠在一起画——不需要我们再乘一次缩放系数。
（若在这里再手动缩放一次，框会整体偏移放大，是很典型的错法。）

与 ``io.visualize`` 的分工
--------------------------
``io.visualize`` 画的是**数据集标注**（验证数据接入是否正确）；
本模块画的是**模型预测 vs 标注**（验证模型是否学到了东西）。
两者都要有：前者错说明数据管道有问题，后者错才是模型的问题。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from lingmou.io.imageio import save_image
from lingmou.io.schema import Box, Sample
from lingmou.io.visualize import letterbox
from lingmou.engine.dataset import DetectionDataset

#: 标注框用细线、预测框用粗线，一张图里能分清谁是谁
GT_THICKNESS = 1
PRED_THICKNESS = 2


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    images: list[torch.Tensor],
    device: torch.device,
    *,
    score_threshold: float = 0.5,
) -> list[dict[str, np.ndarray]]:
    """对一批图像推理，返回 numpy 形式的预测。

    ``model.eval()`` 是必须的：检测模型在 train 模式下返回 loss 而不是预测，
    且 BN 会走训练分支。忘掉这一句的典型症状是"推理输出是个 dict"。
    """
    was_training = model.training
    model.eval()
    outputs = model([img.to(device) for img in images])

    results = []
    for out in outputs:
        keep = out["scores"] >= score_threshold
        results.append(
            {
                "boxes": out["boxes"][keep].detach().cpu().numpy(),
                "scores": out["scores"][keep].detach().cpu().numpy(),
                "labels": out["labels"][keep].detach().cpu().numpy(),
            }
        )
    if was_training:
        model.train()
    return results


def predictions_to_boxes(
    prediction: dict[str, np.ndarray], classes: list[str]
) -> list[Box]:
    """把模型输出转成 :class:`~lingmou.io.schema.Box`，标签带上置信度。

    ``Box`` 会拒绝退化框（宽或高 ≤ 0），所以这里必须先过滤——
    检测模型在阈值边缘确实会吐出零面积框，直接构造会抛异常。
    """
    boxes: list[Box] = []
    for (x1, y1, x2, y2), score, label in zip(
        prediction["boxes"], prediction["scores"], prediction["labels"]
    ):
        if x2 <= x1 or y2 <= y1:
            continue
        idx = int(label) - 1          # 模型标签从 1 开始，0 是背景
        name = classes[idx] if 0 <= idx < len(classes) else f"cls{label}"
        boxes.append(
            Box(float(x1), float(y1), float(x2), float(y2), f"{name} {score:.2f}")
        )
    return boxes


def draw_gt_and_predictions(
    image: np.ndarray,
    gt_boxes: tuple[Box, ...],
    pred_boxes: list[Box],
) -> np.ndarray:
    """在同图上画标注（绿细线）与预测（红粗线）。"""
    canvas = image.copy()
    for b in gt_boxes:
        cv2.rectangle(
            canvas, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)),
            (0, 200, 0), GT_THICKNESS,
        )
    for b in pred_boxes:
        cv2.rectangle(
            canvas, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)),
            (0, 0, 255), PRED_THICKNESS,
        )
    return canvas


def save_prediction_grid(
    samples: list[Sample],
    model: torch.nn.Module,
    classes: list[str],
    path: str | Path,
    *,
    device: torch.device,
    columns: int = 2,
    cell: int = 480,
    score_threshold: float = 0.5,
) -> Path:
    """把若干验证样本的"标注 + 预测"拼成一张网格图。

    绿色细线 = 标注，红色粗线 = 预测。一眼就能看出漏检、误检、框偏。
    """
    if not samples:
        raise ValueError("没有样本可画")

    dataset = DetectionDataset(samples, classes)
    images, _ = zip(*(dataset[i] for i in range(len(samples))))
    preds = predict(model, list(images), device, score_threshold=score_threshold)

    rows = (len(samples) + columns - 1) // columns
    bar = 26
    grid = np.full((rows * (cell + bar), columns * cell, 3), 32, dtype=np.uint8)

    for i, (sample, pred) in enumerate(zip(samples, preds)):
        r, c = divmod(i, columns)
        img = sample.load()
        tile, scale, dx, dy = letterbox(img, cell)

        for b in sample.boxes:
            cv2.rectangle(
                tile, (int(b.x1 * scale) + dx, int(b.y1 * scale) + dy),
                (int(b.x2 * scale) + dx, int(b.y2 * scale) + dy),
                (0, 200, 0), GT_THICKNESS,
            )
        pred_boxes = predictions_to_boxes(pred, classes)
        for b in pred_boxes:
            cv2.rectangle(
                tile, (int(b.x1 * scale) + dx, int(b.y1 * scale) + dy),
                (int(b.x2 * scale) + dx, int(b.y2 * scale) + dy),
                (0, 0, 255), PRED_THICKNESS,
            )

        y0 = r * (cell + bar)
        grid[y0:y0 + cell, c * cell:(c + 1) * cell] = tile
        info = (
            f"{sample.image_id}  gt={len(sample.boxes)}  pred={len(pred_boxes)}"
            f"  thr={score_threshold:g}"
        )
        cv2.putText(
            grid, info, (c * cell + 6, y0 + cell + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA,
        )
    return save_image(path, grid)
