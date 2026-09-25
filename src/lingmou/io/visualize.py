"""检测结果可视化：画框、拼图。

用途
----
D2 的验收物之一是"数据能读能可视化"。这里的图不是给人看着好看的，
而是**验收证据**：框画错位置、类别串了、坐标口径搞反（xywh/xyxy），
肉眼一看就露馅。所以可视化是数据准备阶段最便宜的校验手段。

颜色稳定性
----------
按类别分配颜色时用的是 ``zlib.crc32`` 而不是内置 ``hash()``：
后者对字符串**逐进程随机化**（PYTHONHASHSEED），会导致同一张图
两次运行颜色不同，做前后对比图时非常碍事。
"""

from __future__ import annotations

import zlib
from pathlib import Path

import cv2
import numpy as np

from lingmou.io.imageio import save_image
from lingmou.io.schema import Box, Sample

#: BGR 调色板（OpenCV 顺序）
PALETTE: tuple[tuple[int, int, int], ...] = (
    (0, 0, 255),      # 红
    (0, 200, 0),      # 绿
    (255, 128, 0),    # 蓝
    (0, 200, 255),    # 黄
    (255, 0, 200),    # 紫
    (200, 200, 0),    # 青
    (0, 128, 255),    # 橙
    (128, 0, 255),    # 品红
)

def label_color(label: str) -> tuple[int, int, int]:
    """按类别名稳定地取一个颜色。"""
    idx = zlib.crc32(label.encode("utf-8")) % len(PALETTE)
    return PALETTE[idx]


def draw_boxes(
    image: np.ndarray,
    boxes: tuple[Box, ...] | list[Box],
    *,
    thickness: int = 2,
    show_label: bool = True,
) -> np.ndarray:
    """在图像副本上画框（不修改传入数组）。

    本项目的类别名都是 ASCII（``airplane`` / ``ship`` / …），
    可直接用 ``cv2.putText`` 的 Hershey 字体，无需中文字体依赖。
    """
    canvas = image.copy()
    for b in boxes:
        color = label_color(b.label)
        p1 = (int(round(b.x1)), int(round(b.y1)))
        p2 = (int(round(b.x2)), int(round(b.y2)))
        cv2.rectangle(canvas, p1, p2, color, thickness)

        # 旋转框：把原始四点也画出来，便于核对"拍平"有没有跑偏
        if b.polygon:
            pts = np.array([[int(round(x)), int(round(y))] for x, y in b.polygon], dtype=np.int32)
            cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)

        if show_label:
            text = b.label if not b.difficult else f"{b.label}*"
            (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = max(p1[1] - 4, th + 4)
            cv2.rectangle(canvas, (p1[0], ty - th - 2), (p1[0] + tw + 4, ty + base), color, -1)
            cv2.putText(
                canvas, text, (p1[0] + 2, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )
    return canvas


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """等比缩放到 ``size x size`` 并居中留边，返回缩放系数与偏移。"""
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 32, dtype=np.uint8)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas, scale, dx, dy


def make_grid(
    samples: list[Sample],
    *,
    columns: int = 2,
    cell: int = 480,
    max_items: int = 4,
) -> np.ndarray:
    """把若干样本画成网格图（每格下方留一条信息栏）。

    坐标变换与图像缩放共用同一个 ``scale`` / ``dx`` / ``dy``，
    因此**框如果画歪了，一定意味着坐标口径有问题**——这正是要看图的原因。
    """
    chosen = samples[:max_items]
    if not chosen:
        raise ValueError("没有样本可画")

    rows = (len(chosen) + columns - 1) // columns
    bar = 26
    grid = np.full((rows * (cell + bar), columns * cell, 3), 32, dtype=np.uint8)

    for i, s in enumerate(chosen):
        r, c = divmod(i, columns)
        img = s.load()
        tile, scale, dx, dy = letterbox(img, cell)
        for b in s.boxes:
            p1 = (int(round(b.x1 * scale)) + dx, int(round(b.y1 * scale)) + dy)
            p2 = (int(round(b.x2 * scale)) + dx, int(round(b.y2 * scale)) + dy)
            cv2.rectangle(tile, p1, p2, label_color(b.label), 2)
        y0 = r * (cell + bar)
        grid[y0:y0 + cell, c * cell:(c + 1) * cell] = tile

        info = f"{s.image_id}  {s.width}x{s.height}  {s.source_type.value}  n={len(s.boxes)}"
        cv2.putText(
            grid, info, (c * cell + 6, y0 + cell + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA,
        )
    return grid


def save_grid(path: str | Path, samples: list[Sample], **kwargs) -> Path:
    return save_image(path, make_grid(samples, **kwargs))
