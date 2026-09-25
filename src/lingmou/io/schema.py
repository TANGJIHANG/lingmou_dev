"""数据接入层的统一输入协议。

设计意图
--------
命题要求"标准化多源数据输入"。光学、红外、SAR 三类源的**物理含义完全不同**，
但下游（``engine`` / ``tasks``）不应该为每种源写一套分支——否则每加一个源就要改引擎。

因此本模块定义**一条记录的统一形状**（见 ``io/__init__.py`` 的四字段协议）：

===================  ==========================================================
``source_type``      光学 / 红外 / SAR
``timestamp``        成像时刻；源不提供则为 ``None``（**不编造**）
``geo_hint``         地理提示（经纬度、景号等）；源不提供则为 ``None``
``payload``          图像本体：文件路径，或已解码的 numpy 数组
===================  ==========================================================

诚实原则
--------
``timestamp`` / ``geo_hint`` 为 ``None`` 时表示**该数据集确实不提供**，
而不是"随便填一个"。
D2 选用的 NWPU VHR-10 与 SSDD 都不带这两项元数据——
这一点直接影响后续多源融合的演示方式（融合要有时间轴，届时需要
合成时间戳或改用带元数据的源），必须显式记录，不能靠默认值蒙混过关。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


class SourceType(str, enum.Enum):
    """数据源类型。命题要求 ≥2 类数据源统一接入。"""

    OPTICAL = "optical"
    INFRARED = "infrared"
    SAR = "sar"


@dataclass(frozen=True)
class Box:
    """一个水平矩形检测框（xyxy，像素坐标，左上/右下）。

    为什么同时保留 ``polygon``
    --------------------------
    DOTA 系标注给的是**旋转框**（四点）。D2 的竖线只需要水平框，
    但"把旋转框拍平"是不可逆的信息损失——一旦丢了，后续要训旋转框
    就得重新走一遍数据准备。因此这里**保留原始四点**，`x1..y2` 只是它的外接矩形。
    """

    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    difficult: int = 0
    polygon: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self) -> None:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(
                f"退化框（宽或高 <= 0）：{self.label} ({self.x1},{self.y1})-({self.x2},{self.y2})"
            )

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_xywh(self) -> tuple[float, float, float, float]:
        """COCO 口径的 ``(x, y, w, h)``。"""
        return (self.x1, self.y1, self.width, self.height)

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float, label: str, **kw) -> "Box":
        return cls(x1=x, y1=y, x2=x + w, y2=y + h, label=label, **kw)

    @classmethod
    def from_polygon(
        cls, points: Sequence[tuple[float, float]], label: str, difficult: int = 0
    ) -> "Box":
        """由四点旋转框构造：保留原始四点，外接矩形作为 ``x1..y2``。"""
        pts = tuple((float(px), float(py)) for px, py in points)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return cls(
            x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys),
            label=label, difficult=difficult, polygon=pts,
        )

    def clip(self, width: int, height: int) -> "Box | None":
        """裁剪到图像范围内；裁没了返回 ``None``。

        真实数据里偶有越界框（标注时的取整误差）。静默保留会让后续
        预处理报错或产生零面积样本，因此统一在接入层处理掉并计数。
        """
        x1, y1 = max(self.x1, 0.0), max(self.y1, 0.0)
        x2, y2 = min(self.x2, float(width)), min(self.y2, float(height))
        if x2 <= x1 or y2 <= y1:
            return None
        if (x1, y1, x2, y2) == (self.x1, self.y1, self.x2, self.y2):
            return self
        return Box(x1, y1, x2, y2, self.label, self.difficult, self.polygon)


@dataclass(frozen=True, eq=False)
class Sample:
    """统一输入协议的一条记录：**一张图 + 它的框**。

    ``eq=False``：``payload`` 可能是 numpy 数组，逐元素比较既慢又语义不清。
    需要标识时用 ``image_id``。
    """

    image_id: str
    source_type: SourceType
    payload: Path | np.ndarray
    width: int
    height: int
    boxes: tuple[Box, ...] = ()
    timestamp: float | None = None
    geo_hint: str | None = None

    def __post_init__(self) -> None:
        if not self.image_id:
            raise ValueError("image_id 不能为空 —— 它是跨源追踪与去重的唯一标识")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"非法图像尺寸：{self.width}x{self.height}")
        if isinstance(self.payload, np.ndarray):
            if self.payload.ndim != 3:
                raise ValueError(f"payload 数组应为 HWC 三通道，实际 shape={self.payload.shape}")
            if self.payload.shape[0] != self.height or self.payload.shape[1] != self.width:
                raise ValueError(
                    f"payload 尺寸 {self.payload.shape[1]}x{self.payload.shape[0]} "
                    f"与声明的 {self.width}x{self.height} 不一致"
                )

    # ── 属性 ──────────────────────────────────────────────

    @property
    def has_objects(self) -> bool:
        """是否为负样本（无任何目标）。训练时正负样本比例要靠它统计。"""
        return len(self.boxes) > 0

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(b.label for b in self.boxes)

    @property
    def is_file_backed(self) -> bool:
        return isinstance(self.payload, Path)

    # ── 操作 ──────────────────────────────────────────────

    def load(self, flags: int | None = None) -> np.ndarray:
        """取出图像数组。文件路径就地解码，已解码的直接返回。"""
        if isinstance(self.payload, np.ndarray):
            return self.payload
        from lingmou.io.imageio import load_image

        return load_image(self.payload, flags=flags)

    def clipped(self) -> "Sample":
        """返回裁掉越界框后的副本（``boxes`` 顺序保持稳定）。"""
        kept = tuple(b for b in (box.clip(self.width, self.height) for box in self.boxes) if b)
        if len(kept) == len(self.boxes):
            return self
        return Sample(
            image_id=self.image_id,
            source_type=self.source_type,
            payload=self.payload,
            width=self.width,
            height=self.height,
            boxes=kept,
            timestamp=self.timestamp,
            geo_hint=self.geo_hint,
        )

    def filter_labels(self, keep: Iterable[str]) -> "Sample":
        """只保留指定类别；其余框丢弃。

        D2 硬约束"第一版类别数 ≤3 类"就落在这里：NWPU 原版 10 类，
        过滤到 3 类后**全部框都被滤掉的图自动变成负样本**——
        这一点必须显式统计，否则会误判"数据集缩水了"。
        """
        keep_set = set(keep)
        kept = tuple(b for b in self.boxes if b.label in keep_set)
        if len(kept) == len(self.boxes):
            return self
        return Sample(
            image_id=self.image_id,
            source_type=self.source_type,
            payload=self.payload,
            width=self.width,
            height=self.height,
            boxes=kept,
            timestamp=self.timestamp,
            geo_hint=self.geo_hint,
        )

    def __repr__(self) -> str:  # 避免打印整张图
        where = self.payload.name if isinstance(self.payload, Path) else f"<ndarray {self.payload.shape}>"
        return (
            f"<Sample {self.image_id} {self.source_type.value} {self.width}x{self.height} "
            f"boxes={len(self.boxes)} {where}>"
        )
