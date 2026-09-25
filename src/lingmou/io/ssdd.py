"""SSDD 读取器（SAR）。

数据集
------
SAR Ship Detection Dataset，SAR 影像舰船检测，合成自 RadarSat-2 / TerraSAR-X / Sentinel-1，
分辨率 1–15 m，含 HH/VV/VH/HV 多种极化，近岸与远海场景兼有。**单类：ship**。

目录结构（``data/interim/ssdd/data/SSDD_DOTA/``）::

    JPEGImages/                 1160 张，所有划分共用
    Annotations_train/           928
    Annotations_test/            232
    Annotations_test_inshore/     46   ← test 的子集
    Annotations_test_offshore/   186   ← test 的子集

标注格式为 **DOTA 八点旋转框**::

    x1 y1 x2 y2 x3 y3 x4 y4 ship 0
    （另有可选的 imagesource: / gsd: 头两行）

两个必须写在明面上的事
----------------------
1. **inshore/offshore 不是独立划分**，而是 test 按场景筛出的子集
   （46 + 186 = 232 = test 全量）。把它们当额外训练数据 = 标签泄漏。
   本模块因此把它们标成 ``split="test_inshore"`` 这类"分析用视图"，
   并在 :data:`ANALYSIS_ONLY_SPLITS` 里列出。

2. **旋转框被拍平成外接矩形**。D2 竖线只做水平框，但原始四点已存进
   ``Box.polygon``，后续要训旋转框不必重做数据准备。
   注意拍平后**相邻舰船的外接矩形可能重叠**，这会抬高 NMS 的压制率——
   答辩若被问"为什么舰船漏检"，这是一个必须先排除的已知因素。
"""

from __future__ import annotations

from pathlib import Path

from lingmou.io.imageio import image_size
from lingmou.io.schema import Box, Sample, SourceType

#: SSDD 只有一类
SSDD_CLASSES: tuple[str, ...] = ("ship",)

#: 划分名 -> 标注目录名
_SPLIT_DIRS: dict[str, str] = {
    "train": "Annotations_train",
    "test": "Annotations_test",
    "test_inshore": "Annotations_test_inshore",
    "test_offshore": "Annotations_test_offshore",
}

#: 仅用于**分场景分析**的视图，不是独立划分；拿它们训练会与 test 标签泄漏
ANALYSIS_ONLY_SPLITS: frozenset[str] = frozenset({"test_inshore", "test_offshore"})


class DatasetFormatError(ValueError):
    """标注文件与预期格式不符。"""


def parse_dota_label(text: str, *, source: str = "<text>") -> list[Box]:
    """解析一个 DOTA 标注文件为框列表。

    容忍 ``imagesource:`` / ``gsd:`` 头行与空行；其余无法解析的行报错。
    """
    boxes: list[Box] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("imagesource:", "gsd:")):
            continue  # DOTA 头信息，本项目暂不消费（影像来源/地面采样距离）

        parts = stripped.split()
        if len(parts) < 10:
            raise DatasetFormatError(
                f"{source}:{lineno} 字段数不足（期望 >=10，实际 {len(parts)}）：{stripped!r}"
            )
        try:
            coords = [float(v) for v in parts[:8]]
        except ValueError as exc:
            raise DatasetFormatError(f"{source}:{lineno} 坐标不是数字：{stripped!r}") from exc

        label = parts[8]
        difficult = int(parts[9]) if parts[9].lstrip("-").isdigit() else 0
        points = [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]
        b = Box.from_polygon(points, label=label, difficult=difficult)
        # from_polygon 已保证 x2>x1 且 y2>y1，无需再判退化
        boxes.append(b)
    return boxes


def read_ssdd_dota(
    root: str | Path,
    *,
    split: str = "train",
    keep: tuple[str, ...] | None = SSDD_CLASSES,
) -> list[Sample]:
    """读取 SSDD 的一个划分。

    :param root: SSDD 根目录（其下应有 ``JPEGImages`` 与各 ``Annotations_*``）
    :param split: ``train`` / ``test`` / ``test_inshore`` / ``test_offshore``
    :param keep: 保留类别；``None`` 表示全留
    :return: ``Sample`` 列表，按 ``image_id`` 排序
    """
    root = Path(root)
    if split not in _SPLIT_DIRS:
        raise ValueError(f"未知划分 {split!r}，可选：{sorted(_SPLIT_DIRS)}")

    img_dir = root / "JPEGImages"
    ann_dir = root / _SPLIT_DIRS[split]
    for d in (img_dir, ann_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"数据集目录不完整，缺少：{d}")

    samples: list[Sample] = []
    for ann_path in sorted(ann_dir.iterdir()):
        if ann_path.suffix.lower() != ".txt":
            continue
        img_path = img_dir / f"{ann_path.stem}.jpg"
        if not img_path.exists():
            raise FileNotFoundError(f"标注找不到对应图像：{ann_path.name} -> {img_path}")

        boxes = tuple(parse_dota_label(ann_path.read_text(errors="replace"), source=str(ann_path)))
        w, h = image_size(img_path)
        s = Sample(
            image_id=f"ssdd/{split}/{ann_path.stem}",
            source_type=SourceType.SAR,
            payload=img_path,
            width=w,
            height=h,
            boxes=boxes,
            timestamp=None,   # SSDD 不提供成像时刻
            geo_hint=None,    # SSDD 不提供经纬度（原始产品里有，但发布版未附带）
        )
        samples.append(s.clipped() if keep is None else s.clipped().filter_labels(keep))

    samples.sort(key=lambda s: s.image_id)
    return samples
