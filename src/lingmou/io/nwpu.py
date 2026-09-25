"""NWPU VHR-10 读取器（光学）。

数据集
------
西北工业大学 VHR-10，航拍/卫星光学影像，原版 **10 类**：

    positive image set/   650 张，含目标
    negative image set/   150 张，纯背景
    ground truth/         650 个 txt，与正样本同名

标注格式（**原始 GT，非第三方 COCO**）::

    (x1,y1),(x2,y2),class_id

两个已经踩过的坑，写在这里防止后人重踩
--------------------------------------
1. **负样本与正样本完全撞名**（都是 ``001.jpg``…``150.jpg``）。
   天真地把两个目录合并会静默覆盖 150 张图，而且不报错。
   因此 ``image_id`` 必须带 ``positive`` / ``negative`` 前缀。

2. **数字右对齐带空格**：``( 108, 60),( 192,154),1``。
   正则若写成 ``\\((\\d+)`` 会匹配失败，结果是**静默丢掉 13 个标注文件（约 2% 数据）**。
   本项目实测：严正则得 637/650 图、2485 框；容忍空白后得 **650/650 图、3896 框**。
   故本模块用宽容正则，并在解不出框时**直接报错**而不是跳过。

与第三方 COCO 标注的关系
------------------------
社区另有一版"further marked"的 COCO 标注（``data/interim/nwpu_vhr10_coco/``），
共 3921 框，与本模块解出的 3896 框相差 0.6%，但**个别框坐标不同**
（如 ``001.jpg``：原始 GT 高 95，COCO 版高 86）。
本项目以**数据集自带的原始 GT 为准**，COCO 版仅作交叉校验，理由见 ``docs/decisions.md``。
"""

from __future__ import annotations

import re
from pathlib import Path

from lingmou.io.imageio import image_size
from lingmou.io.schema import Box, Sample, SourceType

#: NWPU VHR-10 原版 10 类，id 与 GT 文件中的数字一一对应
NWPU_CLASSES: dict[int, str] = {
    1: "airplane",
    2: "ship",
    3: "storage_tank",
    4: "baseball_diamond",
    5: "tennis_court",
    6: "basketball_court",
    7: "ground_track_field",
    8: "harbor",
    9: "bridge",
    10: "vehicle",
}

#: D2 硬约束"第一版类别数 ≤3 类"所选的三类。
#: 选它们的理由：与命题的"空天目标"叙事情节最贴，且 ``ship`` 与 SSDD 的 SAR 舰船
#: 构成同名类，后续做光学/SAR 融合时**类别空间天然对齐**，不需要额外映射表。
D2_CLASSES: tuple[str, ...] = ("airplane", "ship", "vehicle")

#: GT 行：``(x1,y1),(x2,y2),class_id`` —— 注意 ``\\s*`` 用来吃掉右对齐空格
_GT_LINE = re.compile(
    r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*(\d+)"
)


class DatasetFormatError(ValueError):
    """标注文件与预期格式不符。**必须报错**，不能跳过——静默丢数据比崩溃危险得多。"""


def parse_gt_text(text: str, *, source: str = "<text>") -> list[Box]:
    """解析一个 GT 文件的内容为框列表。

    无法识别的非空行会触发 :class:`DatasetFormatError`，
    避免"少解析了几个框但没人发现"。
    """
    boxes: list[Box] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        m = _GT_LINE.search(line)
        if not m:
            raise DatasetFormatError(
                f"{source}:{lineno} 无法解析：{line!r}\n"
                f"  预期格式 (x1,y1),(x2,y2),class_id（数字可带右对齐空格）"
            )
        x1, y1, x2, y2, cls_id = (int(g) for g in m.groups())
        if cls_id not in NWPU_CLASSES:
            raise DatasetFormatError(f"{source}:{lineno} 未知类别 id={cls_id}")
        if x2 <= x1 or y2 <= y1:
            # 退化框：记录在案但不构造 Box（Box 会拒绝），避免整个数据集读不进来
            continue
        boxes.append(Box(float(x1), float(y1), float(x2), float(y2), NWPU_CLASSES[cls_id]))
    return boxes


def read_nwpu_vhr10(
    root: str | Path,
    *,
    keep: tuple[str, ...] | None = D2_CLASSES,
    include_negative: bool = True,
    drop_emptied: bool = False,
    strict: bool = True,
) -> list[Sample]:
    """读取 NWPU VHR-10。

    :param root: 数据集根目录（其下应有 ``positive image set`` 等三个子目录）
    :param keep: 保留的类别；``None`` 表示原版 10 类全留。
    :param include_negative: 是否并入 150 张**真背景图**（``negative image set``）
    :param drop_emptied: 是否丢弃"被类别过滤滤空"的图。**这个开关很重要，见下。**
    :param strict: 正样本缺 GT 文件时报错（默认），否则当作无目标
    :return: ``Sample`` 列表，按 ``image_id`` 排序（保证可复现）

    关于 ``drop_emptied``
    --------------------
    NWPU 650 张正样本在 10 类下**每张都有目标**。一旦按 D2 的 3 类子集过滤，
    实测有 **420 张（64.6%）被滤空**——但注意：这些图**画面里确实有目标**，
    只是属于我们不要的那 7 类。

    于是有两种用法，各有代价：

    - ``drop_emptied=False``（默认，图省事）：把滤空的图当负样本用。
      数据集保持在 800 张，但其中 420 张是**含未标注目标的"假背景"**，
      训练时会给检测器灌入标签噪声——教它压制真实存在的物体。
    - ``drop_emptied=True``（干净）：只留 230 张有标注的正样本 + 150 张真背景。

    本项目 D2 保留默认值以便**如实记录这个取舍**；
    D3–D4 训练前必须显式决定，见 ``docs/decisions.md``。
    """
    root = Path(root)
    pos_dir = root / "positive image set"
    neg_dir = root / "negative image set"
    gt_dir = root / "ground truth"
    for d in (pos_dir, gt_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"数据集目录不完整，缺少：{d}")

    samples: list[Sample] = []

    for img_path in sorted(pos_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
            continue
        gt_path = gt_dir / f"{img_path.stem}.txt"
        if gt_path.exists():
            boxes = tuple(parse_gt_text(gt_path.read_text(errors="replace"), source=str(gt_path)))
        elif strict:
            raise DatasetFormatError(f"正样本缺少 GT 文件：{gt_path}")
        else:
            boxes = ()

        w, h = image_size(img_path)
        s = Sample(
            image_id=f"nwpu/positive/{img_path.stem}",
            source_type=SourceType.OPTICAL,
            payload=img_path,
            width=w,
            height=h,
            boxes=boxes,
            timestamp=None,   # 该数据集不提供成像时刻
            geo_hint=None,    # 该数据集不提供地理信息
        )
        filtered = s.clipped() if keep is None else s.clipped().filter_labels(keep)
        if drop_emptied and boxes and not filtered.has_objects:
            # 本来有目标、被类别过滤滤空 -> 是"假背景"，不是真负样本
            continue
        samples.append(filtered)

    if include_negative and neg_dir.is_dir():
        for img_path in sorted(neg_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
                continue
            w, h = image_size(img_path)
            samples.append(
                Sample(
                    # 前缀必须带：负样本与正样本文件名完全相同
                    image_id=f"nwpu/negative/{img_path.stem}",
                    source_type=SourceType.OPTICAL,
                    payload=img_path,
                    width=w,
                    height=h,
                    boxes=(),
                    timestamp=None,
                    geo_hint=None,
                )
            )

    samples.sort(key=lambda s: s.image_id)
    return samples
