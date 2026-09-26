"""检测数据集：把 io 层的 :class:`~lingmou.io.schema.Sample` 接进 torchvision 检测模型。

这一层只做"格式转换"，不做任何清洗——清洗与类别过滤已经在 ``lingmou.io`` 完成。
职责边界要守住：否则同一个过滤规则会在两处各写一遍，迟早不一致。

torchvision 检测模型的输入约定（踩过才知道的几条）
--------------------------------------------------
1. 图像是 **0–1 的 float 张量**，``[C,H,W]``。模型内部才做 mean/std 归一化，
   **不要**在 Dataset 里提前归一化，否则等于归一化两次。
2. ``targets`` 的 ``labels`` **从 1 开始，0 是背景**。类别名到序号的映射必须唯一且稳定，
   否则评测报告里的"第 1 类"到底是谁都说不清。
3. ``targets`` 的 ``boxes`` 是 ``xyxy``、绝对像素坐标。
4. 批量不能用默认 collate：每张图尺寸不同，必须用 :func:`detection_collate`
   返回 ``(list[Tensor], list[dict])``。

关于训练/验证划分
------------------
NWPU VHR-10 **没有官方划分**。本模块用一个固定种子的随机划分，
并把这个事实写在这里：**我们的 mAP 不能和文献里的数字直接比**，
因为划分不同、训练集大小不同。引用时只能作内部对照（例如压缩前后）。
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from lingmou.io.schema import Sample


class DetectionDataset(Dataset):
    """把 ``Sample`` 列表包装成 torchvision 检测模型可消费的数据集。

    :param samples: 已由 io 层过滤好类别的样本
    :param classes: 类别名有序列表；下标 0 对应标签 **1**（0 保留给背景）
    """

    def __init__(self, samples: list[Sample], classes: list[str]) -> None:
        if not classes:
            raise ValueError("classes 不能为空")
        self.samples = list(samples)
        self.classes = list(classes)
        # 标签从 1 开始：0 是背景，这是 torchvision 检测模型的硬约定
        self.class_to_idx = {name: i + 1 for i, name in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict]:
        sample = self.samples[index]

        # cv2 给的是 BGR；模型是在 RGB 上预训练的，通道顺序错了精度会莫名其妙地差
        bgr = sample.load()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0

        boxes, labels = [], []
        for box in sample.boxes:
            idx = self.class_to_idx.get(box.label)
            if idx is None:
                # io 层已按类别过滤过，走到这里说明配置与数据集不一致，必须报错而不是静默丢
                raise KeyError(
                    f"样本 {sample.image_id} 含未知类别 {box.label!r}；"
                    f"当前 classes={self.classes}"
                )
            boxes.append([box.x1, box.y1, box.x2, box.y2])
            labels.append(idx)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
        }
        return image, target

    def label_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in self.classes}
        for s in self.samples:
            for label in s.labels:
                if label in counts:
                    counts[label] += 1
        return counts


def detection_collate(batch):
    """检测任务的 collate：图尺寸不一，只能打包成两个平行列表。"""
    images, targets = zip(*batch)
    return list(images), list(targets)


def split_samples(
    samples: list[Sample],
    *,
    val_ratio: float = 0.2,
    seed: int = 20260925,
) -> tuple[list[Sample], list[Sample]]:
    """按固定种子划分训练/验证集。

    用固定种子而不是 ``random.shuffle``：否则每次跑出来的验证集都不同，
    `docs/decisions.md` 要求的"消融实验可对照"就无从谈起。

    划分前**按 image_id 排序**，保证输入顺序不同也得到同样的划分
    （样本从不同数据源汇总而来时，顺序可能不稳定）。
    """
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio 必须在 [0,1)，实际 {val_ratio}")

    ordered = sorted(samples, key=lambda s: s.image_id)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ordered))

    n_val = int(round(len(ordered) * val_ratio))
    val_idx = set(perm[:n_val].tolist())
    train = [s for i, s in enumerate(ordered) if i not in val_idx]
    val = [s for i, s in enumerate(ordered) if i in val_idx]
    return train, val
