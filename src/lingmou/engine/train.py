"""基线检测器训练循环（D3–D4）。

设计要点
--------
1. **基线就是基线**：FP32、不做任何压缩与加速。roadmap 说得直白——
   "没有 FP32 基线，任何'优化后'的数字都无意义"。所以这里刻意不引入 AMP、
   不做通道剪枝、不换轻量骨干；那些属于 D8+ 的消融实验。
2. **每个 epoch 都记 loss 分量**。检测的总 loss 是分类/回归/centerness 的加权和，
   只看总数无法判断"不收敛"到底卡在哪一路。历史写入 JSON，便于事后画图与对照。
3. **梯度裁剪**。检测任务在训练早期 loss 容易炸（尤其小数据集 + 预训练头被替换后），
   裁剪到 10 是 torchvision 参考实现的做法。
4. **可截断**（``max_train_iters`` / ``max_val_iters``）。搭脚本阶段要能在几十秒内
   验证"链路是否通"，而不是每次都等完整训练跑完。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lingmou.engine.dataset import DetectionDataset, detection_collate


@dataclass
class TrainConfig:
    """训练配置。字段与 ``configs/train_baseline.yaml`` 一一对应。"""

    classes: list[str]
    #: 用 ``configs/datasets.yaml`` 里的哪个数据集
    dataset: str = "nwpu_vhr10"
    detector: str = "fcos"
    pretrained: bool = True

    epochs: int = 20
    batch_size: int = 4
    lr: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 5e-4
    lr_milestones: list[int] = field(default_factory=lambda: [12, 18])
    lr_gamma: float = 0.1
    warmup_iters: int = 200
    clip_grad_norm: float = 10.0

    min_size: int = 600
    max_size: int = 800
    val_ratio: float = 0.2
    seed: int = 20260925
    num_workers: int = 0

    device: str = "auto"          # auto / cuda / cpu
    output_dir: str = "artifacts/train_baseline"
    # 冒烟用：限制每轮迭代数；None 表示跑满
    max_train_iters: int | None = None
    max_val_iters: int | None = None

    def __post_init__(self) -> None:
        if not self.classes:
            raise ValueError("classes 不能为空")
        if self.epochs < 1:
            raise ValueError(f"epochs 必须 >= 1，实际 {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size 必须 >= 1，实际 {self.batch_size}")

    @classmethod
    def from_dict(cls, data: dict) -> "TrainConfig":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            # 静默忽略拼错的键 = 你以为改了参数其实没改，是最难查的一类 bug
            raise ValueError(f"训练配置里有未知字段：{sorted(unknown)}；可用字段：{sorted(known)}")
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        # 允许把配置放在 train: 段下，便于以后一个文件同时描述数据集与训练
        return cls.from_dict(data.get("train", data))


class History:
    """逐 epoch 的 loss 记录。"""

    def __init__(self) -> None:
        self.epochs: list[dict[str, Any]] = []

    def add(self, epoch: int, train: dict[str, float], val: dict[str, float], lr: float, secs: float) -> None:
        self.epochs.append(
            {"epoch": epoch, "lr": lr, "seconds": round(secs, 2),
             "train": train, "val": val}
        )

    def to_dict(self) -> dict:
        return {"epochs": self.epochs}

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return p


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_loader(
    dataset: DetectionDataset, config: TrainConfig, *, shuffle: bool
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=detection_collate,
        # 数据集很小、且图片从磁盘现读，worker 多了反而抢 IO
        pin_memory=torch.cuda.is_available(),
    )


def _warmup_factor(it: int, warmup_iters: int, warmup_factor: float = 1.0 / 1000.0) -> float:
    """线性 warmup 系数。前若干次迭代用小 lr，避免一上来就把预训练特征打坏。"""
    if warmup_iters <= 0 or it >= warmup_iters:
        return 1.0
    alpha = it / warmup_iters
    return warmup_factor * (1.0 - alpha) + alpha


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    *,
    epoch: int,
    warmup_iters: int = 0,
    clip_grad_norm: float = 10.0,
    max_iters: int | None = None,
    print_freq: int = 20,
) -> dict[str, float]:
    """训练一轮，返回各 loss 分量的均值。"""
    model.train()
    totals: dict[str, float] = {}
    count = 0
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    t0 = time.perf_counter()

    for it, (images, targets) in enumerate(loader):
        if max_iters is not None and it >= max_iters:
            break

        # warmup 只在第 0 轮做（torchvision 参考实现的做法）
        if epoch == 0 and warmup_iters > 0:
            factor = _warmup_factor(it, warmup_iters)
            for group, base in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base * factor

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        if clip_grad_norm and clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

        count += 1
        for k, v in loss_dict.items():
            totals[k] = totals.get(k, 0.0) + float(v.detach())
        totals["total"] = totals.get("total", 0.0) + float(losses.detach())

        if print_freq and (it + 1) % print_freq == 0:
            avg = totals["total"] / count
            print(f"    epoch {epoch} it {it + 1}/{len(loader)}  loss {avg:.4f}")

    # 恢复 base lr，交给 scheduler 在 epoch 末统一调度
    for group, base in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base

    if count == 0:
        return {}
    means = {k: v / count for k, v in totals.items()}
    means["_seconds"] = time.perf_counter() - t0
    return means


@torch.no_grad()
def evaluate_loss(
    model: nn.Module, loader: DataLoader, device: torch.device, *, max_iters: int | None = None
) -> dict[str, float]:
    """在验证集上算 loss。

    注意：**这不是 mAP**。D3–D4 的验收物是"收敛的 loss 曲线"，
    真正的精度评测（mAP、虚警率）属于 D6+ 的 ``scripts/eval.py``。
    这里只用来判断"有没有过拟合"。
    """
    model.train()   # 检测模型只有在 train 模式才返回 loss（torchvision 的设计）
    totals: dict[str, float] = {}
    count = 0
    for it, (images, targets) in enumerate(loader):
        if max_iters is not None and it >= max_iters:
            break
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        count += 1
        for k, v in loss_dict.items():
            totals[k] = totals.get(k, 0.0) + float(v.detach())
        totals["total"] = totals.get("total", 0.0) + float(losses.detach())

    if count == 0:
        return {}
    return {k: v / count for k, v in totals.items()}


def build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("模型没有可训练参数——是否误冻结了整个网络？")
    return torch.optim.SGD(
        params, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: TrainConfig
) -> torch.optim.lr_scheduler.MultiStepLR:
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(config.lr_milestones), gamma=config.lr_gamma
    )


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    config: TrainConfig,
    *,
    device: torch.device | None = None,
    print_freq: int = 20,
) -> History:
    """完整训练流程，返回 loss 历史。"""
    device = device or resolve_device(config.device)
    model.to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    history = History()

    for epoch in range(config.epochs):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(
            model, optimizer, train_loader, device,
            epoch=epoch, warmup_iters=config.warmup_iters,
            clip_grad_norm=config.clip_grad_norm,
            max_iters=config.max_train_iters, print_freq=print_freq,
        )
        val_loss = (
            evaluate_loss(model, val_loader, device, max_iters=config.max_val_iters)
            if val_loader is not None else {}
        )
        lr = optimizer.param_groups[0]["lr"]
        history.add(epoch, train_loss, val_loss, lr, time.perf_counter() - t0)
        scheduler.step()

        if train_loss:
            msg = f"  [epoch {epoch}] train total={train_loss.get('total', float('nan')):.4f}"
            if val_loss:
                msg += f"  val total={val_loss.get('total', float('nan')):.4f}"
            msg += f"  lr={lr:.5f}  {time.perf_counter() - t0:.1f}s"
            print(msg)

    return history


def save_checkpoint(
    model: nn.Module, path: str | Path, *, config: TrainConfig, epoch: int, history: History
) -> Path:
    """保存权重。**权重不进 Git**（见 .gitignore 的 ``*.pth``），登记在 artifacts/README.md。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(config),
            "epoch": epoch,
            "history": history.to_dict(),
        },
        p,
    )
    return p


def plot_loss_curve(
    history: History, path: str | Path, *, title: str = "Baseline training loss"
) -> Path:
    """画 loss 曲线。D3–D4 的验收物之一。

    坐标轴标签用英文：matplotlib 默认字体不含中文字形，
    直接写中文会变成一排方框——这种图放进答辩材料很难看。
    """
    import matplotlib

    matplotlib.use("Agg")   # 无显示环境必须显式指定，否则在某些环境下会挂
    import matplotlib.pyplot as plt

    epochs = [e["epoch"] for e in history.epochs]
    if not epochs:
        raise ValueError("history 为空，无法画图")

    fig, ax = plt.subplots(figsize=(8, 5))
    train_total = [e["train"].get("total") for e in history.epochs]
    ax.plot(epochs, train_total, marker="o", label="train total")

    if any(e["val"] for e in history.epochs):
        val_total = [e["val"].get("total") if e["val"] else None for e in history.epochs]
        ax.plot(epochs, val_total, marker="s", label="val total")

    # 各 loss 分量单独画，才能看出"不收敛卡在哪一路"
    components = sorted({k for e in history.epochs for k in e["train"] if not k.startswith("_") and k != "total"})
    for name in components:
        ax.plot(epochs, [e["train"].get(name) for e in history.epochs], linestyle="--", alpha=0.6, label=name)

    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p
