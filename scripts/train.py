"""D3–D4 基线检测器训练入口。

薄入口：解析参数 -> 调 ``lingmou.engine`` -> 落盘。
训练逻辑全在 ``src/lingmou/engine/`` 里，可被测试覆盖。

用法::

    # 冒烟：几十秒验证链路是否通（1 轮、各 3 次迭代）
    .\\.venv\\Scripts\\python.exe scripts\\train.py --smoke

    # 正式基线训练
    .\\.venv\\Scripts\\python.exe scripts\\train.py

    # 临时少跑几轮（不修改配置文件）
    .\\.venv\\Scripts\\python.exe scripts\\train.py --epochs 5

产物（默认 ``artifacts/train_baseline/``）：**均不进 Git**，登记见 ``artifacts/README.md``
- ``best.pth``        最终权重（含 config 与 loss 历史）
- ``history.json``    逐 epoch 的 loss 分量，供复查与画图
- ``loss_curve.png``  D3–D4 验收物之一
- ``predictions.png`` 标注 vs 预测对照图，D3–D4 验收物之一
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Windows 终端默认 GBK，直接 print 中文会乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

from lingmou.engine import (  # noqa: E402
    DetectionDataset,
    TrainConfig,
    build_detector,
    count_parameters,
    fit,
    make_loader,
    plot_loss_curve,
    resolve_device,
    save_checkpoint,
    save_prediction_grid,
    split_samples,
)
from lingmou.io import build_dataset, load_config, summarize  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="D3–D4 基线检测器训练")
    ap.add_argument("--config", default="configs/train_baseline.yaml", help="训练配置")
    ap.add_argument("--datasets", default="configs/datasets.yaml", help="数据集配置")
    ap.add_argument("--epochs", type=int, help="覆盖配置里的 epochs")
    ap.add_argument("--device", help="覆盖配置里的 device（cuda / cpu / auto）")
    ap.add_argument("--batch-size", type=int, help="覆盖配置里的 batch_size")
    ap.add_argument(
        "--smoke", action="store_true",
        help="冒烟模式：1 轮、各 3 次迭代，几十秒内验证链路",
    )
    ap.add_argument("--grid-items", type=int, default=4, help="预测对照图画几张")
    return ap.parse_args(argv)


def build_samples(config: TrainConfig, datasets_cfg: dict):
    """按训练配置取出数据集，并校验类别定义两端一致。"""
    if config.dataset not in datasets_cfg["datasets"]:
        raise SystemExit(
            f"数据集 {config.dataset!r} 不在 datasets.yaml 中；"
            f"可选：{sorted(datasets_cfg['datasets'])}"
        )
    spec = datasets_cfg["datasets"][config.dataset]
    declared = list(spec.get("classes") or [])

    # 顺序也重要：类别顺序决定标签序号（从 1 开始），顺序不同则"第 1 类"不是同一个类，
    # 权重、评测、报告会全部对不上。
    if declared != list(config.classes):
        raise SystemExit(
            f"类别不一致：\n"
            f"  datasets.yaml[{config.dataset}].classes = {declared}\n"
            f"  train.yaml.classes                        = {list(config.classes)}\n"
            f"两处必须完全一致（含顺序）——顺序决定标签序号。"
        )
    return build_dataset(config.dataset, spec, project_root=PROJECT_ROOT)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    config = TrainConfig.from_yaml(PROJECT_ROOT / args.config)
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
    if args.device is not None:
        config = replace(config, device=args.device)
    if args.batch_size is not None:
        config = replace(config, batch_size=args.batch_size)
    if args.smoke:
        config = replace(
            config, epochs=1, max_train_iters=3, max_val_iters=2,
            output_dir=config.output_dir.rstrip("/\\") + "_smoke",
        )

    datasets_cfg = load_config(PROJECT_ROOT / args.datasets)

    print(f"=== 训练配置 {args.config} ===")
    print(f"  检测器    : {config.detector}（pretrained={config.pretrained}）")
    print(f"  数据集    : {config.dataset}")
    print(f"  类别      : {config.classes}")
    print(f"  输入尺寸  : min_size={config.min_size} max_size={config.max_size}")
    print(f"  epochs    : {config.epochs}  batch={config.batch_size}  lr={config.lr}")
    if args.smoke:
        print("  ⚠️ 冒烟模式：只跑 1 轮 × 3 次迭代，结果无意义，仅验证链路")

    samples = build_samples(config, datasets_cfg)
    train_samples, val_samples = split_samples(
        samples, val_ratio=config.val_ratio, seed=config.seed
    )

    print("\n=== 数据 ===")
    for name, part in (("train", train_samples), ("val", val_samples)):
        st = summarize(part)
        print(
            f"  {name:5s}: 图={st['images']:4d}  有目标={st['images_with_objects']:4d}"
            f"  纯背景={st['images_without_objects']:4d}  框={st['boxes']:5d}  {st['labels']}"
        )

    train_ds = DetectionDataset(train_samples, config.classes)
    val_ds = DetectionDataset(val_samples, config.classes)
    train_loader = make_loader(train_ds, config, shuffle=True)
    val_loader = make_loader(val_ds, config, shuffle=False)

    device = resolve_device(config.device)
    print(f"\n=== 设备 ===\n  {device}"
          + (f"  {torch_device_name()}" if device.type == "cuda" else ""))

    model = build_detector(
        config.detector,
        len(config.classes),
        pretrained=config.pretrained,
        min_size=config.min_size,
        max_size=config.max_size,
    )
    total, trainable = count_parameters(model)
    print(f"\n=== 模型 ===\n  {config.detector}: 参数 {total/1e6:.1f} M"
          f"（可训练 {trainable/1e6:.1f} M）")

    print("\n=== 开始训练 ===")
    history = fit(model, train_loader, val_loader, config, device=device)

    out_dir = PROJECT_ROOT / config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = save_checkpoint(
        model, out_dir / "best.pth", config=config, epoch=config.epochs - 1, history=history
    )
    hist_path = history.save(out_dir / "history.json")
    print(f"\n权重     -> {ckpt.relative_to(PROJECT_ROOT)}")
    print(f"loss 历史 -> {hist_path.relative_to(PROJECT_ROOT)}")

    try:
        curve = plot_loss_curve(history, out_dir / "loss_curve.png",
                                title=f"{config.detector} baseline loss")
        print(f"loss 曲线 -> {curve.relative_to(PROJECT_ROOT)}")
    except ValueError as exc:
        print(f"跳过画图：{exc}")

    # 预测对照图：挑验证集里标注最多的几张，漏检/误检最容易看出来
    picks = sorted(
        [s for s in val_samples if s.has_objects], key=lambda s: (-len(s.boxes), s.image_id)
    )[: args.grid_items]
    if picks:
        grid = save_prediction_grid(
            picks, model, config.classes, out_dir / "predictions.png", device=device
        )
        print(f"预测对照图 -> {grid.relative_to(PROJECT_ROOT)}"
              f"（绿=标注 细线，红=预测 粗线）")
    else:
        print("验证集没有含目标的样本，跳过预测对照图")

    return 0


def torch_device_name() -> str:
    import torch

    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""


if __name__ == "__main__":
    raise SystemExit(main())
