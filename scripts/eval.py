"""检测评测与两组对照（D3–D4 消融 / D6 对照的基础）。

薄入口：解析参数 -> 调 ``lingmou.engine.evaluate`` -> 打印对照表。

核心用途是**内部对照**：同一把尺子量两个模型。所以本脚本强制把三件事显式化，
因为它们最容易把对照做错：

1. **评测集从哪来**（``--eval-spec``）——两个模型必须在**同一批图**上评。
2. **有没有见过答案**（``--train-spec`` / 权重内记录的 ``dataset_spec``）——
   若 A 的评测图出现在 B 的训练集里，B 就是"背过答案"，比它强不代表模型好。
   脚本会**主动把污染比例打出来**，不需要你记得去查。
3. **干净子集**（``--clean-of``）——排除掉对方训练过的图之后再看一遍。
   污染集上"仍然输"是**保守证据**（偏向对方都赢不了），干净集才是公平对比。

用法::

    # 单模型
    .\\.venv\\Scripts\\python.exe scripts\\eval.py --run "基线=artifacts\\train_baseline_run1\\best.pth"

    # 两组对照（详见 docs/decisions.md 的消融记录）
    .\\.venv\\Scripts\\python.exe scripts\\eval.py `
        --run "A_干净=artifacts\\train_baseline_run1\\best.pth" `
        --run "B_假背景=artifacts\\train_baseline_ablation_fake_bg\\best.pth" `
        --train-spec "A_干净=drop_emptied=true" `
        --train-spec "B_假背景=drop_emptied=false" `
        --eval-spec drop_emptied=true `
        --clean-of drop_emptied=false
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lingmou.engine import build_detector, evaluate as evaluate_model, split_samples  # noqa: E402
from lingmou.engine.evaluate import format_results_table                              # noqa: E402
from lingmou.engine.train import TrainConfig, resolve_device                          # noqa: E402
from lingmou.io import build_dataset, load_config                                     # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

DATASETS_CONFIG = "configs/datasets.yaml"


def parse_spec(text: str) -> dict:
    """把 ``drop_emptied=true,include_negative=true`` 解析成 dict（自动转 bool/int/float）。"""
    out: dict = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"规格项应为 key=value，收到 {item!r}")
        k, v = item.split("=", 1)
        k, v = k.strip(), v.strip()
        low = v.lower()
        if low in ("true", "false"):
            out[k] = low == "true"
        else:
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out


def parse_pairs(items: list[str] | None, what: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"{what} 应为 LABEL=VALUE，收到 {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def make_split(spec_overrides: dict, train_cfg: TrainConfig, datasets_cfg: dict, *, limit: int | None = None):
    """按数据集规格构造，并用训练配置的 val_ratio/seed 切分。"""
    if train_cfg.dataset not in datasets_cfg["datasets"]:
        raise SystemExit(f"数据集 {train_cfg.dataset!r} 不在 {DATASETS_CONFIG} 中")
    spec = dict(datasets_cfg["datasets"][train_cfg.dataset])
    spec.update(spec_overrides)
    samples = build_dataset(train_cfg.dataset, spec, project_root=PROJECT_ROOT)
    train, val = split_samples(samples, val_ratio=train_cfg.val_ratio, seed=train_cfg.seed)
    if limit:
        val = val[:limit]
    return train, val, spec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="检测评测与对照")
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=PATH",
                    help="要评测的权重，可重复")
    ap.add_argument("--train-spec", action="append", metavar="LABEL=SPEC",
                    help="该权重训练时用的数据集规格；权重内没记录时用它做污染核查")
    ap.add_argument("--eval-from", default="configs/train_baseline.yaml",
                    help="提供 val_ratio/seed/类别/检测器的训练配置")
    ap.add_argument("--eval-spec", default=None,
                    help="评测集用哪个数据集规格的 val 划分（默认用 datasets.yaml 的配置）")
    ap.add_argument("--clean-of", default=None,
                    help="额外报告一个排除该规格训练集后的干净子集")
    ap.add_argument("--thresholds", nargs="*", type=float, default=[0.5],
                    help="置信度阈值，可多个（默认只跑 0.5）")
    ap.add_argument("--iou", type=float, default=0.5, help="命中判定 IoU 阈值")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="分批前向的批大小（必须分批：整集一次前向会爆显存）")
    ap.add_argument("--limit", type=int, default=None, help="只评前 N 张（冒烟用，结果无意义）")
    args = ap.parse_args(argv)

    runs = parse_pairs(args.run, "--run")
    train_specs = parse_pairs(args.train_spec, "--train-spec")

    train_cfg = TrainConfig.from_yaml(PROJECT_ROOT / args.eval_from)
    datasets_cfg = load_config(PROJECT_ROOT / DATASETS_CONFIG)
    device = resolve_device(args.device or train_cfg.device)

    eval_over = parse_spec(args.eval_spec) if args.eval_spec else {}
    _, eval_samples, eval_used = make_split(eval_over, train_cfg, datasets_cfg, limit=args.limit)

    print(f"=== 评测集 ===")
    print(f"  来源        : {args.eval_from} 的 val 划分 × 规格 {eval_over or '(datasets.yaml 默认)'}")
    print(f"  实际规格    : drop_emptied={eval_used.get('drop_emptied')} "
          f"include_negative={eval_used.get('include_negative')}")
    print(f"  规模        : {len(eval_samples)} 张"
          f"（有目标 {sum(1 for s in eval_samples if s.has_objects)} / "
          f"真背景 {sum(1 for s in eval_samples if not s.has_objects)}）")
    if args.limit:
        print("  ⚠️ --limit 生效，结果无意义，仅验证链路")

    clean_samples = None
    if args.clean_of:
        clean_over = parse_spec(args.clean_of)
        other_train, _, other_used = make_split(clean_over, train_cfg, datasets_cfg)
        other_ids = {s.image_id for s in other_train}
        eval_ids = {s.image_id for s in eval_samples}
        clean_samples = [s for s in eval_samples if s.image_id not in other_ids]
        print(f"\n=== 干净子集（排除 {args.clean_of} 的训练图）===")
        print(f"  {len(eval_ids)} 张中排除 {len(eval_ids & other_ids)} 张 -> 剩 {len(clean_samples)} 张")

    # ── 污染核查：每个模型见过多少评测图 ──────────────────────
    print(f"\n=== 污染核查（评测图出现在各模型训练集里的比例）===")
    for label in runs:
        ckpt = torch.load(PROJECT_ROOT / runs[label], map_location="cpu", weights_only=False)
        spec = ckpt.get("dataset_spec")
        src = "权重内记录"
        if spec is None:
            if label not in train_specs:
                print(f"  {label:12s}: 权重未记录 dataset_spec 且未给 --train-spec，无法核查")
                continue
            spec = dict(datasets_cfg["datasets"][train_cfg.dataset])
            spec.update(parse_spec(train_specs[label]))
            src = "--train-spec"
        run_train, _, _ = make_split(
            {k: v for k, v in spec.items()
             if k in ("drop_emptied", "include_negative", "classes")},
            train_cfg, datasets_cfg,
        )
        seen = {s.image_id for s in run_train} & {s.image_id for s in eval_samples}
        pct = len(seen) / len(eval_samples) if eval_samples else 0
        flag = "  ← 评测集偏向它" if pct > 0.1 else ""
        print(f"  {label:12s}: {len(seen):4d} / {len(eval_samples)} ({pct:5.1%})  [{src}]{flag}")

    # ── 评测 ─────────────────────────────────────────────────
    for tag, samples in (("主评测集", eval_samples), ("干净子集", clean_samples)):
        if samples is None:
            continue
        print(f"\n{'=' * 78}\n[{tag}] {len(samples)} 张\n{'=' * 78}")
        for thr in args.thresholds:
            results = []
            for label, path in runs.items():
                ckpt = torch.load(PROJECT_ROOT / path, map_location="cpu", weights_only=False)
                c = ckpt["config"]
                model = build_detector(
                    c["detector"], len(c["classes"]), pretrained=False,
                    min_size=c["min_size"], max_size=c["max_size"],
                )
                model.load_state_dict(ckpt["model_state"])
                model.to(device)
                results.append(evaluate_model(
                    model, samples, list(c["classes"]), device,
                    label=label, checkpoint=path,
                    score_threshold=thr, iou_threshold=args.iou,
                    batch_size=args.batch_size,
                ))
            print()
            print(format_results_table(results, title=f"置信度阈值 {thr}"))
            print()

    print("=" * 78)
    print("读法提醒：")
    print("  · 主评测集若对某一方有污染，该方占便宜；它【仍然输】才是硬证据（保守）。")
    print("  · 干净子集才是公平对比，但样本量小，置信区间宽，别读小数点后第二位。")
    print("  · 这些数字不是 mAP，只能内部对照，不可与文献比。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
