"""D5 导出 ONNX 入口。

薄入口：解析参数 -> 调 ``lingmou.engine.export`` -> 落盘 + 自检。

用法::

    # 从基线权重导出（默认读 configs/train_baseline.yaml 的检测器与类别）
    .\\.venv\\Scripts\\python.exe scripts\\export_onnx.py

    # 指定权重与输出
    .\\.venv\\Scripts\\python.exe scripts\\export_onnx.py `
        --checkpoint artifacts\\train_baseline_run1\\best.pth `
        --output artifacts\\lingmou_fcos.onnx

    # 关掉动态 H/W（导出定尺寸图，便于某些后端/工具链）
    .\\.venv\\Scripts\\python.exe scripts\\export_onnx.py --fixed-hw 600x800

产物：
- ``<output>``           ONNX 图
- ``<output>.spec.json`` 导出规格（opset / IR / 动态轴 / 输入尺寸）——**D6 必须用同一份**

验收（D5）：
1. 文件存在，且 Netron 能打开看到计算图  ← 需要你（或我给出图结构摘要）确认
2. IR version 已压低、onnxruntime 能加载
3. 与 PyTorch **原始输出**在多组尺寸上数值一致（这是 D6 逐框对齐的前置）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lingmou.engine import build_detector                      # noqa: E402
from lingmou.engine.export import (                            # noqa: E402
    DetectorRawOutput,
    ExportSpec,
    export_onnx,
    inspect_onnx,
)
from lingmou.engine.train import TrainConfig                   # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="D5 导出 ONNX")
    ap.add_argument("--config", default="configs/train_baseline.yaml", help="训练配置（取检测器/类别/尺寸）")
    ap.add_argument("--checkpoint", default="artifacts/train_baseline_run1/best.pth")
    ap.add_argument("--output", default="artifacts/model.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--ir-version", type=int, default=8)
    ap.add_argument(
        "--fixed-hw", metavar="HxW",
        help="导出定尺寸图（如 600x800）；默认导出动态 H/W",
    )
    ap.add_argument(
        "--verify-sizes", nargs="*",
        # 默认覆盖 3 个数量级：典型检测尺寸 -> 训练 min_size 附近 -> 架构下限附近。
        # 之所以把 32x32 也放进来：docs/decisions.md 里记着"32x32 ~ 600x800 全区间一致"
        # 这个结论，而结论必须能用本仓库的一条命令复现 —— 不能只活在某人的临时脚本里。
        default=["600x800", "448x640", "192x256", "128x128", "64x64", "32x32"],
        help="用这些尺寸做 PyTorch vs ORT 的数值一致性检查（默认跨 3 个数量级）",
    )
    return ap.parse_args(argv)


def _parse_hw(text: str) -> tuple[int, int]:
    try:
        h, w = text.lower().split("x")
        return int(h), int(w)
    except Exception as exc:
        raise SystemExit(f"尺寸格式应为 HxW，例如 600x800；收到 {text!r}") from exc


#: 数值比对用的探针输入种子。
#:
#: 为什么必须固定：首版用裸 ``torch.rand()`` 生成探针，**每次运行输入都不同**，
#: 于是"最大绝对差"这一列会逐次漂移（实测 600x800 在 8.58e-06 ~ 1.05e-05 之间浮动）。
#: 结论虽然不变，但**文档里写下的具体数字就无法复现** —— 违反本项目
#: "所有指标都要有口径"的纪律。固定种子后，本表可逐位复现。
PROBE_SEED = 20260925


def _probe_input(h: int, w: int) -> "torch.Tensor":
    """生成确定性的探针输入。

    每个尺寸用**独立**的种子（与 h/w 绑定），而不是全局 seed 后顺序消耗：
    这样增删 ``--verify-sizes`` 里的某一项，**不会改变其它行的数字**，
    表与表之间才能逐行对照。
    """
    gen = torch.Generator().manual_seed(PROBE_SEED + h * 100003 + w)
    return torch.rand(1, 3, h, w, generator=gen)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    cfg = TrainConfig.from_yaml(PROJECT_ROOT / args.config)
    ckpt_path = PROJECT_ROOT / args.checkpoint
    out_path = PROJECT_ROOT / args.output

    print(f"=== D5 导出 ONNX ===")
    print(f"  训练配置 : {args.config}")
    print(f"  权重     : {args.checkpoint}")
    print(f"  检测器   : {cfg.detector}   类别 {cfg.classes}（{len(cfg.classes)} 类前景）")
    print(f"  输出     : {args.output}")

    if not ckpt_path.exists():
        raise SystemExit(f"权重不存在：{ckpt_path}\n先跑 scripts/train.py，或用 --checkpoint 指定。")

    model = build_detector(
        cfg.detector, len(cfg.classes), pretrained=False,
        min_size=cfg.min_size, max_size=cfg.max_size,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  权重载入 : OK（训练配置 epoch={ckpt.get('epoch')}）")

    # 规格：动态轴时 sample 尺寸只用于 trace，不代表运行限制
    if args.fixed_hw:
        h, w = _parse_hw(args.fixed_hw)
        dynamic = False
    else:
        h, w = 600, 800
        dynamic = True

    spec = ExportSpec(
        opset=args.opset, ir_version=args.ir_version, dynamic_hw=dynamic,
        sample_height=h, sample_width=w,
        detector=cfg.detector, num_classes=len(cfg.classes), classes=tuple(cfg.classes),
    )
    print(f"  规格     : opset={spec.opset}  IR={spec.ir_version}  "
          f"动态H/W={spec.dynamic_hw}  trace尺寸={spec.sample_height}x{spec.sample_width}")

    print("\n=== 导出 ===")
    path, used = export_onnx(model, out_path, spec=spec)
    info = inspect_onnx(path)
    print(f"  文件     : {path.relative_to(PROJECT_ROOT)}  ({info['size_bytes'] / 1e6:.1f} MB)")
    print(f"  IR/opset : {info['ir_version']} / {info['opset']}")
    print(f"  计算图   : {info['node_count']} 个节点, {info['initializer_count']} 个初始化张量, "
          f"{len(info['op_types'])} 种算子")
    print(f"  输入     : {info['inputs']}")
    print(f"  输出     : {info['outputs']}")
    print("  算子清单 :")
    for op, n in info["op_types"].items():
        print(f"      {op:<24} {n}")

    spec_path = used.to_json(path.with_suffix(".onnx.spec.json"))
    print(f"\n  规格文件 : {spec_path.relative_to(PROJECT_ROOT)}")

    # ── 验收 2：onnxruntime 能否加载 ──────────────────────────
    print("\n=== 验收：onnxruntime 加载 ===")
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    print(f"  onnxruntime {ort.__version__} 加载成功")
    for i in sess.get_inputs():
        print(f"    输入 {i.name}: {i.shape} {i.type}")
    for o in sess.get_outputs():
        print(f"    输出 {o.name}: {o.shape} {o.type}")

    # ── 验收 3：与 PyTorch 原始输出的数值一致性 ────────────────
    # 这是 D6「逐框对齐」的前置：图本身先对上，后处理才谈得上对上。
    print("\n=== 验收：PyTorch vs ORT 原始输出一致性 ===")
    print(f"  （探针输入按固定种子生成，故本表数字可精确复现；seed={PROBE_SEED}）")
    wrapper = DetectorRawOutput(model).eval()
    worst = 0.0
    for text in args.verify_sizes:
        vh, vw = _parse_hw(text)
        x = _probe_input(vh, vw)
        with torch.no_grad():
            torch_out = wrapper(x)
        ort_out = sess.run(None, {"images": x.numpy()})

        shapes_ok = all(t.shape == o.shape for t, o in zip(torch_out, ort_out))
        diffs = [float(np.abs(t.numpy() - o).max()) for t, o in zip(torch_out, ort_out)]
        worst = max(worst, max(diffs))
        flag = "OK " if shapes_ok else "形状不一致!"
        print(f"  {vh:>4}x{vw:<4}  形状一致={shapes_ok}  {flag}  最大绝对差={max(diffs):.3e}")
        if not shapes_ok:
            raise SystemExit(
                "输出形状不一致 —— 动态轴没真正生效（某些算子把 trace 尺寸固化进了图）。"
                "改用 --fixed-hw 导出定尺寸图。"
            )

    print(f"\n  全部尺寸的最大绝对差：{worst:.3e}")
    print("\n=== D5 完成 ===")
    print("  ⚠️ 本产物只含 backbone+FPN+head，**不含后处理**（anchor/解码/NMS 在图外）。")
    print("     谁要用它就得自己实现后处理；逐框对齐属 D6。")
    print(f"  ⚠️ 下一步请在 Netron 中打开 {path.name} 确认计算图可见（D5 验收物）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
