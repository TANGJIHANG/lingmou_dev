"""国产算力后端冒烟测试（hello-world 级）。

为什么需要它
------------
``docs/decisions.md`` 里写着："应尽早对板子做一次 hello-world 级冒烟测试"。
原因是我们练部署链路用的是 **WSL2（x86_64 / Ubuntu）**，而板子是
**aarch64 / openEuler + 昇腾**——**两者不等价**，WSL 上能跑不代表板子上能跑。

这个脚本的作用就是把"能跑"这件事压缩成一条**可复制的命令**：
在目标机器上跑一次，得到的环境描述与数值吻合度，直接进
《国产化适配测试报告》的"环境清单"与"首轮冒烟"两节。

它刻意**不依赖 torch**：板子上未必装得动 PyTorch，
而参考值用 numpy 现算（``X @ W + b``），任何有 numpy 的环境都能验证。

用法::

    # 开发机 / WSL：CPU 后端
    python tools/backend_smoke.py

    # 本机 NVIDIA（仅开发期对拍用）
    python tools/backend_smoke.py --providers CUDAExecutionProvider

    # 昇腾板卡（需 onnxruntime 的 CANN 构建）
    python tools/backend_smoke.py --providers CANNExecutionProvider

退出码 0 = 通过，1 = 数值不一致，2 = 环境/后端问题。
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np

# Windows 终端默认 GBK，直接 print 中文会乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))   # 板子上通常没装 -e .，靠路径也能跑

from lingmou.runtime import registry            # noqa: E402
from lingmou.runtime.base import BackendError   # noqa: E402

IN_FEATURES = 8
OUT_FEATURES = 4
SEED = 20260925


def build_tiny_model(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """造一个最小 ONNX 图：``Y = X @ W + b``。

    返回 ``(X, W, b)`` 供 numpy 侧算参考值。
    选 MatMul+Add 而不是 Conv：它是所有推理后端**必定实现**的算子，
    冒烟测试要暴露的是"环境装没装对"，而不是"某算子支持不支持"。
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(SEED)
    w = rng.standard_normal((IN_FEATURES, OUT_FEATURES)).astype(np.float32)
    b = rng.standard_normal((OUT_FEATURES,)).astype(np.float32)
    x = rng.standard_normal((1, IN_FEATURES)).astype(np.float32)

    nodes = [
        helper.make_node("MatMul", ["X", "W"], ["mm"], name="matmul"),
        helper.make_node("Add", ["mm", "b"], ["Y"], name="add"),
    ]
    graph = helper.make_graph(
        nodes,
        "backend_smoke",
        inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, IN_FEATURES])],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, OUT_FEATURES])],
        initializer=[
            numpy_helper.from_array(w, "W"),
            numpy_helper.from_array(b, "b"),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="lingmou-backend-smoke",
    )
    # ── 关键：压低 IR version ──────────────────────────────────
    # onnx 与 onnxruntime 是两个独立发版的包，**onnx 写出的 IR 版本经常高于
    # onnxruntime 能读的上限**。本项目实测：
    #     onnx 1.23.0 默认写 ir_version=14
    #     onnxruntime 1.23.2 最高支持 11
    #     -> "Unsupported model IR version: 14, max supported IR version: 11"
    # 这正是 D5 导出 -> D6 ORT 推理 最典型的版本错配。
    # 冒烟测试要暴露的是"环境装没装对"，不能被自家测试模型的版本问题干扰，
    # 故主动压到 8（opset 13 要求 IR >= 7，8 是安全且被广泛支持的值）。
    model.ir_version = 8
    # 不跑 checker 的完整校验：板卡环境的 onnx 版本可能偏旧，checker 会因
    # 无关的 opset 规则变化报错，反而掩盖真正要测的推理链路。
    onnx.save(model, str(path))
    return x, w, b


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="国产算力后端冒烟测试")
    ap.add_argument("--backend", default="ort", help="后端名（runtime.registry 中注册的）")
    ap.add_argument(
        "--providers", nargs="+", default=["CPUExecutionProvider"],
        help="执行后端；板卡上换成 CANNExecutionProvider",
    )
    ap.add_argument("--atol", type=float, default=1e-4, help="允许的最大绝对误差")
    ap.add_argument("--keep-model", help="把测试用 ONNX 保存到该路径（便于 Netron 查看）")
    args = ap.parse_args(argv)

    env = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    print("=== 环境 ===")
    for k, v in env.items():
        print(f"  {k:9s}: {v}")

    tmpdir = None
    if args.keep_model:
        model_path = Path(args.keep_model)
        model_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        tmpdir = tempfile.TemporaryDirectory()
        model_path = Path(tmpdir.name) / "smoke.onnx"

    x, w, b = build_tiny_model(model_path)
    expected = (x @ w + b).astype(np.float32)

    print("\n=== 推理 ===")
    try:
        backend = registry.create(args.backend, providers=args.providers)
        backend.load(str(model_path))
        actual = backend.infer({"X": x})["Y"]
        info = backend.describe()
    except BackendError as exc:
        # 环境类问题与数值类问题要分开报，否则上板时排查方向会被带偏。
        # 底层原因链必须打全：最常见的就是 IR version / opset 不受支持。
        print(f"  [环境失败] {exc}", file=sys.stderr)
        cause = exc.__cause__
        while cause is not None:
            print(f"    底层: {type(cause).__name__}: {cause}", file=sys.stderr)
            cause = cause.__cause__
        return 2

    print(f"  输入 X      : {x.shape} {x.dtype}")
    print(f"  输出 Y      : {actual.shape} {actual.dtype}")
    print(f"  期望值      : {np.array2string(expected[0], precision=6)}")
    print(f"  实际值      : {np.array2string(actual[0], precision=6)}")

    diff = float(np.abs(actual - expected).max())
    print(f"  最大绝对误差: {diff:.3e}   阈值 {args.atol:.1e}")

    print("\n=== 后端自述（进《国产化适配测试报告》环境清单）===")
    print(json.dumps(info, ensure_ascii=False, indent=2))

    if tmpdir is not None:
        tmpdir.cleanup()
    else:
        print(f"\n模型已保存：{model_path}")

    if diff > args.atol:
        print(f"\nFAIL：数值不一致（{diff:.3e} > {args.atol:.1e}）", file=sys.stderr)
        return 1
    if info.get("has_nvidia_components"):
        # 不算失败，但国产化叙事里必须显式提示
        print("\n⚠️  当前执行后端含 NVIDIA 组件 —— 国产化自证要求此项为 False")
    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
