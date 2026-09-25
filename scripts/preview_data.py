"""D2 数据集预览：读得进来 + 画得出图 + 出得了统计。

薄入口：解析参数 -> 调 ``lingmou.io`` -> 落盘。
业务逻辑（配置映射、统计口径）全在 ``src/lingmou/io/`` 里，可被测试覆盖。

用法::

    .\\.venv\\Scripts\\python.exe scripts\\preview_data.py
    .\\.venv\\Scripts\\python.exe scripts\\preview_data.py --out artifacts\\d2 --grid-items 6

产物（默认落在 ``artifacts/d2/``）：
- ``<数据集名>_sample.png``  抽样网格图（带框），D2 验收证据
- ``<数据集名>_stats.json``  统计量，供报告引用
- ``summary.md``             汇总表
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Windows 终端默认 GBK，直接 print 中文会乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lingmou.io import build_all, load_config, save_grid, summarize, summary_markdown  # noqa: E402
from lingmou.io.schema import Sample  # noqa: E402


def pick_representative(samples: list[Sample], n: int) -> list[Sample]:
    """挑框最多的几张作图。

    挑"框多"而不是随机：网格图要能一眼看出类别是否画对、密集目标有没有漏，
    空图或单目标图看不出问题。排序键带上 ``image_id`` 保证结果可复现。
    """
    with_objects = [s for s in samples if s.has_objects]
    with_objects.sort(key=lambda s: (-len(s.boxes), s.image_id))
    return with_objects[:n]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="D2 数据集预览与验收出图")
    ap.add_argument("--config", default="configs/datasets.yaml", help="数据集配置")
    ap.add_argument("--out", default="artifacts/d2", help="产物目录")
    ap.add_argument("--grid-items", type=int, default=4, help="每个数据集画几张")
    ap.add_argument("--only", nargs="*", help="只处理指定数据集名")
    args = ap.parse_args(argv)

    config = load_config(PROJECT_ROOT / args.config)
    if args.only:
        config["datasets"] = {k: v for k, v in config["datasets"].items() if k in args.only}
        if not config["datasets"]:
            print(f"没有匹配的数据集。可选：{list(load_config(PROJECT_ROOT / args.config)['datasets'])}")
            return 2

    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"构建数据集（配置：{args.config}）...")
    datasets = build_all(config, project_root=PROJECT_ROOT)

    sections: list[str] = ["# D2 数据集统计", ""]
    for name, samples in datasets.items():
        spec = config["datasets"][name]
        stats = summarize(samples)
        print(f"\n[{name}] {spec.get('title', name)}  图像={stats['images']}  框={stats['boxes']}")

        grid_path = out_dir / f"{name}_sample.png"
        chosen = pick_representative(samples, args.grid_items)
        if chosen:
            save_grid(grid_path, chosen, columns=2, cell=480)
            print(f"  网格图 -> {grid_path.relative_to(PROJECT_ROOT)}  （{len(chosen)} 张）")
        else:
            print("  该数据集没有含目标的样本，跳过出图")

        (out_dir / f"{name}_stats.json").write_text(
            json.dumps(
                {"name": name, "title": spec.get("title", name), "classes": spec.get("classes"), **stats},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        sections.append(summary_markdown(spec.get("title", name), stats))
        sections.append("")

    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"\n汇总 -> {summary_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
