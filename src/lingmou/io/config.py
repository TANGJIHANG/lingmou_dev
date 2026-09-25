"""由 ``configs/datasets.yaml`` 构造 :class:`~lingmou.io.schema.Sample` 列表。

为什么这层不放脚本里
--------------------
``scripts/README.md`` 规定"业务逻辑不要写在 scripts/ 里，否则无法被测试和复用"。
"配置项 -> 哪个读取器 -> 哪个目录"这套映射属于业务逻辑，
放在这里就能被 ``tests/test_io.py`` 覆盖，脚本只剩参数解析与打印。
"""

from __future__ import annotations

from pathlib import Path

import yaml

from lingmou.io.nwpu import read_nwpu_vhr10
from lingmou.io.schema import Sample
from lingmou.io.ssdd import read_ssdd_dota


class DatasetConfigError(ValueError):
    """数据集配置缺失或非法。"""


def load_config(path: str | Path) -> dict:
    """读取数据集配置 YAML。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"数据集配置不存在：{p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if "datasets" not in data or not data["datasets"]:
        raise DatasetConfigError(f"{p} 里没有 datasets 段")
    return data


def build_dataset(name: str, spec: dict, *, project_root: str | Path) -> list[Sample]:
    """按一条配置构造样本列表。

    ``root`` 支持相对项目根目录的相对路径，便于团队成员各自放置数据。
    """
    project_root = Path(project_root)
    reader = spec.get("reader")
    root = project_root / spec["root"]
    classes = tuple(spec.get("classes") or ()) or None

    if reader == "nwpu":
        return read_nwpu_vhr10(
            root,
            keep=classes,
            include_negative=bool(spec.get("include_negative", True)),
            drop_emptied=bool(spec.get("drop_emptied", False)),
        )
    if reader == "ssdd":
        samples: list[Sample] = []
        for split in spec.get("splits", ["train"]):
            samples.extend(read_ssdd_dota(root, split=split, keep=classes))
        samples.sort(key=lambda s: s.image_id)
        return samples
    raise DatasetConfigError(f"数据集 {name!r} 的 reader={reader!r} 未知（可选：nwpu / ssdd）")


def build_all(config: dict, *, project_root: str | Path) -> dict[str, list[Sample]]:
    """构造配置中的全部数据集，键为数据集名。"""
    return {
        name: build_dataset(name, spec, project_root=project_root)
        for name, spec in config["datasets"].items()
    }
