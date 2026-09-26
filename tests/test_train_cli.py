"""``scripts/train.py`` 的命令行覆盖行为测试。

为什么要为"一个脚本参数"写测试
------------------------------
``--drop-emptied`` 是**消融实验**的入口。消融结论要写进 `docs/decisions.md`，
所以这条路径必须可靠：它若静默失效（比如覆盖没生效、却仍然照常跑完），
我们就会拿到一份"以为做了对照、其实两组只差一个随机种子"的实验数据，
并据此做出错误决策。这比程序崩溃危险得多。

同样重要的是：覆盖**不得改写配置文件**，也不得污染已加载的配置对象。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from lingmou.engine import TrainConfig
from lingmou.io.imageio import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train.py"


def _load_train_module():
    """按路径加载 scripts/train.py（它不在包内，无法直接 import）。"""
    spec = importlib.util.spec_from_file_location("train_script", TRAIN_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mini_nwpu(tmp_path: Path):
    """迷你 NWPU：2 张正样本（airplane / harbor）+ 1 张真背景。

    正样本里有一张只含 ``harbor``——按 3 类过滤后会被滤空，
    正是 ``drop_emptied`` 要区分的对象。
    """
    root = tmp_path / "NWPU VHR-10 dataset"
    img = np.zeros((48, 48, 3), dtype=np.uint8)
    for name in ("001", "002"):
        save_image(root / "positive image set" / f"{name}.jpg", img)
    save_image(root / "negative image set" / "001.jpg", img)

    gt = root / "ground truth"
    gt.mkdir(parents=True)
    (gt / "001.txt").write_text("(4,4),(20,20),1\n", encoding="utf-8")    # airplane -> 保留
    (gt / "002.txt").write_text("(4,4),(20,20),8\n", encoding="utf-8")    # harbor   -> 滤空
    return root


@pytest.fixture
def datasets_cfg(mini_nwpu: Path) -> dict:
    return {
        "datasets": {
            "nwpu_vhr10": {
                "reader": "nwpu",
                "root": str(mini_nwpu),
                "classes": ["airplane", "ship", "vehicle"],
                "include_negative": True,
                "drop_emptied": True,
            }
        }
    }


def test_no_override_uses_config_value(datasets_cfg: dict) -> None:
    mod = _load_train_module()
    cfg = TrainConfig(classes=["airplane", "ship", "vehicle"])
    samples = mod.build_samples(cfg, datasets_cfg)
    ids = [s.image_id for s in samples]
    # drop_emptied=True -> 被 harbor 滤空的 002 应当被丢弃
    assert "nwpu/positive/002" not in ids
    assert "nwpu/positive/001" in ids
    assert "nwpu/negative/001" in ids


def test_override_to_false_keeps_emptied_images(datasets_cfg: dict) -> None:
    mod = _load_train_module()
    cfg = TrainConfig(classes=["airplane", "ship", "vehicle"])
    samples = mod.build_samples(cfg, datasets_cfg, drop_emptied=False)
    ids = [s.image_id for s in samples]
    assert "nwpu/positive/002" in ids       # 被滤空但保留，作为"假背景"
    assert len(samples) == 3


def test_override_does_not_mutate_loaded_config(datasets_cfg: dict) -> None:
    """覆盖只作用于内存副本 —— 否则同一次运行里后续步骤会读到被改过的配置。"""
    mod = _load_train_module()
    cfg = TrainConfig(classes=["airplane", "ship", "vehicle"])
    mod.build_samples(cfg, datasets_cfg, drop_emptied=False)
    assert datasets_cfg["datasets"]["nwpu_vhr10"]["drop_emptied"] is True


def test_class_mismatch_is_rejected(datasets_cfg: dict) -> None:
    """类别顺序决定标签序号，两处配置不一致必须报错而不是将就。"""
    mod = _load_train_module()
    cfg = TrainConfig(classes=["ship", "airplane", "vehicle"])   # 顺序变了
    with pytest.raises(SystemExit, match="类别不一致"):
        mod.build_samples(cfg, datasets_cfg)
