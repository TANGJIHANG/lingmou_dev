"""io 层（数据接入）单元测试。

分成两组：
1. **纯单元测试** —— 用 ``tmp_path`` 造迷你数据集，不依赖 ``data/`` 是否存在。
   真实数据集不进 Git（见 ``data/README.md``），因此 CI / 新克隆必须能跑过。
2. **真实数据冒烟测试** —— 数据不在时自动跳过。

跑法::

    .\\.venv\\Scripts\\python.exe -m pytest -q
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lingmou.io.imageio import ImageDecodeError, image_size, load_image, save_image
from lingmou.io.nwpu import D2_CLASSES, NWPU_CLASSES, DatasetFormatError as NwpuFormatError
from lingmou.io.nwpu import parse_gt_text, read_nwpu_vhr10
from lingmou.io.schema import Box, Sample, SourceType
from lingmou.io.ssdd import DatasetFormatError as SsddFormatError
from lingmou.io.ssdd import parse_dota_label, read_ssdd_dota
from lingmou.io.stats import label_counts, summarize
from lingmou.io.visualize import label_color, make_grid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NWPU_ROOT = PROJECT_ROOT / "data" / "interim" / "nwpu_vhr10" / "NWPU VHR-10 dataset"
SSDD_ROOT = PROJECT_ROOT / "data" / "interim" / "ssdd" / "data" / "SSDD_DOTA"


# ── 夹具 ──────────────────────────────────────────────────

def _write_image(path: Path, w: int = 200, h: int = 150) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[10:60, 10:60] = 255
    return save_image(path, img)


@pytest.fixture
def nwpu_root(tmp_path: Path) -> Path:
    """迷你 NWPU VHR-10：**正负样本故意同名**，用于验证前缀命名。"""
    root = tmp_path / "NWPU VHR-10 dataset"
    _write_image(root / "positive image set" / "001.jpg")
    _write_image(root / "positive image set" / "002.jpg")
    _write_image(root / "negative image set" / "001.jpg")   # 与正样本同名
    gt = root / "ground truth"
    gt.mkdir(parents=True)
    # 注意：故意带右对齐空格 —— 严正则会在真实数据上静默丢掉 13 个文件
    (gt / "001.txt").write_text("( 10, 10),( 50, 50),1 \n", encoding="utf-8")
    (gt / "002.txt").write_text("( 20, 20),( 80, 90),10 \n", encoding="utf-8")
    return root


@pytest.fixture
def ssdd_root(tmp_path: Path) -> Path:
    """迷你 SSDD：含 DOTA 头行与一个旋转框。"""
    root = tmp_path / "SSDD_DOTA"
    _write_image(root / "JPEGImages" / "000001.jpg")
    ann = root / "Annotations_train"
    ann.mkdir(parents=True)
    (ann / "000001.txt").write_text(
        "imagesource:Sentinel-1\ngsd:10.0\n10 10 50 10 50 50 10 50 ship 1\n",
        encoding="utf-8",
    )
    return root


# ── schema ────────────────────────────────────────────────

def test_box_rejects_degenerate() -> None:
    with pytest.raises(ValueError, match="退化框"):
        Box(10, 10, 10, 50, "ship")


def test_box_from_xywh_roundtrip() -> None:
    b = Box.from_xywh(10, 20, 30, 40, "ship")
    assert (b.x1, b.y1, b.x2, b.y2) == (10, 20, 40, 60)
    assert b.as_xywh() == (10, 20, 30, 40)
    assert b.area == 1200


def test_box_from_polygon_keeps_original_points() -> None:
    """旋转框拍平后必须保留原始四点，否则信息不可逆丢失。"""
    b = Box.from_polygon([(10, 10), (50, 20), (40, 60), (0, 50)], "ship")
    assert b.polygon is not None and len(b.polygon) == 4
    assert (b.x1, b.y1, b.x2, b.y2) == (0, 10, 50, 60)


def test_box_clip_out_of_bounds() -> None:
    b = Box(-5, -5, 150, 120, "ship")
    clipped = b.clip(100, 100)
    assert clipped is not None
    assert (clipped.x1, clipped.y1, clipped.x2, clipped.y2) == (0, 0, 100, 100)
    assert Box(200, 200, 300, 300, "ship").clip(100, 100) is None


def test_sample_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="image_id"):
        Sample("", SourceType.SAR, Path("x.jpg"), 10, 10)


def test_sample_rejects_bad_array_shape() -> None:
    with pytest.raises(ValueError, match="HWC"):
        Sample("x", SourceType.SAR, np.zeros((4, 4), dtype=np.uint8), 4, 4)


def test_sample_filter_labels_turns_empty_into_negative() -> None:
    """类别过滤把框全滤掉时，样本应变成负样本，而不是报错。"""
    s = Sample(
        "x", SourceType.OPTICAL, Path("x.jpg"), 100, 100,
        boxes=(Box(1, 1, 9, 9, "airplane"), Box(2, 2, 8, 8, "vehicle")),
    )
    assert s.has_objects
    kept = s.filter_labels(("ship",))
    assert not kept.has_objects
    assert kept.labels == ()
    assert len(s.filter_labels(("airplane",)).boxes) == 1


# ── 解析 ──────────────────────────────────────────────────

def test_parse_gt_text_tolerates_right_aligned_spaces() -> None:
    boxes = parse_gt_text("( 108, 60),( 192,154),1 \n(  5,  6),( 20, 30),2 \n")
    assert len(boxes) == 2
    assert boxes[0].label == "airplane"
    assert boxes[1].label == "ship"


def test_parse_gt_text_rejects_unknown_class() -> None:
    with pytest.raises(NwpuFormatError, match="未知类别"):
        parse_gt_text("(1,1),(9,9),99", source="t.txt")


def test_parse_gt_text_rejects_malformed_line() -> None:
    """解不出必须报错 —— 静默跳过正是 2% 数据丢失的成因。"""
    with pytest.raises(NwpuFormatError, match="无法解析"):
        parse_gt_text("(1,1),(9,9),1\n这不是标注\n", source="t.txt")


def test_parse_dota_label_skips_header_lines() -> None:
    boxes = parse_dota_label("imagesource:GF-2\ngsd:0.5\n10 10 50 10 50 50 10 50 ship 1\n")
    assert len(boxes) == 1
    assert boxes[0].label == "ship"
    assert boxes[0].difficult == 1
    assert boxes[0].polygon is not None


def test_parse_dota_label_rejects_short_line() -> None:
    with pytest.raises(SsddFormatError, match="字段数不足"):
        parse_dota_label("10 10 50 10 ship\n", source="t.txt")


# ── 读取器（迷你数据集） ─────────────────────────────────

def test_read_nwpu_namespaces_colliding_filenames(nwpu_root: Path) -> None:
    """正负样本同名是真实数据里的坑；image_id 必须带前缀且唯一。"""
    samples = read_nwpu_vhr10(nwpu_root)
    ids = [s.image_id for s in samples]
    assert len(ids) == len(set(ids)) == 3
    assert "nwpu/positive/001" in ids
    assert "nwpu/negative/001" in ids
    by_id = {s.image_id: s for s in samples}
    assert by_id["nwpu/positive/001"].has_objects
    assert not by_id["nwpu/negative/001"].has_objects


def test_read_nwpu_defaults_to_three_classes(nwpu_root: Path) -> None:
    samples = read_nwpu_vhr10(nwpu_root)
    labels = {b.label for s in samples for b in s.boxes}
    assert labels <= set(D2_CLASSES)
    assert len(D2_CLASSES) == 3


def test_read_nwpu_keep_none_returns_all_ten(nwpu_root: Path) -> None:
    lanes = {b.label for s in read_nwpu_vhr10(nwpu_root, keep=None) for b in s.boxes}
    assert lanes <= set(NWPU_CLASSES.values())


def test_read_nwpu_strict_raises_on_missing_gt(nwpu_root: Path) -> None:
    _write_image(nwpu_root / "positive image set" / "003.jpg")
    with pytest.raises(NwpuFormatError, match="缺少 GT"):
        read_nwpu_vhr10(nwpu_root)


def test_read_nwpu_can_skip_negatives(nwpu_root: Path) -> None:
    assert len(read_nwpu_vhr10(nwpu_root, include_negative=False)) == 2


def test_read_nwpu_drop_emptied_distinguishes_fake_background(nwpu_root: Path) -> None:
    """被类别过滤滤空的图是"假背景"（画面里有目标，只是不要那个类）。

    ``drop_emptied`` 让这个取舍显式化：默认保留当负样本，置 True 则丢弃。
    """
    # 夹具里 002.jpg 是 vehicle，取 airplane/ship 会被滤空
    kept = read_nwpu_vhr10(nwpu_root, keep=("airplane", "ship"), include_negative=False)
    assert len(kept) == 2
    assert sum(1 for s in kept if not s.has_objects) == 1     # 假背景仍在

    dropped = read_nwpu_vhr10(
        nwpu_root, keep=("airplane", "ship"), include_negative=False, drop_emptied=True
    )
    assert [s.image_id for s in dropped] == ["nwpu/positive/001"]


def test_read_nwpu_drop_emptied_keeps_real_negatives(nwpu_root: Path) -> None:
    """``drop_emptied`` 只能丢"假背景"，不能连 150 张真背景图一起丢。"""
    samples = read_nwpu_vhr10(
        nwpu_root, keep=("airplane", "ship"), include_negative=True, drop_emptied=True
    )
    assert "nwpu/negative/001" in [s.image_id for s in samples]


def test_read_ssdd_dota_basic(ssdd_root: Path) -> None:
    samples = read_ssdd_dota(ssdd_root, split="train")
    assert len(samples) == 1
    s = samples[0]
    assert s.source_type is SourceType.SAR
    assert s.image_id == "ssdd/train/000001"
    assert s.labels == ("ship",)
    assert s.timestamp is None and s.geo_hint is None   # 数据集不提供，不编造


def test_read_ssdd_rejects_unknown_split(ssdd_root: Path) -> None:
    with pytest.raises(ValueError, match="未知划分"):
        read_ssdd_dota(ssdd_root, split="nope")


# ── 统计与可视化 ─────────────────────────────────────────

def test_summarize_excludes_negatives_from_mean() -> None:
    """平均框数只在含目标图上算 —— 混入负样本会稀释成无意义的数。"""
    samples = [
        Sample("a", SourceType.OPTICAL, Path("a.jpg"), 10, 10, boxes=(Box(1, 1, 5, 5, "ship"),)),
        Sample("b", SourceType.OPTICAL, Path("b.jpg"), 10, 10, boxes=(Box(1, 1, 5, 5, "ship"),)),
        Sample("c", SourceType.OPTICAL, Path("c.jpg"), 10, 10, boxes=()),
    ]
    st = summarize(samples)
    assert st["images"] == 3
    assert st["images_without_objects"] == 1
    assert st["boxes_per_positive_image_mean"] == 1.0
    assert st["labels"] == {"ship": 2}


def test_label_counts_sorted_desc() -> None:
    samples = [
        Sample("a", SourceType.SAR, Path("a.jpg"), 10, 10,
               boxes=(Box(1, 1, 5, 5, "ship"), Box(1, 1, 5, 5, "ship"))),
        Sample("b", SourceType.SAR, Path("b.jpg"), 10, 10, boxes=(Box(1, 1, 5, 5, "harbor"),)),
    ]
    assert list(label_counts(samples).items()) == [("ship", 2), ("harbor", 1)]


def test_label_color_is_process_stable() -> None:
    """颜色必须跨进程稳定，否则前后对比图没法看（内置 hash 会随机化）。"""
    assert label_color("ship") == label_color("ship")
    assert len({label_color(x) for x in ("ship", "airplane", "vehicle")}) == 3


def test_make_grid_draws_requested_count(tmp_path: Path) -> None:
    img_path = _write_image(tmp_path / "g.jpg", w=64, h=48)
    samples = [
        Sample(f"s{i}", SourceType.OPTICAL, img_path, 64, 48,
               boxes=(Box(2, 2, 20, 20, "airplane"),))
        for i in range(3)
    ]
    grid = make_grid(samples, columns=2, cell=64, max_items=3)
    # 3 张图排 2 列 -> 2 行；每行高度 = 图 64 + 信息栏 26
    assert grid.shape[0] == 2 * (64 + 26)
    assert grid.shape[1] == 2 * 64


def test_make_grid_rejects_empty() -> None:
    with pytest.raises(ValueError, match="没有样本"):
        make_grid([], columns=2)


def test_image_size_matches_saved(tmp_path: Path) -> None:
    p = _write_image(tmp_path / "size_probe.png", w=53, h=37)
    assert image_size(p) == (53, 37)


def test_load_image_raises_on_missing() -> None:
    with pytest.raises(ImageDecodeError, match="不存在"):
        load_image("definitely-not-here.jpg")


# ── 真实数据冒烟（数据不在则跳过） ───────────────────────

@pytest.mark.skipif(not NWPU_ROOT.is_dir(), reason="NWPU VHR-10 未下载")
def test_real_nwpu_parses_all_gt_files() -> None:
    """真实数据上：650 正样本全部解出，且三子集框数为实测的 1657。

    这两个数字是 D2 的口径基准，写死在这里是为了——
    **哪天换了数据版本或改坏了正则，测试会立刻失败**，而不是等到报告里数字对不上。
    """
    samples = read_nwpu_vhr10(NWPU_ROOT)
    positives = [s for s in samples if s.image_id.startswith("nwpu/positive/")]
    assert len(positives) == 650
    assert len(samples) == 800
    assert sum(len(s.boxes) for s in samples) == 1657


@pytest.mark.skipif(not NWPU_ROOT.is_dir(), reason="NWPU VHR-10 未下载")
def test_real_nwpu_three_class_coverage_is_measured() -> None:
    """3 类子集只覆盖 230/650 张正样本，其余 420 张被滤空。

    这个"35.4% 覆盖率"是 D2 选型的**已知代价**，固化成测试以免日后
    有人以为数据集有 800 张有效正样本。
    """
    samples = read_nwpu_vhr10(NWPU_ROOT, include_negative=False)
    with_objects = [s for s in samples if s.has_objects]
    assert len(with_objects) == 230
    assert len(samples) - len(with_objects) == 420


@pytest.mark.skipif(not SSDD_ROOT.is_dir(), reason="SSDD 未下载")
def test_real_ssdd_box_count_matches_official() -> None:
    """SSDD 官方声明 2587 个框；解不出这个数说明 DOTA 解析有问题。"""
    samples = read_ssdd_dota(SSDD_ROOT, split="train") + read_ssdd_dota(SSDD_ROOT, split="test")
    assert len(samples) == 1160
    assert sum(len(s.boxes) for s in samples) == 2587


@pytest.mark.skipif(not SSDD_ROOT.is_dir(), reason="SSDD 未下载")
def test_real_ssdd_has_no_background_images() -> None:
    """SSDD 1160 张图**全部含目标**，没有纯背景图。

    意味着单靠 SSDD 训不出"什么是背景"，虚警率会偏高；
    这也是 D2 同时接入 NWPU（含 150 张真背景）的实际价值之一。
    """
    samples = read_ssdd_dota(SSDD_ROOT, split="train") + read_ssdd_dota(SSDD_ROOT, split="test")
    assert all(s.has_objects for s in samples)
