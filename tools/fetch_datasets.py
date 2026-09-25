"""数据集获取工具（D2）

职责
----
只做三件事：**下载 → 校验 → 解包**。

目录约定（见 ``data/README.md``）
--------------------------------
- ``data/raw/``     原始压缩包，下载后**只读**，永不修改
- ``data/interim/`` 解包产物（原始标注格式，尚未归一化）

之所以不直接解到 ``raw/``：一旦解包产物和原始包混在一起，
"这份图到底是官方原版还是我们改过的"就说不清了。

用法
----
    .\\.venv\\Scripts\\python.exe tools\\fetch_datasets.py            # 全部
    .\\.venv\\Scripts\\python.exe tools\\fetch_datasets.py nwpu_vhr10 # 单个

已存在的文件默认跳过；``--force`` 强制重下。
下载后在 ``data/raw/FETCH_MANIFEST.json`` 落一份实际大小与 SHA256 记录。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"

USER_AGENT = "lingmou-dataset-fetcher/0.1 (+research; contact: project maintainer)"


@dataclass(frozen=True)
class Dataset:
    """一条数据集获取记录。

    ``sha256`` 为 ``None`` 表示首次获取、哈希待回填；脚本会把实测值打印出来。
    """

    key: str
    title: str
    # 镜像直链。注意：本项目环境实测 huggingface.co 不可达、hf-mirror.com 可达，
    # 因此统一走镜像；原始出处记在 source_page 里，引用时以原始出处为准。
    url: str
    filename: str
    kind: str  # "zip" | "targz"
    source_page: str
    license_note: str
    sha256: str | None = None


DATASETS: tuple[Dataset, ...] = (
    Dataset(
        key="nwpu_vhr10",
        title="NWPU VHR-10 图像 + 原始 GT（光学，VOC 风格 txt）",
        url="https://hf-mirror.com/datasets/isaaccorley/vhr10/resolve/main/NWPU%20VHR-10%20dataset.zip",
        filename="NWPU_VHR-10_dataset.zip",
        kind="zip",
        source_page="https://huggingface.co/datasets/isaaccorley/vhr10",
        license_note="镜像标注 CC-BY-NC-4.0（非商业）；原数据集（西北工业大学）声明仅限研究用途",
    ),
    Dataset(
        key="nwpu_vhr10_coco",
        title="NWPU VHR-10 第三方 COCO 标注（仅作交叉校验，非训练用）",
        url="https://hf-mirror.com/datasets/isaaccorley/vhr10/resolve/main/annotations.json",
        filename="NWPU_VHR-10_annotations.json",
        kind="json",
        source_page="https://huggingface.co/datasets/isaaccorley/vhr10",
        license_note="同 nwpu_vhr10",
    ),
    Dataset(
        key="ssdd",
        title="SSDD (SAR 舰船, DOTA 标注)",
        url="https://hf-mirror.com/datasets/Gordy333/ssdd-rsdd-data/resolve/main/SSDD_DOTA.tar.gz",
        filename="SSDD_DOTA.tar.gz",
        kind="targz",
        source_page="https://github.com/TianwenZhang0825/Official-SSDD",
        license_note=(
            "原仓库 Apache-2.0；影像合成自 RadarSat-2 / TerraSAR-X / Sentinel-1，"
            "其中 TerraSAR-X 部分存在二级传感器权利不确定性（详见 DATA_SOURCES.md）"
        ),
    ),
)


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download(ds: Dataset, force: bool = False) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    dest = RAW / ds.filename

    if dest.exists() and not force:
        print(f"[skip] {ds.key}: 已存在 {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    print(f"[get ] {ds.key}: {ds.url}")
    req = urllib.request.Request(ds.url, headers={"User-Agent": USER_AGENT})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 (固定白名单 URL)
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        with tmp.open("wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100 // total
                    print(f"\r       {done / 1e6:7.1f}/{total / 1e6:.1f} MB ({pct:3d}%)", end="")
        print()
    tmp.replace(dest)

    digest = sha256_of(dest)
    if ds.sha256 and digest != ds.sha256:
        # 哈希不符意味着"同一 URL 的内容变了"，必须人工确认，不能默默继续
        raise SystemExit(f"[FAIL] {ds.key} SHA256 不匹配\n  期望 {ds.sha256}\n  实际 {digest}")
    print(f"[hash] {ds.key}: {digest}")
    return dest


def extract(ds: Dataset, archive: Path) -> Path:
    """解包到 data/interim/<key>/。已解过则跳过。"""
    out_dir = INTERIM / ds.key
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"[skip] {ds.key}: 已解包 -> {out_dir}")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if ds.kind == "zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out_dir)
    elif ds.kind == "targz":
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(out_dir)  # noqa: S202 (来源固定，非用户输入)
    else:
        shutil.copy2(archive, out_dir / archive.name)
    print(f"[unpk] {ds.key} -> {out_dir}")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="下载 D2 数据集（NWPU VHR-10 / SSDD）")
    ap.add_argument("keys", nargs="*", help="只处理指定 key，留空表示全部")
    ap.add_argument("--force", action="store_true", help="忽略已存在文件，强制重下")
    args = ap.parse_args(argv)

    wanted = [d for d in DATASETS if not args.keys or d.key in args.keys]
    if not wanted:
        print(f"未知 key。可选：{[d.key for d in DATASETS]}", file=sys.stderr)
        return 2

    records = []
    for ds in wanted:
        archive = download(ds, force=args.force)
        extract(ds, archive)
        records.append(
            {
                **asdict(ds),
                "local_path": str(archive.relative_to(ROOT)).replace("\\", "/"),
                "bytes": archive.stat().st_size,
                "sha256_actual": sha256_of(archive),
            }
        )

    manifest = RAW / "FETCH_MANIFEST.json"
    manifest.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] 记录已写入 {manifest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
