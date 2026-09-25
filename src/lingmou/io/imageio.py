"""图像解码/编码。

为什么不用 ``cv2.imread`` 了事
-----------------------------
``cv2.imread`` 在 Windows 上**遇到非 ASCII 路径会静默返回 None**（底层用 ANSI 版
``fopen``）。本项目的目录名里就带空格和中文（``NWPU VHR-10 dataset``、
将来的中文数据集目录），踩这个坑的表现是"图读不出来但不报错"，极难排查。
因此统一走 ``np.fromfile`` + ``cv2.imdecode``：先把字节读进来，再在内存里解码，
绕开路径编码问题。

解码开销是 200 ms 预算里的大头
------------------------------
端侧 200 ms 是**全链路**预算，图像解码不是免费的。所有推理链路里
取图必须走本模块，以便将来在解码处统一打点（见 ``scripts/README.md``）。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class ImageDecodeError(OSError):
    """图像读取或解码失败。"""


def load_image(path: str | Path, flags: int | None = None) -> np.ndarray:
    """读取图像为 BGR ``uint8`` HWC 数组。

    :param path: 图像路径（支持非 ASCII 路径）
    :param flags: ``cv2.IMREAD_*``；默认三通道彩色
    :raises ImageDecodeError: 文件不存在或内容不是可解码图像
    """
    p = Path(path)
    if not p.exists():
        raise ImageDecodeError(f"图像不存在：{p}")
    buf = np.fromfile(str(p), dtype=np.uint8)
    if buf.size == 0:
        raise ImageDecodeError(f"图像为空文件：{p}")
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR if flags is None else flags)
    if img is None:
        raise ImageDecodeError(f"解码失败（可能不是图像，或扩展名与实际格式不符）：{p}")
    return img


def image_size(path: str | Path) -> tuple[int, int]:
    """只取 ``(width, height)``，**不解码像素**。

    准备数据集时要读上千张图的尺寸，整图解码纯属浪费。走 Pillow 的
    ``Image.open``，它只读文件头。注意不要用 ``cv2.imdecode`` 加快捷参数来省事：
    JPEG 解码需要完整码流，喂部分字节会失败或给出错误尺寸。
    """
    from PIL import Image

    p = Path(path)
    try:
        with Image.open(p) as im:
            return int(im.width), int(im.height)
    except Exception as exc:  # Pillow 的异常类型较杂，统一归口
        raise ImageDecodeError(f"读取图像尺寸失败：{p}（{exc}）") from exc


def save_image(path: str | Path, image: np.ndarray) -> Path:
    """写出图像，同样绕开非 ASCII 路径问题。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix or ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise ImageDecodeError(f"编码失败：{p}")
    buf.tofile(str(p))
    return p
