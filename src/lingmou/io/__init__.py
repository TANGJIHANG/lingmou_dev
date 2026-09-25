"""数据接入层 —— 光学 / 红外 / SAR 统一接入。

职责
----
把三类异构数据源归一化为统一的下游可消费形式，是系统架构图的入口层。

统一输入协议（已实现）
----------------------
每条记录都长成 :class:`~lingmou.io.schema.Sample`：``source_type`` / ``timestamp`` /
``geo_hint`` / ``payload`` 四字段，三源共用。下游（``engine`` / ``tasks``）
不需要为每种源写分支。

已实现（D2，2026-09-25）
------------------------
- :mod:`~lingmou.io.schema`  统一数据模型（``SourceType`` / ``Box`` / ``Sample``）
- :mod:`~lingmou.io.imageio` 图像解码，绕开 Windows 非 ASCII 路径静默失败
- :mod:`~lingmou.io.nwpu`    NWPU VHR-10 读取（光学，VOC 风格 txt）
- :mod:`~lingmou.io.ssdd`    SSDD 读取（SAR，DOTA 八点旋转框）
- :mod:`~lingmou.io.stats`   统一口径的数据集统计
- :mod:`~lingmou.io.visualize` 画框与拼图（验收证据）

待实现
------
- **流式输入路径**：文件用于复现与评测，流用于演示
  （命题要求"标准化多源数据输入"）。当前只有文件路径。
- **红外源**：D2 只接了光学与 SAR，``SourceType.INFRARED`` 已定义但无读取器。
- **单源可用性自评**：输出每个源的置信度/可用性，供融合层做**动态权重门控**
  （光学白天好、红外夜间好、SAR 全天候）。
- **时间与地理元数据**：NWPU / SSDD 都不提供 ``timestamp`` / ``geo_hint``，
  当前如实置 ``None``。多源融合演示需要时间轴，届时须补合成时间戳或改用带元数据的源。
- **解码耗时打点**：JPEG / 视频帧解码在端侧是 200 ms 预算里的大头之一，
  必须在 :mod:`~lingmou.io.imageio` 统一打点，不要当作免费操作。

依赖约束
--------
本层只能依赖 ``runtime`` 层，不得直接 import 推理框架。
"""

from __future__ import annotations

from lingmou.io.config import DatasetConfigError, build_all, build_dataset, load_config
from lingmou.io.imageio import ImageDecodeError, image_size, load_image, save_image
from lingmou.io.nwpu import D2_CLASSES, NWPU_CLASSES, read_nwpu_vhr10
from lingmou.io.schema import Box, Sample, SourceType
from lingmou.io.ssdd import (
    ANALYSIS_ONLY_SPLITS,
    SSDD_CLASSES,
    read_ssdd_dota,
)
from lingmou.io.stats import by_source, label_counts, summarize, summary_markdown
from lingmou.io.visualize import draw_boxes, make_grid, save_grid

__all__ = [
    # 数据模型
    "SourceType",
    "Box",
    "Sample",
    # 图像 IO
    "load_image",
    "save_image",
    "image_size",
    "ImageDecodeError",
    # 读取器
    "read_nwpu_vhr10",
    "NWPU_CLASSES",
    "D2_CLASSES",
    "read_ssdd_dota",
    "SSDD_CLASSES",
    "ANALYSIS_ONLY_SPLITS",
    # 统计与可视化
    "summarize",
    "label_counts",
    "by_source",
    "summary_markdown",
    "draw_boxes",
    "make_grid",
    "save_grid",
    # 配置
    "load_config",
    "build_dataset",
    "build_all",
    "DatasetConfigError",
]
