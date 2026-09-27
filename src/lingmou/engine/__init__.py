"""解译引擎层 —— 检测 / 分割 / 精定位 / 多源融合 / 跟踪。

职责
----
系统的算法主体。**本层的输出就是命题要的"目标类别 + 位置态势"。**

已实现（D3–D4）
---------------
- :mod:`~lingmou.engine.dataset`   ``Sample`` -> torchvision 检测模型的格式转换与划分
- :mod:`~lingmou.engine.model`     基线检测器构建（FCOS / RetinaNet / Faster R-CNN）
- :mod:`~lingmou.engine.train`     训练循环、loss 历史、曲线绘制
- :mod:`~lingmou.engine.infer`     推理与"标注 vs 预测"可视化

已实现（D5）
------------
- :mod:`~lingmou.engine.export`    导出 ONNX（**只导出 backbone+FPN+head，后处理留在图外**）

待实现
------
- **预处理**：letterbox / resize / 归一化。**必须统计耗时**——端侧瓶颈是搬数据不是算数据。
- **后处理**：置信度阈值、NMS、旋转框解码、坐标反变换回原图。同样要打点计时。
  ⚠️ 这一项目前**不在 ONNX 图里**，是 D6 的主要工作之一。
- **多源融合决策**：本项目的**创新点所在**。注意口径——
  创新点在"初筛 + 融合决策"，"精判"是常规模型推理，不要包装成创新。
- **精定位**：像素坐标 → 态势坐标/经纬度的换算与精度评估。
- **精度评测**：mAP / 虚警率（属 D6+ 的 ``scripts/eval.py``，当前只算 loss）。

纪律
----
1. 先跑通，再优化；先有 FP32 基线，再谈压缩与加速。
   基线配置是 ``configs/train_baseline.yaml``，任何优化都要与它对比。
2. 任何一次优化都要在 docs/decisions.md 留消融记录。
3. 为部署而设计：写网络结构前先查目标芯片的算子支持表。
   **已实测的具体待查项**：FCOS 检测头的 GroupNorm 被导出为
   ``InstanceNormalization``（40 个）配 ``Reshape``。
   昇腾 CANN 对该算子的覆盖必须在上板前确认；
   若不支持，退路见 ``engine/model.py`` 顶部说明（换 RetinaNet，头是 Conv+ReLU）。
"""

from __future__ import annotations

from lingmou.engine.dataset import DetectionDataset, detection_collate, split_samples
from lingmou.engine.export import (
    DetectorRawOutput,
    ExportSpec,
    export_onnx,
    inspect_onnx,
)
from lingmou.engine.infer import (
    draw_gt_and_predictions,
    predict,
    predictions_to_boxes,
    save_prediction_grid,
)
from lingmou.engine.model import UnknownDetectorError, build_detector, count_parameters
from lingmou.engine.train import (
    History,
    TrainConfig,
    build_optimizer,
    build_scheduler,
    evaluate_loss,
    fit,
    make_loader,
    plot_loss_curve,
    resolve_device,
    save_checkpoint,
    train_one_epoch,
)

__all__ = [
    # 数据
    "DetectionDataset",
    "detection_collate",
    "split_samples",
    # 模型
    "build_detector",
    "count_parameters",
    "UnknownDetectorError",
    # 训练
    "TrainConfig",
    "History",
    "fit",
    "train_one_epoch",
    "evaluate_loss",
    "make_loader",
    "build_optimizer",
    "build_scheduler",
    "resolve_device",
    "save_checkpoint",
    "plot_loss_curve",
    # 推理
    "predict",
    "predictions_to_boxes",
    "draw_gt_and_predictions",
    "save_prediction_grid",
    # 导出（D5）
    "export_onnx",
    "inspect_onnx",
    "ExportSpec",
    "DetectorRawOutput",
]
