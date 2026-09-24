# configs/ —— 配置目录

## 为什么要有这一层

**不要把超参数写死在代码里。** 训练配置、推理配置、后端配置都放这里，
好处有三个：

1. 做消融实验时改配置不动代码，`docs/decisions.md` 里能直接写"用哪个配置跑的"；
2. 评测报告里的数字能追溯到具体配置，可复现；
3. 部署时换后端只改配置，不改代码——这正是 `runtime` 层抽象的意义。

## 命名约定

```
configs/
├─ infer.yaml          # 推理配置（后端、provider、阈值）
├─ train_baseline.yaml # 基线训练配置（FP32，未压缩）
└─ train_lite.yaml     # 轻量化训练配置
```

> **基线配置要保留**。没有 FP32 基线，"压缩率 80%""精度掉 5pp"这些指标就没有参照物。

## 示例：infer.yaml

```yaml
model:
  path: artifacts/model.onnx

# 换芯片 = 换这一段，业务代码不动
runtime:
  backend: ort
  providers:
    - CPUExecutionProvider      # 开发机/精度对齐
    # - CANNExecutionProvider   # 昇腾板卡
    # - CUDAExecutionProvider   # 本机 NVIDIA（仅开发期）

# 业务参数：调这些不改代码，但会影响精度与耗时，改动要记进 docs/decisions.md
infer:
  input_size: [640, 640]
  conf_threshold: 0.25
  iou_threshold: 0.45
  warmup: 5
```
