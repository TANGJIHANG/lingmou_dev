# 训练与基准环境清单

> 本文件是《测试评测报告》"环境清单"章节的原始数据来源。
> **数字必须与它成立的条件绑在一起**，否则不可复现。

## 1. 硬件

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，8 GB，计算能力 **sm_120** |
| GPU 驱动 | 592.01 |
| CPU | _待填：`wmic cpu get name` 或任务管理器_ |
| 内存 | _待填_ |
| 存储 | C 盘可用约 206 GB（创建 venv 时） |

## 2. 软件

| 项 | 值 |
|---|---|
| OS | _待填：Windows 版本_ |
| Python | 3.10.11（venv，路径 `.venv/`） |
| torch | **2.14.0+cu130** |
| CUDA（torch 构建） | 13.0 |
| arch list | `sm_75, sm_80, sm_86, sm_90, sm_100, sm_120` |
| `allow_tf32`（matmul） | _待填：`python -c "import torch;print(torch.backends.cuda.matmul.allow_tf32)"`_ |

**为什么必须包含 `sm_120`**：CUDA 12.8 才引入 Blackwell 消费级架构。
更早的构建（如 cu126）不含 sm_120 kernel，症状是 `no kernel image is available
for execution on the device`——而 `torch.cuda.is_available()` 仍会返回 `True`，
所以只检查可用性是查不出来的。

## 3. 基准测试

**方法**（`tools/bench.py`）：
warmup 10 次 → 50 次迭代 → `torch.cuda.Event` 计时 → `synchronize()` 后取均值。

**用例**：4096×4096 FP32 矩阵乘，FLOPs = 2×4096³ ≈ 1.374×10¹¹

**运行条件**（离开这些条件数字不成立）：
已接通电源；Windows 电源模式 = **最佳性能**；无其他 GPU 占用；_室温待填_。

### 最终结果

| 设备 | 耗时 | 算力 |
|---|---|---|
| CPU | 134.24 ms | 1.02 TFLOPS |
| GPU | **12.79 ms** | **10.75 TFLOPS** |
| 倍数 | **10.5×** | — |

GPU 达到 FP32 峰值的约 67%，属 cuBLAS SGEMM 的正常健康水平，无需继续调优。

### 历史记录（保留，用于说明数字为什么会变）

| 条件 | GPU 耗时 | 算力 | 倍数 | 偏差原因 |
|---|---|---|---|---|
| 无 warmup、单次迭代 | 156.8 ms | — | 2.0× | 一次性开销（cuBLAS handle、kernel 选择、显存池）被计入，**偏差 7.5 倍** |
| 有 warmup，电源模式未调 | 21.01 ms | 6.54 TFLOPS | 11.8× | 笔记本功耗墙压低 CPU 与 GPU 频率 |
| **有 warmup + 最佳性能** | **12.79 ms** | **10.75 TFLOPS** | **10.5×** | 定稿 |

> **注意倍数与绝对值的反向关系**：绝对性能从 21.01 ms 提升到 12.79 ms（快 64%），
> 但"倍数"却从 11.8× **降到** 10.5×——因为 CPU 侧同样被电源模式抬快了。
> 结论见 `docs/decisions.md` 第 2 条：**倍数指标是比率，必须同时报告两侧绝对值与运行条件。**

## 4. 定位说明

本机是**训练环境**，不是部署环境。命题限制的是**端侧推理部署平台**必须 100% 国产化，
训练阶段使用 NVIDIA GPU 不违反该要求。部署链路在 WSL2 / 国产 Linux 上验证。
