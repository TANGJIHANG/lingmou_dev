# 训练与基准环境清单

> 本文件是《测试评测报告》"环境清单"章节的原始数据来源。
> **数字必须与它成立的条件绑在一起**，否则不可复现。

## 1. 硬件

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，8 GB，计算能力 **sm_120** |
| GPU 驱动 | 592.01 |
| CPU | AMD Ryzen 9 8945HX with Radeon Graphics |
| 内存 | 15.3 GB |
| 存储 | C 盘可用约 164 GB（2026-09-25 实测；建 venv 时约 206 GB） |

## 2. 软件

| 项 | 值 |
|---|---|
| OS | Microsoft Windows 11 家庭版 中文版（10.0.26200）|
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

---

## 5. 部署链路练习环境（WSL2 / Linux）

> D5 起（ONNX → ONNX Runtime 推理）在 Linux 上练，原因见 `docs/decisions.md`
> ——**昇腾 CANN 工具链（含 ATC 转换器）仅有 Linux 版本**。

### 5.1 实测环境（2026-09-25）

| 项 | 值 |
|---|---|
| Windows 侧 WSL 版本 | **2.7.14.0**（内核 6.18.33.2-microsoft-standard-WSL2，WSLg 1.0.73.2）|
| 发行版 | **Ubuntu 24.04.5 LTS**（`x86_64`，默认 WSL 版本 2）|
| Linux Python | 3.12.3 |
| 虚拟环境 | `~/.venvs/lingmou`（**建在 Linux 文件系统，不在 /mnt/c**）|
| numpy / onnx | 2.5.3 / 1.23.0 |
| onnxruntime | **1.30.0** |
| 可用 providers | `AzureExecutionProvider`, `CPUExecutionProvider` |

一键复现：`wsl -d Ubuntu-24.04 -u root -- bash tools/setup_wsl.sh`

### 5.2 后端冒烟测试结果

`tools/backend_smoke.py` —— hello-world 级验证：一个 `Y = X @ W + b` 的最小 ONNX 图，
用 numpy 现算参考值（**刻意不依赖 torch**，板子上未必装得动 PyTorch）。

| 环境 | Python | onnxruntime | 最大绝对误差 | 含 NVIDIA 组件 |
|---|---|---|---|---|
| Windows（训练机）| 3.10.11 | 1.23.2 | **0.000e+00** | `false` |
| WSL Ubuntu 24.04 | 3.12.3 | **1.30.0** | **0.000e+00** | `false` |

**这组数字的意义**：
1. 同一模型在两个 **onnxruntime 版本不同**（1.23.2 / 1.30.0）、
   **Python 版本不同**（3.10 / 3.12）、**操作系统不同**的环境上得到**逐位一致**的结果，
   说明该算子的数值路径在这两个版本间稳定。
2. `has_nvidia_components = false` 是命题**国产化自证**要的字段，
   将来换 `--providers CANNExecutionProvider` 后此字段仍须为 `false`。

> ⚠️ **WSL 不等于板子**：WSL2 是 `x86_64 / Ubuntu`，板子是 `aarch64 / openEuler + 昇腾`。
> 上面这张表说明的是"链路能通"，**不能**用来推断板子上也能通。
> 板子到手后必须用**同一条命令**重跑一次，那份结果才是《国产化适配测试报告》的正式数据。

### 5.3 WSL 网络注意事项（实测）

| 目标 | WSL 内直连 | 说明 |
|---|---|---|
| `archive.ubuntu.com` / `security.ubuntu.com` | ✅ 200 | `apt` 可用 |
| `pypi.org` / `bootstrap.pypa.io` | ✅ 200 | `pip` 可用 |
| `github.com` | ❌ 超时 | 仅在 `git clone` 时需要 |

Windows 主机挂着本地代理 `127.0.0.1:7897`，WSL 在 **NAT 模式**下访问不到它，
启动时会提示 *"检测到 localhost 代理配置，但未镜像到 WSL"*。
**处置见 `docs/decisions.md`（不为此关闭主机代理）**；若确需 WSL 访问 GitHub，
应改用 `networkingMode=mirrored`，而不是关闭代理。

> **2026-09-25 当日晚些时候的复测**：主机代理已关闭，此时**直连 `github.com` 反而可用**
> （此前直连超时、须经代理）。说明上表的 ✅/❌ **只是某一时刻的网络状态快照**，
> 会随代理开关、节点选择、TUN 模式变化。需要时**重新实测**，不要当成永久结论。
> `git push` 当前**无需任何代理参数**；也不要为此写全局 git 代理配置——
> 代理一关，那条配置会让推送持续失败。
