# 灵眸天衡 · 空天智能解译载荷

面向"空天智能解译载荷"企业命题的**软硬件一体化国产化智能解译系统**。

## 一句话目标

一件能自己看图、自己认目标、自己报位置的国产智能载荷：

> 光学 / SAR / 红外多源融合 → 国产 AI 芯片端侧推理 → 端到端 **≤200 ms** 输出目标类别与位置态势

## 架构断言

本项目全部工程决策围绕一条主线：

> **换芯片 = 换执行后端（Execution Provider）**

模型格式（ONNX）与芯片解耦，跨芯片适配靠替换推理后端，而非重写推理链路。
该断言的代码落点是 [`src/lingmou/runtime/`](src/lingmou/runtime/)。

## 当前状态

| 阶段 | 内容 | 状态 |
|---|---|---|
| D0 | 训练环境打通（RTX 5060 / torch 2.14.0+cu130 / sm_120） | ✅ 已完成 |
| D1 | 仓库骨架、目录分层、runtime 适配层接口 | ✅ 已完成 |
| D2 | 数据集选型与最小可用数据集 | ⏳ 进行中 |
| D3–D4 | 基线检测器训练 | ☐ |
| D5 | 导出 ONNX | ☐ |
| D6 | ONNX Runtime 推理 + 与 PyTorch 精度对齐 | ☐ |
| D7 | 竖线收口：README + 演示录屏 | ☐ |

**D1–D7 的验收标准一句话**：
> 一张图 → 模型出框 → 导出 ONNX → 用 ONNX Runtime 推理 → 框与 PyTorch 结果一致。

## 环境

本地为**训练环境**，不是部署环境。两者的区别见 [`docs/decisions.md`](docs/decisions.md)。

- **硬件**：NVIDIA GeForce RTX 5060 Laptop GPU（8 GB，sm_120），驱动 592.01
- **软件**：Windows / Python 3.10.11（venv）/ torch 2.14.0+cu130 / CUDA 13.0
- **详细环境清单与基准数据**：见 [`docs/env.md`](docs/env.md)

> ⚠️ Windows 只用于训练。**部署链路（ONNX 导出后的部分）在 WSL2 / 国产 Linux 上练**——
> 目标平台是 openEuler / 银河麒麟，且昇腾 CANN 工具链仅有 Linux 版本。

## 目录结构

目录分层与申报书中宣称的系统架构一一对应，便于逐层单独演示与答疑。

```
src/lingmou/
├─ io/        数据接入层 —— 光学/红外/SAR 统一读取 + 时间戳与元数据
├─ tasks/     任务管理层 —— 任务队列、优先级、多模型调度
├─ engine/    解译引擎层 —— 检测 / 分割 / 精定位 / 多源融合 / 跟踪
├─ runtime/   国产算力适配层 —— 统一推理接口 → CPU/CUDA/CANN/RKNN 后端可插拔
├─ output/    输出层 —— 标准化态势报文（JSON）
└─ viz/       可视化 —— 画框、渲染、结果导出
```

**依赖方向（不得违反）**：`io / tasks / engine / output / viz` → `runtime` → 具体后端。
**业务代码永远不要直接 `import onnxruntime`**，只依赖 `runtime.base.InferenceBackend`。

## 快速开始

```powershell
# 1. 建环境（CUDA 版 torch 必须走 PyTorch 官方索引，否则装成 CPU 版）
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. 以可编辑模式安装本包，之后才能 import lingmou
.\.venv\Scripts\python.exe -m pip install -e .

# 3. 验证环境（应输出 PASS 且 arch list 含 sm_120）
.\.venv\Scripts\python.exe tools\env_check.py

# 3. 基准测试（应插电源 + 电源模式设为"最佳性能"）
.\.venv\Scripts\python.exe tools\bench.py
```

## 开发纪律

1. **先跑通，再优化；先基线，再对比。** 没有 FP32 基线，任何"优化后"的数字都无意义。
2. **所有指标都要有口径。** 报倍数必须同时报分子绝对值、分母绝对值和完整运行条件。
3. **每次优化留消融记录**，写进 `docs/decisions.md`。
4. **不提交数据集与模型权重**，大文件走共享盘 + SHA256 记录。
5. **commit 粒度 = 一件事**，写清 scope。`git log` 是"我们真的在写代码"的证据。
6. **为部署而设计，不要训完再想办法部署。** 写网络前先查算子支持表。

## 文档索引

| 文件 | 作用 |
|---|---|
| [`docs/env.md`](docs/env.md) | 训练环境清单与基准数据（将来进《测试评测报告》环境章节） |
| [`docs/decisions.md`](docs/decisions.md) | 技术决策记录：决定 / 理由 / 被放弃的备选 |
| [`docs/roadmap.md`](docs/roadmap.md) | D0–D7 竖线打通计划与验收标准 |
