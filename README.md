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
| D2 | 数据集选型与最小可用数据集 | ✅ 已完成（2026-09-25） |
| D3–D4 | 基线检测器训练 | ✅ 已完成（2026-09-25） |
| D5 | 导出 ONNX | ✅ 已完成（2026-09-25） |
| D6 | ONNX Runtime 推理 + 与 PyTorch 精度对齐 | ☐ |
| D7 | 竖线收口：README + 演示录屏 | ☐ |

**D1–D7 的验收标准一句话**：
> 一张图 → 模型出框 → 导出 ONNX → 用 ONNX Runtime 推理 → 框与 PyTorch 结果一致。

## 数据集（D2）

两个源、都跑通了统一接入，来源与许可见 [`data/DATA_SOURCES.md`](data/DATA_SOURCES.md)：

| 源 | 类型 | 规模 | 类别 | 实测标注框 |
|---|---|---|---|---|
| NWPU VHR-10 | 光学 | 800 图（650 正 + 150 背景）| 3 类（airplane / ship / vehicle）| 1657 |
| SSDD | SAR | 1160 图 | 1 类（ship）| **2587**（与官方声明一致）|

> ⚠️ **三项已知代价，别当成没发生**（详见 [`docs/decisions.md`](docs/decisions.md)）：
> 1. 3 类只覆盖 NWPU **230/650** 张正样本，其余 420 张被滤空；
>    已决策 `drop_emptied: true` **丢弃**它们——那些图画面里有目标，
>    当背景用等于给检测器灌标签噪声。训练集因此为 **380 张**（230 正 + 150 真背景）。
> 2. NWPU VHR-10 为 **CC-BY-NC-4.0（非商业）**，企业交付前必须处置。
> 3. `onnx` 与 `onnxruntime` 的 **IR version 上限不同步**（实测 onnx 1.23 写 IR 14、
>    ORT 1.23 只支持到 11）——导出侧必须显式压低，否则 D5 的产物加载不了。

## 训练（D3–D4）

```powershell
# 冒烟：几十秒验证"数据 -> 模型 -> loss -> 出图"整条链路（结果无意义）
.\.venv\Scripts\python.exe scripts\train.py --smoke

# 正式基线（FP32，配置见 configs/train_baseline.yaml）
.\.venv\Scripts\python.exe scripts\train.py
```

- **模型**：FCOS / ResNet50-FPN + COCO 预训练，3 类，**32.1 M 参数**（这是 D8+ 压缩率的分母）
- **为什么是 FCOS**：无锚框、无 RoIAlign → D6 的"逐框对齐"变量最少、ONNX 导出最干净
- **产物**（`artifacts/train_baseline/`，不进 Git）：权重、loss 历史、`loss_curve.png`、`predictions.png`
  权重的 SHA256 与完整配方登记在 [`artifacts/README.md`](artifacts/README.md)

### 基线结果（2026-09-25）

| 项 | 值 |
|---|---|
| 训练耗时 | **11.4 min**（20 轮，平均 34.3 s/轮，RTX 5060 Laptop）|
| train total loss | 2.7300 → **0.7311** |
| val total loss | 4.0773 → **0.6278**（最低 0.6183）|
| 验证集 IoU@0.5 贪心匹配 | 召回 **0.978** / 精度 **0.952** |
| 纯背景图出框 | 31 张共 **1** 个（阈值 0.7 时为 **0** 个）|

验收物：`docs/img/20260925_d3_baseline_loss_curve.png`、`..._predictions.png`。

> ⚠️ **以上数字必须带两条限定一起引用**（详见 [`docs/decisions.md`](docs/decisions.md)）：
> 1. 那个 0.978 / 0.952 是**贪心 IoU@0.5**，**不是 mAP**；验证集只有 45 张含目标图，置信区间宽。
> 2. "随机划分把重复场景分到两侧"这条怀疑**已主动排查并证伪**（0/76 张 val 图与 train 相似度 ≥0.80），
>    但整图相关性查不出"同机场不同裁剪"，且该任务本身不难，数字偏高属预期。
>
> 另：NWPU 无官方划分，本项目自定固定种子划分，**mAP 不能与文献数字直接比较**，只作内部对照。

## 导出 ONNX（D5）

```powershell
.\.venv\Scripts\python.exe scripts\export_onnx.py
```

产物成对出现：`artifacts/model.onnx`（计算图）+ `artifacts/model.onnx.spec.json`（导出规格，
**D6 必须用同一份**）。哈希登记在 [`artifacts/README.md`](artifacts/README.md)。

| 项 | 实测 |
|---|---|
| 文件 | 128.6 MB，1040 节点 / 1492 条边，17 种算子 |
| IR / opset | **IR 8** / opset 17（IR 显式压低，见下）|
| 图合法性 | `onnx.checker.check_model(full_check=True)` **通过** |
| onnxruntime 加载 | 1.23.2 可加载 |
| 动态 H/W | **32×32 ~ 600×800 全区间**与 PyTorch 一致，最大绝对差 **8.583e-06**；误差不随尺寸缩小而增大 |
| 可复现性 | **导出字节级可复现**（两次导出 SHA256 相同）；对照之下**训练不可复现** |

看计算图：`netron artifacts\model.onnx`（或 https://netron.app）。

> ⚠️ **三条必须一起说明的限定**（详见 [`docs/decisions.md`](docs/decisions.md)）：
> 1. **图只含 backbone+FPN+head，不含后处理**（anchor / 解码 / NMS 都在图外）。
>    它**不是**"喂图直接吐框"的成品——把它写成"检测模型已部署"就是不实陈述。
>    这样切是为了 D6：两侧共用同一套后处理，差异才只可能来自模型本身。
> 2. 用的是 legacy（`dynamo=False`）导出器——torch 已明确警告**将被移除**
>    （本机默认的 dynamo 导出器需要 `onnxscript`，环境未装）。升级 torch 时须重验导出。
> 3. **IR version 必须显式压低**：实测 onnx 1.23 默认写 IR 14，而 onnxruntime 1.23.2
>    只支持到 11，直接报 `Unsupported model IR version`。
>
> 📌 **上板待查算子**：FCOS 的 GroupNorm 导出为 **`InstanceNormalization`（40 个）**，
> 需确认昇腾 CANN 是否支持；不支持则换 RetinaNet（头是 Conv+ReLU，本仓库已支持切换）。

## 环境

本地为**训练环境**，不是部署环境。两者的区别见 [`docs/decisions.md`](docs/decisions.md)。

- **硬件**：NVIDIA GeForce RTX 5060 Laptop GPU（8 GB，sm_120），驱动 592.01
- **软件**：Windows 11（10.0.26200）/ Python 3.10.11（venv）/ torch 2.14.0+cu130 / CUDA 13.0
- **详细环境清单与基准数据**：见 [`docs/env.md`](docs/env.md)

> ⚠️ Windows 只用于训练。**部署链路（ONNX 导出后的部分）在 WSL2 / 国产 Linux 上练**——
> 目标平台是 openEuler / 银河麒麟，且昇腾 CANN 工具链仅有 Linux 版本。

### 部署练习环境（WSL2，已就绪）

**Ubuntu 24.04 LTS / Python 3.12**，一键搭建与验证：

```powershell
# 一键准备 Linux 侧环境（装 venv + numpy/onnx/onnxruntime，不装 torch）
wsl -d Ubuntu-24.04 -u root -- bash tools/setup_wsl.sh

# 后端冒烟测试：应输出 PASS 且 has_nvidia_components 为 false
wsl -d Ubuntu-24.04 -u legion -- /home/legion/.venvs/lingmou/bin/python tools/backend_smoke.py
```

实测：同一模型在 Windows（ORT 1.23.2）与 WSL Ubuntu 24.04（ORT 1.30.0）上
**最大绝对误差均为 0.000e+00**，`has_nvidia_components = false`。

> ⚠️ **WSL ≠ 板子**：WSL2 是 `x86_64 / Ubuntu`，板子是 `aarch64 / openEuler + 昇腾`。
> 上板后必须用**同一条命令**重跑一次，那份结果才是《国产化适配测试报告》的正式数据。

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

# 4. 基准测试（应插电源 + 电源模式设为"最佳性能"）
.\.venv\Scripts\python.exe tools\bench.py

# 5. 拉数据集（约 126 MB，幂等；数据不进 Git，见 data/DATA_SOURCES.md）
.\.venv\Scripts\python.exe tools\fetch_datasets.py

# 6. 出数据集统计与验收图 -> artifacts/d2/
.\.venv\Scripts\python.exe scripts\preview_data.py

# 7. 跑测试
.\.venv\Scripts\python.exe -m pytest -q
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
| [`data/DATA_SOURCES.md`](data/DATA_SOURCES.md) | 数据来源、许可条款、SHA256 与实测统计（**数据集本体不进 Git**）|
