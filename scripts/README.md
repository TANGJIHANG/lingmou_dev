# scripts/ —— 入口脚本

## 约定

这里只放**薄入口**：解析参数 → 调 `src/lingmou/` 里的实现 → 输出结果。
**业务逻辑不要写在这里**，否则无法被测试和复用。

## 规划中的脚本

| 脚本 | 作用 | 状态 |
|---|---|---|
| `preview_data.py` | 数据集预览：统计 + 带框网格图（D2 验收证据）| ✅ D2 |
| `train.py` | 训练检测器（读 `configs/train_*.yaml`） | ✅ D3–D4 完成 |
| `export_onnx.py` | 导出 ONNX | ✅ D5 |
| `eval.py` | 评测：召回/精度/虚警 + 两组对照（内置**污染核查**）| ✅ D3–D4 消融 / D6 基础；mAP 待补 |
| `infer.py` | 单图/单流推理入口 | ☐ D6 |
| `infer_bench.py` | 分段耗时打点（解码/预处理/推理/后处理） | ☐ D11 |

## 训练

```powershell
# 冒烟：1 轮 × 3 次迭代，几十秒验证"数据 -> 模型 -> loss -> 出图"整条链路通不通
# 结果落在 artifacts/train_baseline_smoke/，**数字无意义**，只看链路
.\.venv\Scripts\python.exe scripts\train.py --smoke

# 正式基线（FP32，配置见 configs/train_baseline.yaml）
.\.venv\Scripts\python.exe scripts\train.py

# 临时少跑几轮，不改配置文件
.\.venv\Scripts\python.exe scripts\train.py --epochs 5
```

### 消融实验：用命令行覆盖，不要去改 YAML

```powershell
# 反例（危险）：验证 drop_emptied 的影响
.\.venv\Scripts\python.exe scripts\train.py --drop-emptied false --output-dir artifacts/ablation_fake_bg
```

`--drop-emptied` 只在**内存里**覆盖 `configs/datasets.yaml` 的同名项，
配置文件不动，两次实验的输出目录也不同、不会互相覆盖。

> ⚠️ **不要用 PowerShell 就地改写配置文件**。下面这种写法是本项目实际踩过的坑：
>
> ```powershell
> # 千万不要这样写 —— 会把文件清成 0 字节
> (Get-Content configs\datasets.yaml) -replace 'true', 'false' | Set-Content configs\datasets.yaml
> ```
>
> 管道里 `Set-Content` **先**打开并截断文件，`Get-Content` **后**才去读，
> 于是读到空、写回空，配置当场丢失（本项目已发生，靠 `git checkout` 救回）。
> 要改配置就用编辑器；要跑变体就用上面的 CLI 参数。

产物（默认 `artifacts/train_baseline/`，**不进 Git**，登记见 `artifacts/README.md`）：

| 文件 | 用途 |
|---|---|
| `best.pth` | 权重（内含 config 与 loss 历史，便于追溯）|
| `history.json` | 逐 epoch 的 loss **分量**——只看总 loss 无法判断卡在哪一路 |
| `loss_curve.png` | D3–D4 验收物 |
| `predictions.png` | 标注（绿细线）vs 预测（红粗线），D3–D4 验收物 |

其它可用覆盖参数：`--epochs` / `--batch-size` / `--device` / `--config` / `--datasets`。
**训练不是确定性的**（未固定随机种子），重跑同一份配置不会得到同一个 SHA256——
所以每次正式训练都单独归档，并在 `artifacts/README.md` **另起一行**登记新哈希。

## 导出 ONNX（D5）

```powershell
# 从基线权重导出（默认读 configs/train_baseline.yaml 的检测器/类别/尺寸）
.\.venv\Scripts\python.exe scripts\export_onnx.py

# 指定权重与输出
.\.venv\Scripts\python.exe scripts\export_onnx.py `
    --checkpoint artifacts\train_baseline_run1\best.pth --output artifacts\model.onnx

# 定尺寸图（默认是动态 H/W）
.\.venv\Scripts\python.exe scripts\export_onnx.py --fixed-hw 600x800
```

产物两个，**必须成对使用**：

| 文件 | 说明 |
|---|---|
| `model.onnx` | 计算图 |
| `model.onnx.spec.json` | 导出规格（opset / IR / 动态轴 / 输入尺寸）——**D6 必须用同一份** |

> ⚠️ **导出的图只含 backbone+FPN+head，不含后处理**（anchor/解码/NMS 都在图外）。
> 它不是"喂图片直接吐框"的成品；逐框对齐属 D6。

看计算图：装 `netron` 后 `netron artifacts\model.onnx`，或在 https://netron.app 打开。

## 数据准备类脚本

```powershell
# 下载数据集（约 126 MB，幂等）——工具放 tools/，因为它是开发环境工具而非产品入口
.\.venv\Scripts\python.exe tools\fetch_datasets.py

# 出统计与验收图（默认落 artifacts/d2/）
.\.venv\Scripts\python.exe scripts\preview_data.py
```

## 分段耗时打点是重点

命题的 200 ms 是**全链路端到端**指标，必须能拆开看每一段花了多少。
实现时统一用 `tools/bench.py` 的计时方法（warmup + Event + synchronize），
理由见 `docs/decisions.md` 第 2 条。
