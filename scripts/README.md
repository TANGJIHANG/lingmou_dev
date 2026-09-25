# scripts/ —— 入口脚本

## 约定

这里只放**薄入口**：解析参数 → 调 `src/lingmou/` 里的实现 → 输出结果。
**业务逻辑不要写在这里**，否则无法被测试和复用。

## 规划中的脚本

| 脚本 | 作用 | 状态 |
|---|---|---|
| `preview_data.py` | 数据集预览：统计 + 带框网格图（D2 验收证据）| ✅ D2 |
| `train.py` | 训练检测器（读 `configs/train_*.yaml`） | ☐ D3–D4 |
| `export_onnx.py` | 导出 ONNX | ☐ D5 |
| `eval.py` | 评测：精度、虚警率、时延 | ☐ D6+ |
| `infer.py` | 单图/单流推理入口 | ☐ D6 |
| `infer_bench.py` | 分段耗时打点（解码/预处理/推理/后处理） | ☐ D11 |

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
