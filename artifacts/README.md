# artifacts/ —— 模型与部署产物（**不进 Git**）

## 为什么不用 Git 存权重

权重与 ONNX/OM 文件是大二进制，Git 无法有效增量存储，且一旦提交就永久留在历史里。
**文件走共享盘，本文件记录可追溯信息。**

## 登记表（每产出一个产物就加一行）

| 日期 | 产物 | 来源模型/配置 | 文件大小 | SHA256 |
|---|---|---|---|---|
| 2026-09-25 | `train_baseline_run1/best.pth` | FCOS-R50-FPN + COCO 预训练，3 类；`configs/train_baseline.yaml`（20 轮 / bs4 / lr0.005 / min_size600）| 128,799,730 B | `4576d830445f9671aca63624689f9685031c246908fe7d45779caa5a444d46b6` |
| 2026-09-25 | `model.onnx` | 由上一行权重导出（D5）；opset 17 / IR 8 / 动态 H/W；规格见 `model.onnx.spec.json` | 128,586,408 B | `8a8f5b36fd9deb2bdc6fba6c6945b9e52ac73562fb5def8c82e4529a8386a58a` |

> **`run1` 为什么要单独归档**：训练**不是确定性的**（未固定随机种子、cuDNN 算法选择有随机性），
> 重跑同一份配置**不会**得到同一个 SHA256。所以每次正式训练的产物单独归档一份，
> 否则登记表里的哈希会被下一次运行覆盖成"查无此物"。
> 重跑后请**另起一行**登记新哈希，而不是改旧行。

> **怎么用**：拿到任何一份 `.pth` / `.onnx` / `.om`，先算 SHA256 与本表比对，
> 就能确认"它到底是不是那次训练出来的"。核对不上就是另一个东西，不要猜。

配方摘要（复现这次基线所需的全部信息）：
- 数据：NWPU VHR-10，3 类（airplane / ship / vehicle），`drop_emptied: true`
  → 380 张，划分后 train 304 / val 76，`seed=20260925`
- 模型：`torchvision` FCOS + ResNet50-FPN，COCO 预训练后替换检测头，**32.1 M 参数**
- 训练：20 轮，实测 **686.9 s（11.4 min）**，平均 34.3 s/轮，RTX 5060 Laptop
- 结果：train total 2.7300 → **0.7311**；val total 4.0773 → **0.6278**（最低 0.6183）
- 验收物副本：`docs/img/20260925_d3_baseline_loss_curve.png`、`..._predictions.png`

### 配套文件（同一目录）

| 文件 | 说明 |
|---|---|
| `history.json` | 逐 epoch 的 loss **分量**（classification / bbox_regression / bbox_ctrness）+ lr + 耗时 |
| `loss_curve.png` | loss 曲线，D3–D4 验收物 |
| `predictions.png` | 标注（绿细线）vs 预测（红粗线）对照图，D3–D4 验收物 |
| `train.log` | 完整训练日志 |

```powershell
# 算 SHA256
Get-FileHash .\artifacts\train_baseline\best.pth -Algorithm SHA256
```

## 为什么必须记哈希

评委问"这个 `.om` 文件是哪个模型、哪次训练转出来的"——**报得出哈希、对得上记录**，
这就是可追溯。这也是《国产化适配测试报告》里"转换日志与产物"那一节要的东西。
