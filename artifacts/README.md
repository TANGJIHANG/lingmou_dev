# artifacts/ —— 模型与部署产物（**不进 Git**）

## 为什么不用 Git 存权重

权重与 ONNX/OM 文件是大二进制，Git 无法有效增量存储，且一旦提交就永久留在历史里。
**文件走共享盘，本文件记录可追溯信息。**

## 登记表（每产出一个产物就加一行）

| 日期 | 产物 | 来源模型/配置 | 文件大小 | SHA256 |
|---|---|---|---|---|
| 2026-09-25 | `train_baseline_run1/best.pth` | FCOS-R50-FPN + COCO 预训练，3 类；`configs/train_baseline.yaml`（20 轮 / bs4 / lr0.005 / min_size600）；**`drop_emptied=true`** | 128,799,730 B | `4576d830445f9671aca63624689f9685031c246908fe7d45779caa5a444d46b6` |
| 2026-09-26 | `train_baseline_ablation_fake_bg/best.pth` | ⚠️ **目录名有误导性：它实际是 `drop_emptied=true` 的run**（按每轮 34.6 s ≈ A 的 34.3 s 判定，非目录名）| 128,799,730 B | `3475ddd6c4e6505e583dd585bde5b2cd11d4efe99667a6b5d18822017a583f16` |
| 2026-09-27 | `ablation_fake_bg/best.pth` | **`drop_emptied=false`**（按每轮 63.4 s ≈ 2× 判定，且 train loss 末值 0.5790 显著最低，梯度步数多 2.1 倍）| 128,799,730 B | `c363124506a39e88f20df90a77c0dfc8a15b9cd2cbdba94a3adf313e5b97be0c` |
| 2026-09-27 | `train_baseline_repeat/best.pth` | `drop_emptied=true`（**权重内 `dataset_spec` 已自记录**，第一个自描述的产物）| 128,800,306 B | `a1c48206c7ce37a63b702e55a759eb03892131d53841c8698fc7b377c0a36974` |
| 2026-09-25 | `model.onnx` | 由 `train_baseline_run1/best.pth` 导出（D5）；opset 17 / IR 8 / 动态 H/W；规格见 `model.onnx.spec.json` | 128,586,408 B | `8a8f5b36fd9deb2bdc6fba6c6945b9e52ac73562fb5def8c82e4529a8386a58a` |

> 🛑 **上面第二行为什么要加那句警告**：`best.pth` 当时**没有记录 `dataset_spec`**，
> 只能靠目录名判断它属于哪个实验条件——而**目录名是错的**。
> 本项目的消融分析因此一度用错了权重、得出了反向结论（详见 `docs/decisions.md` 的重大更正）。
>
> **规则（从这次事故得出）**：
> 1. **产物的身份必须由产物自身携带**（`dataset_spec`），不能由文件名/目录名承载；
> 2. 无法自证的旧产物，**不要凭命名去推断**——用可测量的物理量（如每轮耗时）交叉验证，
>    或者直接标注"归属不明"；
> 3. **每产出一个产物立刻登记**。本次三个权重是在事后核查时才补登的，
>    正是这段空窗让错误结论跑了出去。

> **`run1` 为什么要单独归档**：训练**不是确定性的**（未固定随机种子、cuDNN 算法选择有随机性），
> 重跑同一份配置**不会**得到同一个 SHA256。所以每次正式训练的产物单独归档一份，
> 否则登记表里的哈希会被下一次运行覆盖成"查无此物"。
> 重跑后请**另起一行**登记新哈希，而不是改旧行。

> **怎么用这张表 —— 两种产物，两种语义，别混**
>
> | 产物 | 哈希能证明什么 | 能否重建 |
> |---|---|---|
> | `.pth`（训练权重）| 只能证明"**确实产出过这个文件**"（防篡改）| ❌ **训练不可复现**，重训得到另一组权重 |
> | `.onnx`（导出产物）| 可**核对"是不是同一次导出"** | ✅ **导出字节级可复现**，一条命令重建 |
>
> 实测依据：连续两次导出，SHA256 完全相同（`8a8f5b36…`），且与登记值一致；
> 而训练未固定种子，重跑必得不同哈希。
>
> ⚠️ **不要拿 `.pth` 的哈希当"可复现性证明"**——那只能说明文件没被改过，
> 不能说明"换台机器重跑能得到同样结果"。报告里这么写就是**过度声明**。
>
> 📌 由此得出的保管策略：**`best.pth` 是唯一丢了就拿不回来的产物，必须单独备份**
> （共享盘 / Release 附件）。`.onnx`、曲线图、预测图都可重建，不必备份。

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
