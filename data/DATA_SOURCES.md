# data/DATA_SOURCES.md —— 数据来源与许可登记

> 数据集本体**不进 Git**（见 `data/README.md`），但这份登记表必须进 Git：
> 答辩引用公开数据集时，评委可能追问"哪来的、能不能用"，
> 而**只报得出名字、报不出哈希与许可**是最容易被问倒的地方。
>
> 本表由 `tools/fetch_datasets.py` 的产物 `data/raw/FETCH_MANIFEST.json` 提供哈希依据，
> 统计数字由 `.\\.venv\\Scripts\\python.exe scripts\\preview_data.py` 复现。

**下载日期**：2026-09-25
**网络环境实测**：`huggingface.co` 不可达，`hf-mirror.com` 可达 → 故一律走镜像；
原始出处仍以本节 `source_page` 为准。

---

## 1. NWPU VHR-10（光学）

| 项 | 值 |
|---|---|
| 原始出处 | 西北工业大学（Northwestern Polytechnical University）|
| 镜像页面 | https://huggingface.co/datasets/isaaccorley/vhr10 |
| 直链 | `https://hf-mirror.com/datasets/isaaccorley/vhr10/resolve/main/NWPU%20VHR-10%20dataset.zip` |
| 上游来源（镜像自述）| Google Drive `1--foZ3dV5OCsqXQXT84UeKtrAqc5CkAE`，与 https://github.com/chaozhong2010/VHR-10_dataset_coco |
| 镜像自述的改动 | **仅把 RAR 转成 ZIP，图像未修改** |
| 本地文件 | `data/raw/NWPU_VHR-10_dataset.zip`（76,822,614 B）|
| SHA256 | `3e8c0299bad6b5d2b4d4034095c3581f50a02bc0dcb97fca70f6ad739f7cbf53` |
| 解包位置 | `data/interim/nwpu_vhr10/` |
| 许可（镜像标注）| **CC-BY-NC-4.0 —— 非商业性使用** |
| 许可（原始数据集）| 研究用途；**未见于官方页面明确授予商业许可** |

**实测统计**（本仓库 `lingmou.io.read_nwpu_vhr10` 产出，可复现）

| 指标 | 数值 |
|---|---|
| 图像总数 | 800（650 正样本 + 150 纯背景）|
| 原始标注框总数（10 类）| 3896 |
| 本阶段采用类别数 | **3**（airplane / ship / vehicle）|
| 本阶段标注框数（3 类）| 1657（占全部框 42.5%）|

| 类别 | 框数 |
|---|---|
| airplane | 757 |
| storage_tank | 655 |
| vehicle | 598 |
| tennis_court | 524 |
| baseball_diamond | 390 |
| ship | 302 |
| harbor | 224 |
| ground_track_field | 163 |
| basketball_court | 159 |
| bridge | 124 |

---

## 2. NWPU VHR-10 第三方 COCO 标注（**仅交叉校验，不作为训练标注**）

| 项 | 值 |
|---|---|
| 页面 | https://huggingface.co/datasets/isaaccorley/vhr10 |
| 本地文件 | `data/raw/NWPU_VHR-10_annotations.json`（1,265,416 B）|
| SHA256 | `dde27d9362d9c6aa358a1cab160c9e075e57405d3fe876de049973c6e6150f0e` |
| 内容 | 650 图 / 3921 框 / 10 类（COCO 格式）|

**为什么下载它，又为什么不用它训练**：
本仓库自己实现的 GT 解析器解出 **3896 框**，与这份独立的 COCO 标注 **3921 框**相差 0.6%——
两个独立来源数量级吻合，说明解析器没有系统性错漏。
但抽查发现**个别框坐标不一致**（`001.jpg`：原始 GT 高 95 px，COCO 版高 86 px），
说明二者并非同一套标注。按项目纪律「所有指标都要有口径」，
**以数据集自带的原始 GT 为准**，COCO 版降级为交叉校验物。
决策记录见 `docs/decisions.md`。

---

## 3. SSDD —— SAR Ship Detection Dataset（SAR）

| 项 | 值 |
|---|---|
| 原始出处 | 电子科技大学；论文 Remote Sens. 13(18):3690, 2021 |
| 官方仓库 | https://github.com/TianwenZhang0825/Official-SSDD |
| 镜像页面 | https://huggingface.co/datasets/Gordy333/ssdd-rsdd-data |
| 直链 | `https://hf-mirror.com/datasets/Gordy333/ssdd-rsdd-data/resolve/main/SSDD_DOTA.tar.gz` |
| 本地文件 | `data/raw/SSDD_DOTA.tar.gz`（49,308,340 B）|
| SHA256 | `a51bcdf0b691e45d8ccd2c99a6a171966bfdceb6caea7a8ad22119591e89b413` |
| 解包位置 | `data/interim/ssdd/` |
| 许可（官方仓库）| **Apache-2.0** ✅ 已用 GitHub LICENSE 探测接口核实（`path=LICENSE`, `spdx=Apache-2.0`），非仅凭徽章转述 |

**实测统计**（本仓库 `lingmou.io.read_ssdd_dota` 产出，可复现）

| 指标 | 数值 |
|---|---|
| 图像总数 | 1160（train 928 + test 232）|
| 标注框总数 | **2587** —— 与论文/官方声明的 2587 **完全一致** |
| 类别数 | 1（ship）|
| 标注格式 | DOTA 八点旋转框 |

> 框数能对上官方数字，是 DOTA 解析器正确性的直接证据，已固化为
> `tests/test_io.py::test_real_ssdd_box_count_matches_official`。

---

## ⚠️ 许可风险（**答辩前必须处理，不要拖到交付**）

### 风险 1：NWPU VHR-10 是 **非商业（CC-BY-NC-4.0）**

- 比赛本身属研究/竞赛用途，一般可用；
- 但本轮命题是**企业命题**，若交付物（含训练权重）预期被命题企业**商用**，
  NWPU VHR-10 的训练产物存在**许可瑕疵**。
- **建议处置**：竖线打通阶段继续用（它只是工程验证）；
  进入交付/商用叙事时，主力数据换成许可更宽松的源（如 Apache-2.0 的 SAR 源、
  或自采/命题方提供的数据），并在《测试评测报告》中明确数据来源与许可边界。

### 风险 2：SSDD 影像含 **TerraSAR-X 成分**（二级权利不确定）

- SSDD 仓库自身是 Apache-2.0，但其影像合成自 RadarSat-2 / TerraSAR-X / Sentinel-1；
- **TerraSAR-X / TanDEM-X 数据由 DLR 按科研许可分发，一般不许可自由再分发衍生影像产品**；
- 公开信息无法判定哪几张图来自哪个传感器，SSDD 作者是否单独取得再分发授权亦不明。
- **建议处置**：与风险 1 同；引用时注明这一不确定性，不要声称"许可完全干净"。
  这条我们**主动写出来**——评委若自己发现，性质就从"如实说明"变成"隐瞒"。

### 待办（人为动作，脚本代不了）

- [ ] 对上述三个来源页面**截图留档**，存 `docs/img/`，文件名带日期
      （如 `20260925_license_nwpu_vhr10.png`）。`data/README.md` 明确要求"截图留档"。
- [ ] 若后续更换数据集，本文件必须同步更新，并保留旧条目而不是删除。

---

## 复现方式

```powershell
# 重新下载（幂等：已存在则跳过，--force 强制重下）
.\.venv\Scripts\python.exe tools\fetch_datasets.py

# 重新出统计与验收图
.\.venv\Scripts\python.exe scripts\preview_data.py
```
