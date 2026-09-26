#!/usr/bin/env bash
# 在 WSL2 / Linux 上准备"部署链路练习环境"（D5–D6 前置）。
#
# 为什么需要这个脚本
# ------------------
# roadmap 规定 D5 起（ONNX -> ORT 推理）在 Linux 上练，因为昇腾 CANN 工具链
# （含 ATC 转换器）**只有 Linux 版本**。而 WSL 重装、换机器、最终上板时，
# 都要把这套环境重新搭一遍——写死成人可复制的脚本，比"我记得当时装了什么"可靠。
#
# 用法（在项目根目录）
# --------------------
#     wsl -d Ubuntu-24.04 -u root -- bash tools/setup_wsl.sh
#     wsl -d Ubuntu-24.04 -u root -- bash tools/setup_wsl.sh <linux用户名>
#
# 注意
# ----
# 1. 虚拟环境建在 Linux 文件系统（~/.venvs）里，**不建在 /mnt/c**：
#    DrvFs 上建 venv 又慢又容易因权限位问题出错。
# 2. 本脚本只装冒烟测试所需的最小依赖（numpy/onnx/onnxruntime），**不装 torch**：
#    Windows 才是训练环境，Linux 侧只练"导出后的那段链路"。
# 3. WSL 若不继承 Windows 代理，`pip` 走直连即可（PyPI 通常可达）；
#    GitHub 不通不影响本脚本。

set -euo pipefail

TARGET_USER="${1:-$(ls /home 2>/dev/null | head -1)}"
if [ -z "${TARGET_USER:-}" ] || [ ! -d "/home/$TARGET_USER" ]; then
  echo "找不到目标用户，请显式传入：bash tools/setup_wsl.sh <用户名>" >&2
  exit 2
fi
TARGET_HOME="/home/$TARGET_USER"
VENV="$TARGET_HOME/.venvs/lingmou"

echo "=== 目标用户: $TARGET_USER  虚拟环境: $VENV ==="

if [ "$(id -u)" -eq 0 ]; then
  export DEBIAN_FRONTEND=noninteractive
  echo "--- 安装系统包 python3-venv / python3-pip ---"
  apt-get update -qq
  apt-get install -y -qq python3-venv python3-pip
fi

echo "--- 创建虚拟环境 ---"
if [ ! -x "$VENV/bin/python" ]; then
  if [ "$(id -u)" -eq 0 ]; then
    runuser -u "$TARGET_USER" -- python3 -m venv "$VENV"
  else
    python3 -m venv "$VENV"
  fi
else
  echo "已存在，跳过"
fi

echo "--- 安装依赖（最小集，不含 torch）---"
if [ "$(id -u)" -eq 0 ]; then
  runuser -u "$TARGET_USER" -- "$VENV/bin/python" -m pip install -q --upgrade pip
  runuser -u "$TARGET_USER" -- "$VENV/bin/python" -m pip install -q numpy onnx onnxruntime
else
  "$VENV/bin/python" -m pip install -q --upgrade pip
  "$VENV/bin/python" -m pip install -q numpy onnx onnxruntime
fi

echo
echo "--- 环境自述 ---"
"$VENV/bin/python" - <<'PYEOF'
import platform, sys
import numpy, onnx, onnxruntime as ort
print("platform   :", platform.platform())
print("python     :", sys.version.split()[0])
print("numpy      :", numpy.__version__)
print("onnx       :", onnx.__version__)
print("onnxruntime:", ort.__version__)
print("providers  :", ort.get_available_providers())
PYEOF

cat <<EOF

=== 完成 ===
跑冒烟测试：

    wsl -d Ubuntu-24.04 -u $TARGET_USER -- $VENV/bin/python tools/backend_smoke.py
EOF
