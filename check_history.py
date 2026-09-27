import json
from pathlib import Path

files = [
    "artifacts/train_baseline_run1/history.json",
    "artifacts/train_baseline_ablation_fake_bg/history.json",
]

for f in files:
    hist = json.loads(Path(f).read_text(encoding="utf-8"))
    epochs = hist["epochs"]

    last = epochs[-1]
    best = min(epochs, key=lambda e: e["val"]["total"])

    print(f)
    print(f"  last: epoch={last['epoch']:2d}  train={last['train']['total']:.4f}  val={last['val']['total']:.4f}")
    print(f"  best: epoch={best['epoch']:2d}  train={best['train']['total']:.4f}  val={best['val']['total']:.4f}")