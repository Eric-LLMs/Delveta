#!/usr/bin/env python3
"""Plot the 10-epoch fresh-training analysis curves from per-epoch Val metrics.

Data source: out/select/epoch-N.val.report.json (Val split) + out/epoch-N/train_meta.json
(Train loss). Test is never read here. Produces 5 standalone PNGs + a combined 5-panel
figure + raw values as JSON/CSV.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROWS = json.loads((HERE / "val_metrics.json").read_text(encoding="utf-8"))

best_epoch = max(ROWS, key=lambda r: (r["val_top1"], -r["epoch"]))["epoch"]

# ── raw values: CSV ──────────────────────────────────────────────────────────
cols = ["epoch", "train_loss", "val_top1", "val_ece", "val_reject_recall",
        "val_reject_fpr", "val_reject_precision", "gold_reject_n"]
with (HERE / "val_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    w.writerows(ROWS)

epochs = [r["epoch"] for r in ROWS]

SERIES = [
    ("train_loss", "Train Loss vs Epoch", "Train Loss (mean, CE+RL)", "train_loss_vs_epoch.png"),
    ("val_top1", "Val Top-1 vs Epoch", "Val Top-1", "val_top1_vs_epoch.png"),
    ("val_ece", "Val ECE vs Epoch", "Val ECE", "val_ece_vs_epoch.png"),
    ("val_reject_recall", "Val REJECT Recall vs Epoch", "Val REJECT Recall", "val_reject_recall_vs_epoch.png"),
    ("val_reject_fpr", "Val REJECT FPR vs Epoch", "Val REJECT FPR", "val_reject_fpr_vs_epoch.png"),
]


def draw(ax, vals, title, ylabel):
    ax.plot(epochs, vals, marker="o", linewidth=1.8, color="#1f77b4")
    bi = epochs.index(best_epoch)
    ax.scatter([best_epoch], [vals[bi]], s=90, facecolors="none",
               edgecolors="#d62728", linewidths=2, zorder=5, label=f"best epoch {best_epoch}")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_xticks(epochs)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


# ── 5 standalone PNGs ────────────────────────────────────────────────────────
for key, title, ylabel, fname in SERIES:
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=140)
    draw(ax, [r[key] for r in ROWS], title, ylabel)
    fig.tight_layout()
    fig.savefig(HERE / fname)
    plt.close(fig)

# ── combined 5-panel figure ──────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(12, 12), dpi=140)
flat = [a for row in axes for a in row]
for ax, (key, title, ylabel, _) in zip(flat, SERIES):
    draw(ax, [r[key] for r in ROWS], title, ylabel)
flat[-1].axis("off")
sup = (f"LayaChoice V2 — 10-epoch fresh training (Train 2112 / Val 300)\n"
       f"best epoch = {best_epoch} (val top1 = "
       f"{max(r['val_top1'] for r in ROWS):.4f}); curves from Val + Train only, no Test")
fig.suptitle(sup, fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(HERE / "val_curves.png")
plt.close(fig)

print("wrote:", HERE)
for _, _, _, f in SERIES:
    print("  ", f)
print("   val_curves.png")
print("   val_metrics.csv")
print("best_epoch", best_epoch)
