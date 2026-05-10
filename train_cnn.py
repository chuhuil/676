from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import sys
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_worker import TrainingJob, train_one  # noqa: E402

CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
SEED    = 42

OUT_ROOT = PROJECT_ROOT / "Stage1_rerun/runs"
FIG_DIR  = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class CNNRecipe:
    epochs:int  = 12
    batch_size:int  = 128
    base_config:str  = "configs/cnn_balanced.yaml"
    strategy:str  = "full_finetune"
    seed:int  = SEED

def make_jobs(recipe: CNNRecipe, backbones=("resnet50", "resnet152", "densenet121", "densenet201"),
              gpus=(0, 1, 2, 3)) -> list[TrainingJob]:
    assert len(backbones) == len(gpus), "one GPU per backbone"
    return [TrainingJob(backbone=bb, strategy=recipe.strategy, gpu=g,
                        epochs=recipe.epochs, batch_size=recipe.batch_size,
                        base_config=recipe.base_config, seed=recipe.seed,
                        out_root=str(OUT_ROOT))
            for bb, g in zip(backbones, gpus)]


def parallel_train(jobs: list[TrainingJob]) -> list[dict]:
    ctx = mp.get_context("spawn")
    ctx.set_executable(sys.executable)
    procs = []
    for j in jobs:
        name = f"{j.backbone}_{j.strategy}_gpu{j.gpu}"
        p = ctx.Process(target=train_one, args=(j,), name=name)
        p.start(); procs.append((p, j))
        print(f"  started {name} (pid {p.pid})")
    rcs = []
    for p, j in procs:
        p.join()
        rcs.append({"run_name": f"{j.backbone}_{j.strategy}_seed{j.seed}",
                    "gpu": j.gpu, "exit": p.exitcode})
    return rcs


def plot_training_curves(run_names: list[str]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, run in zip(axes.ravel(), run_names):
        df = pd.read_csv(OUT_ROOT / run / "logs/metrics.csv")
        ax.plot(df.epoch, df.train_acc, label="train acc")
        ax.plot(df.epoch, df.val_acc,   label="val acc")
        ax.set_xlabel("epoch"); ax.set_title(run); ax.legend(fontsize=8)
    fig.suptitle("CNN training curves (full_finetune)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "cnn_curves.png", dpi=120); plt.close(fig)


def plot_test_bars(run_names: list[str]) -> pd.DataFrame:
    rows = []
    for run in run_names:
        s = json.load(open(OUT_ROOT / run / "summary.json"))
        rows.append({"run": run, "test_acc": s["test_accuracy"],
                     "macro_f1": s["test_macro_f1"]})
    df = pd.DataFrame(rows).sort_values("test_acc")
    x = np.arange(len(df)); w = 0.4
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, df.test_acc, w, label="test acc")
    ax.bar(x + w / 2, df.macro_f1, w, label="macro F1")
    for xi, a, f in zip(x, df.test_acc, df.macro_f1):
        ax.text(xi - w / 2, a, f"{a:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + w / 2, f, f"{f:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(df.run, rotation=20, ha="right")
    ax.set_ylim(0, 0.85); ax.legend()
    ax.set_title("CNN — test accuracy & macro F1")
    fig.tight_layout(); fig.savefig(FIG_DIR / "cnn_test_bars.png", dpi=120); plt.close(fig)
    return df


def plot_confusion(run_names: list[str]) -> None:
    fig, axes = plt.subplots(1, len(run_names), figsize=(4.5 * len(run_names), 4.4))
    for ax, run in zip(axes, run_names):
        s = json.load(open(OUT_ROOT / run / "summary.json"))
        cm = np.asarray(s["test_confusion_matrix"], dtype=float)
        cmn = cm / cm.sum(1, keepdims=True)
        im = ax.imshow(cmn, vmin=0, vmax=1)
        ax.set_xticks(range(7)); ax.set_yticks(range(7))
        ax.set_xticklabels(CLASSES, rotation=45, ha="right"); ax.set_yticklabels(CLASSES)
        ax.set_title(f"{run.split('_')[0]} (acc={s['test_accuracy']:.3f})")
        for i in range(7):
            for j in range(7):
                ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.045)
    fig.suptitle("CNN confusion matrices")
    fig.tight_layout(); fig.savefig(FIG_DIR / "cnn_confusion.png", dpi=120); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("backbone", nargs="?", default=None)
    args = p.parse_args()

    recipe = CNNRecipe()
    print(f"== CNN training recipe ==")
    print(f"  epochs       : {recipe.epochs}")
    print(f"  batch_size   : {recipe.batch_size}")
    print(f"  strategy     : {recipe.strategy}")
    print(f"  base_config  : {recipe.base_config}")
    print(f"  out_root     : {OUT_ROOT}")

    if args.backbone is not None:
        jobs = [TrainingJob(backbone=args.backbone, strategy=recipe.strategy, gpu=0,
                            epochs=recipe.epochs, batch_size=recipe.batch_size,
                            base_config=recipe.base_config, seed=recipe.seed,
                            out_root=str(OUT_ROOT))]
    else:
        jobs = make_jobs(recipe)

    print("\n== Launching ==")
    for j in jobs:
        print(f"  {j.backbone:13s} -> GPU {j.gpu}")
    results = parallel_train(jobs)
    for r in results:
        tag = "OK" if r["exit"] == 0 else f"FAIL(rc={r['exit']})"
        print(f"  {tag}  {r['run_name']}  (GPU {r['gpu']})")

    run_names = [r["run_name"] for r in results
                 if (OUT_ROOT / r["run_name"] / "summary.json").exists()]
    if not run_names:
        print("no completed runs found — skipping plots"); return
    print(f"\n== Plotting from {len(run_names)} completed runs ==")
    plot_training_curves(run_names)
    df = plot_test_bars(run_names)
    plot_confusion(run_names)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
