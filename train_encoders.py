from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_worker import TrainingJob, train_one

CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
SEED    = 42

OUT_ROOT = PROJECT_ROOT / "Stage1_rerun/runs"
FIG_DIR  = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ENCODERS   = ("clip_vitl14", "dinov2_vitl14", "sapiens_06b")
STRATEGIES = ("linear_probe", "full_finetune", "llrd_finetune", "frozen_llrd")


@dataclass
class EncoderRecipe:
    epochs:      int = 12
    batch_size:  int = 128
    base_config: str = "configs/base.yaml"
    seed:        int = SEED

def make_jobs_for_backbone(recipe: EncoderRecipe, backbone: str,
                           gpus=(0, 1, 2, 3)) -> list[TrainingJob]:
    is_sapiens = backbone == "sapiens_06b"
    bsz = 64 if is_sapiens else recipe.batch_size
    jobs = []
    for st, gpu in zip(STRATEGIES, gpus):
        feb = 16 if (is_sapiens and st == "frozen_llrd") else None
        jobs.append(TrainingJob(
            backbone=backbone, strategy=st, gpu=gpu,
            epochs=recipe.epochs, batch_size=bsz,
            base_config=recipe.base_config, seed=recipe.seed,
            frozen_early_blocks=feb, out_root=str(OUT_ROOT),
        ))
    return jobs

def parallel_train(jobs: list[TrainingJob]) -> list[dict]:
    ctx = mp.get_context("spawn")
    ctx.set_executable(sys.executable)
    procs = []
    for j in jobs:
        name = f"{j.backbone}_{j.strategy}_gpu{j.gpu}"
        p = ctx.Process(target=train_one, args=(j,), name=name)
        p.start(); procs.append((p, j))
        print(f"  started {name} (pid {p.pid})  bs={j.batch_size}")
    rcs = []
    for p, j in procs:
        p.join()
        rcs.append({"run_name": f"{j.backbone}_{j.strategy}_seed{j.seed}",
                    "gpu": j.gpu, "exit": p.exitcode})
    return rcs


def _run_dir(backbone: str, strategy: str, seed: int = SEED) -> Path:
    return OUT_ROOT / f"{backbone}_{strategy}_seed{seed}"


def _completed_runs() -> list[tuple[str, str]]:
    out = []
    for bb in ENCODERS:
        for st in STRATEGIES:
            if (_run_dir(bb, st) / "summary.json").exists():
                out.append((bb, st))
    return out


def plot_training_curves(pairs: list[tuple[str, str]]) -> None:
    fig, axes = plt.subplots(len(STRATEGIES), len(ENCODERS),
                             figsize=(4 * len(ENCODERS), 2.6 * len(STRATEGIES)))
    for r, st in enumerate(STRATEGIES):
        for c, bb in enumerate(ENCODERS):
            ax = axes[r, c] if axes.ndim == 2 else axes[c]
            run_dir = _run_dir(bb, st)
            if (bb, st) not in pairs:
                ax.set_title(f"{bb}\n{st}\n(missing)", fontsize=8)
                ax.axis("off"); continue
            df = pd.read_csv(run_dir / "logs/metrics.csv")
            ax.plot(df.epoch, df.train_acc, label="train")
            ax.plot(df.epoch, df.val_acc,   label="val")
            ax.set_title(f"{bb}\n{st}", fontsize=8)
            ax.set_xlabel("epoch"); ax.legend(fontsize=7)
    fig.suptitle("ViT-L encoder training curves")
    fig.tight_layout(); fig.savefig(FIG_DIR / "encoder_curves.png", dpi=120); plt.close(fig)


def plot_strategy_grid(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    M = np.full((len(ENCODERS), len(STRATEGIES)), np.nan)
    for i, bb in enumerate(ENCODERS):
        for j, st in enumerate(STRATEGIES):
            if (bb, st) in pairs:
                s = json.load(open(_run_dir(bb, st) / "summary.json"))
                M[i, j] = s["test_accuracy"] * 100

    x = np.arange(len(STRATEGIES)); w = 0.27
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, bb in enumerate(ENCODERS):
        ax.bar(x + (i - 1) * w, M[i], w, label=bb)
        for j, v in enumerate(M[i]):
            if not np.isnan(v):
                ax.text(x[j] + (i - 1) * w, v, f"{v:.1f}",
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(STRATEGIES)
    ax.set_ylim(0, 90); ax.set_ylabel("test accuracy (%)")
    ax.set_title("ViT-L encoder × strategy")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "strategy_grid.png", dpi=120); plt.close(fig)
    return pd.DataFrame(M, index=list(ENCODERS), columns=list(STRATEGIES)).round(2)


def plot_best_per_encoder(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for bb in ENCODERS:
        cands = [(bb, st) for st in STRATEGIES if (bb, st) in pairs]
        if not cands: continue
        best = max(cands,
                   key=lambda p: json.load(open(_run_dir(*p) / "summary.json"))["test_accuracy"])
        s = json.load(open(_run_dir(*best) / "summary.json"))
        rows.append({"encoder":  bb, "strategy": best[1],
                     "test_acc": s["test_accuracy"], "macro_f1": s["test_macro_f1"]})
    if not rows:
        return pd.DataFrame(rows)
    df = pd.DataFrame(rows)
    x = np.arange(len(df)); w = 0.4
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, df.test_acc, w, label="test acc")
    ax.bar(x + w / 2, df.macro_f1, w, label="macro F1")
    for xi, a, f in zip(x, df.test_acc, df.macro_f1):
        ax.text(xi - w / 2, a, f"{a:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + w / 2, f, f"{f:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.encoder}\n({r.strategy})" for _, r in df.iterrows()], fontsize=8)
    ax.set_ylim(0, 0.9); ax.legend()
    ax.set_title("best strategy per encoder")
    fig.tight_layout(); fig.savefig(FIG_DIR / "best_per_encoder.png", dpi=120); plt.close(fig)
    return df


PALETTE = [
    (0.835, 0.275, 0.235), (0.561, 0.345, 0.706), (0.282, 0.478, 0.725),
    (0.922, 0.682, 0.176), (0.337, 0.604, 0.392), (0.871, 0.463, 0.196),
    (0.275, 0.627, 0.667),
]


def _stratified_subsample(X: np.ndarray, y: np.ndarray, n: int, seed: int):
    if n >= len(X): return X, y
    rng = np.random.default_rng(seed)
    take_per = max(1, n // 7)
    sel = []
    for c in range(7):
        idx = np.flatnonzero(y == c)
        if idx.size == 0: continue
        sel.append(rng.choice(idx, size=min(take_per, idx.size), replace=False))
    sel = np.concatenate(sel)
    if sel.size > n:
        sel = rng.choice(sel, size=n, replace=False)
    return X[sel], y[sel]


def _project_2d(X: np.ndarray, seed: int = SEED) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import umap
    Xs = StandardScaler().fit_transform(X)
    if Xs.shape[1] > 50:
        Xs = PCA(n_components=50, random_state=seed).fit_transform(Xs)
    return umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                     random_state=seed).fit_transform(Xs)


def plot_umap(pairs: list[tuple[str, str]], n_sub: int = 2500) -> None:
    try:
        import umap
    except ImportError:
        print("umap-learn not installed; skipping UMAP plot"); return
    warnings.filterwarnings("ignore", category=UserWarning)
    fig, axes = plt.subplots(len(ENCODERS), len(STRATEGIES),
                             figsize=(5.0 * len(STRATEGIES), 5.0 * len(ENCODERS)))
    axes = np.atleast_2d(axes)
    for r, bb in enumerate(ENCODERS):
        for c, st in enumerate(STRATEGIES):
            ax = axes[r, c]
            emb_path = _run_dir(bb, st) / "embeddings_test.npz"
            if (bb, st) not in pairs or not emb_path.exists():
                ax.text(0.5, 0.5, f"missing\n{bb}\n{st}",
                        ha="center", va="center", transform=ax.transAxes, fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            d = np.load(emb_path)
            feats, labels = d["feats"].astype(np.float32), d["labels"].astype(np.int64)
            Xs, ys = _stratified_subsample(feats, labels, n_sub, SEED)
            print(f"  projecting {bb} × {st}: {Xs.shape}", flush=True)
            z = _project_2d(Xs, seed=SEED)
            for cls in range(7):
                m = ys == cls
                ax.scatter(z[m, 0], z[m, 1], s=6, alpha=0.55,
                           c=[PALETTE[cls]],
                           label=CLASSES[cls] if (r == 0 and c == 3) else None)
            s = json.load(open(_run_dir(bb, st) / "summary.json"))
            ax.set_title(f"{bb} × {st}\nacc={s['test_accuracy']*100:.2f}%  "
                         f"F1={s['test_macro_f1']*100:.2f}%", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0 and c == 3:
                ax.legend(loc="upper right", fontsize=7, markerscale=1.5, framealpha=0.95)
    fig.suptitle("ViT-L encoder embeddings (UMAP) — rows: encoder, cols: strategy",
                 fontsize=12, y=0.995)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "encoder_umap.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("backbone", nargs="?", default=None, choices=(*ENCODERS, None))
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--skip-train", action="store_true")
    args = p.parse_args()

    gpus = tuple(int(x) for x in args.gpus.split(","))
    assert len(gpus) == 4, "need 4 GPUs (one per strategy)"

    recipe = EncoderRecipe()
    print(f"== ViT-L encoder training recipe ==")
    print(f"  epochs       : {recipe.epochs}")
    print(f"  batch_size   : {recipe.batch_size}  (sapiens uses 64)")
    print(f"  base_config  : {recipe.base_config}")
    print(f"  out_root     : {OUT_ROOT}")
    print(f"  GPUs         : {gpus}")

    if not args.skip_train:
        backbones = [args.backbone] if args.backbone else list(ENCODERS)
        for bb in backbones:
            print(f"\n== Wave: {bb} ==")
            jobs = make_jobs_for_backbone(recipe, bb, gpus=gpus)
            for j in jobs:
                print(f"  {j.strategy:14s} -> GPU {j.gpu}  bs={j.batch_size}"
                      + (f"  (frozen_early_blocks={j.frozen_early_blocks})"
                         if j.frozen_early_blocks else ""))
            results = parallel_train(jobs)
            for r in results:
                tag = "OK" if r["exit"] == 0 else f"FAIL(rc={r['exit']})"
                print(f"  {tag}  {r['run_name']}  (GPU {r['gpu']})")

    pairs = _completed_runs()
    if not pairs:
        print("\nno completed encoder runs found — skipping plots"); return
    print(f"\n== Plotting from {len(pairs)} completed runs ==")
    plot_training_curves(pairs)
    grid = plot_strategy_grid(pairs)
    print(grid)
    best = plot_best_per_encoder(pairs)
    print(best.to_string(index=False))
    plot_umap(pairs)


if __name__ == "__main__":
    main()
