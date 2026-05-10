from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED    = PROJECT_ROOT / "data/processed"
FIG_DIR      = Path(__file__).resolve().parent / "figures"
TABLES_DIR   = Path(__file__).resolve().parent / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
SEED    = 42

# 68 s (used by dlib's shape predictor)
LANDMARK_GROUPS = {
    "jaw":           (range(0, 17),  "tab:gray"),
    "right_eyebrow": (range(17, 22), "tab:red"),
    "left_eyebrow":  (range(22, 27), "tab:orange"),
    "nose":          (range(27, 36), "tab:olive"),
    "right_eye":     (range(36, 42), "tab:cyan"),
    "left_eye":      (range(42, 48), "tab:blue"),
    "mouth":         (range(48, 68), "tab:pink"),
}

LM_ANCHORS = [17, 19, 21, 22, 24, 26, 36, 39, 42, 45, 30, 31, 35, 48, 54, 51, 57, 8]


def _load_raw_imgs() -> np.ndarray:
    parts = []
    for sp in ("train", "val", "test"):
        obj = torch.load(PROCESSED / f"{sp}.pt", map_location="cpu", weights_only=False)
        x = obj["images"].numpy() if hasattr(obj["images"], "numpy") else np.asarray(obj["images"])
        parts.append(x)
    return np.concatenate(parts, axis=0)


def load_landmarks():
    lm = torch.load(PROCESSED / "dlib_landmarks.pt", map_location="cpu", weights_only=False)
    def _np(x): return x.numpy() if hasattr(x, "numpy") else np.asarray(x)
    return {
        "xy":           _np(lm["landmarks_xy_48"]).astype(np.float32),
        "labels":       _np(lm["label"]).astype(np.int64),
        "found":        _np(lm["found"]).astype(bool),
        "split":        np.asarray(lm["split"]),
        "global_index": _np(lm["global_index"]).astype(np.int64),
    }

def shape_features(xy: np.ndarray) -> np.ndarray:
    eye_l  = xy[:, 36:42].mean(axis=1)
    eye_r  = xy[:, 42:48].mean(axis=1)
    centre = (eye_l + eye_r) / 2
    scale  = np.maximum(np.linalg.norm(eye_l - eye_r, axis=1), 1e-6)
    pts    = (xy[:, LM_ANCHORS] - centre[:, None, :]) / scale[:, None, None]
    return pts.reshape(len(xy), -1)

def plot_landmarks_overlay(lm: dict, raw_imgs: np.ndarray) -> None:
    examples = []
    for c in range(7):
        cand = np.where((lm["labels"] == c) & lm["found"])[0]
        examples.append(int(cand[0]) if len(cand) else None)

    fig, axes = plt.subplots(1, 7, figsize=(15, 2.6))
    for ax, idx, name in zip(axes, examples, CLASSES):
        if idx is None:
            ax.axis("off"); ax.set_title(f"{name}\n(no face)"); continue
        gi = int(lm["global_index"][idx])
        ax.imshow(raw_imgs[gi], cmap="gray")
        pts = lm["xy"][idx]
        for gname, (rng, col) in LANDMARK_GROUPS.items():
            ix = list(rng)
            ax.scatter(pts[ix, 0], pts[ix, 1], s=10, c=col, label=gname if ax is axes[0] else None)
        ax.scatter(pts[LM_ANCHORS, 0], pts[LM_ANCHORS, 1],
                   s=70, facecolors="none", edgecolors="yellow", linewidths=1.0)
        ax.set_title(name); ax.set_xticks([]); ax.set_yticks([])
    fig.legend(loc="lower center", ncol=7, fontsize=7, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("dlib 68 landmarks (yellow rings = 18 anchors used by shape_features)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "landmarks_overlay.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def train_classifiers(lm: dict):
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    X = shape_features(lm["xy"])
    y = lm["labels"]
    good = lm["found"] & np.isfinite(lm["xy"]).all(axis=(1, 2))
    tr = good & (lm["split"] == "train")
    va = good & (lm["split"] == "val")
    te = good & (lm["split"] == "test")
    print(f"  face-detected splits: train={tr.sum()} val={va.sum()} test={te.sum()}")

    # MLP — keep loss + val accuracy per epoch
    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4, max_iter=300,
                      early_stopping=True, validation_fraction=0.1,
                      random_state=SEED),
    )
    mlp.fit(X[tr], y[tr])
    mlp_clf = mlp.named_steps["mlpclassifier"]

    n_grid = list(range(20, 321, 20))
    rf = RandomForestClassifier(n_estimators=n_grid[0], max_depth=18,
                                class_weight="balanced", oob_score=True,
                                warm_start=True, random_state=SEED, n_jobs=-1,
                                bootstrap=True)
    oob_curve = []
    for n in n_grid:
        rf.n_estimators = n
        rf.fit(X[tr], y[tr])
        oob_curve.append(rf.oob_score_)

    metrics = []
    for name, model in [("Random Forest", rf), ("MLP", mlp)]:
        yp_te = model.predict(X[te]); yp_va = model.predict(X[va])
        metrics.append({
            "model":         "dlib ERT",
            "classifier":    name,
            "feature_dim":   X.shape[1],
            "val_accuracy":  accuracy_score(y[va], yp_va),
            "test_accuracy": accuracy_score(y[te], yp_te),
            "macro_f1":      f1_score(y[te], yp_te, average="macro"),
            "weighted_f1":   f1_score(y[te], yp_te, average="weighted"),
        })
    df = pd.DataFrame(metrics)
    df.to_csv(TABLES_DIR / "landmark_models.csv", index=False)
    return {
        "rf": rf, "mlp": mlp, "mlp_clf": mlp_clf,
        "X": X, "y": y, "tr": tr, "va": va, "te": te,
        "oob_curve": oob_curve, "n_grid": n_grid, "metrics": df,
    }


def plot_mlp_training(out: dict) -> None:
    mlp = out["mlp_clf"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(np.arange(1, len(mlp.loss_curve_) + 1), mlp.loss_curve_, marker="o", ms=3)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("training loss")
    axes[0].set_title(f"MLP training loss ({len(mlp.loss_curve_)} epochs)")
    axes[0].grid(alpha=0.3)
    if len(mlp.validation_scores_):
        axes[1].plot(np.arange(1, len(mlp.validation_scores_) + 1),
                     mlp.validation_scores_, marker="o", ms=3)
        axes[1].axhline(max(mlp.validation_scores_), ls="--", lw=0.8, color="k",
                        label=f"best={max(mlp.validation_scores_):.3f}")
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("held-out val accuracy")
        axes[1].set_title("MLP validation accuracy (early stopping)")
        axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mlp_training.png", dpi=120); plt.close(fig)


def plot_rf_diagnostics(out: dict) -> None:
    rf, n_grid, oob = out["rf"], out["n_grid"], out["oob_curve"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(n_grid, oob, marker="o")
    axes[0].set_xlabel("n_estimators"); axes[0].set_ylabel("OOB accuracy")
    axes[0].set_title("Random Forest OOB accuracy vs n_estimators")
    axes[0].grid(alpha=0.3)

    # Aggregate per-anchor importance: 36 features = 18 anchors × (x, y)
    imp_xy = rf.feature_importances_.reshape(18, 2).sum(axis=1)
    order  = np.argsort(imp_xy)[::-1]
    group_for_anchor = []
    for ap in LM_ANCHORS:
        for gname, (rng, _col) in LANDMARK_GROUPS.items():
            if ap in rng:
                group_for_anchor.append(gname); break
    labels = [f"{ap}\n({group_for_anchor[i]})" for i, ap in enumerate(LM_ANCHORS)]
    axes[1].bar(np.arange(18), imp_xy[order])
    axes[1].set_xticks(np.arange(18))
    axes[1].set_xticklabels([labels[i] for i in order], fontsize=6, rotation=45, ha="right")
    axes[1].set_ylabel("importance (x+y combined)")
    axes[1].set_title("RF feature importance per anchor (sorted)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rf_diagnostics.png", dpi=120); plt.close(fig)


def plot_method1_confusion(out: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (name, clf) in zip(axes, [("Random Forest", out["rf"]), ("MLP", out["mlp"])]):
        yp = clf.predict(out["X"][out["te"]])
        cm = confusion_matrix(out["y"][out["te"]], yp, labels=list(range(7)))
        cmn = cm / cm.sum(1, keepdims=True).clip(1)
        im = ax.imshow(cmn, vmin=0, vmax=1)
        ax.set_xticks(range(7)); ax.set_yticks(range(7))
        ax.set_xticklabels(CLASSES, rotation=45, ha="right"); ax.set_yticklabels(CLASSES)
        acc = (yp == out["y"][out["te"]]).mean()
        ax.set_title(f"{name} — test acc {acc:.3f}")
        for i in range(7):
            for j in range(7):
                ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.045)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "method1_confusion.png", dpi=120); plt.close(fig)


def plot_method1_summary(out: dict) -> None:
    df = out["metrics"]
    labs = [f"{r.model}+{r.classifier}" for _, r in df.iterrows()]
    x = np.arange(len(labs)); w = 0.4
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, df.test_accuracy, w, label="test acc")
    ax.bar(x + w / 2, df.macro_f1,      w, label="macro F1")
    for xi, a, f in zip(x, df.test_accuracy, df.macro_f1):
        ax.text(xi - w / 2, a, f"{a:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + w / 2, f, f"{f:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylim(0, 0.7); ax.legend()
    ax.set_title("Method 1: dlib 36-d shape features + classifier")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "method1_summary.png", dpi=120); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-train", action="store_true")
    args = p.parse_args()

    print(f"== dlib landmarks pipeline ==")
    print(f"  landmark file : {PROCESSED/'dlib_landmarks.pt'}")
    print(f"  figures out   : {FIG_DIR}")

    raw_imgs = _load_raw_imgs()
    lm       = load_landmarks()
    print(f"  total rows    : {len(lm['xy'])}  | face_found: {int(lm['found'].sum())}")

    plot_landmarks_overlay(lm, raw_imgs)
    print("  saved landmarks_overlay.png")

    if args.skip_train:
        return

    out = train_classifiers(lm)
    plot_mlp_training(out)
    plot_rf_diagnostics(out)
    plot_method1_confusion(out)
    plot_method1_summary(out)
    print("  saved mlp_training.png, rf_diagnostics.png, "
          "method1_confusion.png, method1_summary.png")
    print(out["metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
