from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
OUT_ROOT   = PROJECT_ROOT / "Stage1_rerun/runs"
FIG_DIR    = Path(__file__).resolve().parent / "figures"
TABLES_DIR = Path(__file__).resolve().parent / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

CLASSES   = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
CNN_RUNS  = ["resnet50_full_finetune_seed42",
             "resnet152_full_finetune_seed42",
             "densenet121_full_finetune_seed42",
             "densenet201_full_finetune_seed42"]

def load_run(run_name: str) -> dict | None:
    run_dir = OUT_ROOT / run_name
    test_npz = run_dir / "predictions_test.npz"
    if not test_npz.exists():
        return None
    test = np.load(test_npz)
    val_npz = run_dir / "predictions_val.npz"
    val = np.load(val_npz) if val_npz.exists() else None
    summary = json.load(open(run_dir / "summary.json"))
    return {
        "run":         run_name,
        "logits":      test["logits"],
        "probs":       test["probs"],
        "labels":      test["labels"],
        "val_logits":  val["logits"] if val is not None else None,
        "val_labels":  val["labels"] if val is not None else None,
        "test_acc":    float(summary["test_accuracy"]),
        "macro_f1":    float(summary["test_macro_f1"]),
        "per_class_f1": [summary["test_per_class_f1"][c] for c in CLASSES],
    }


def best_run_for(prefix: str) -> str | None:
    cands = []
    for d in OUT_ROOT.iterdir():
        if not d.name.startswith(prefix): continue
        sj = d / "summary.json"
        if not sj.exists(): continue
        cands.append((json.load(open(sj))["test_accuracy"], d.name))
    if not cands: return None
    return max(cands)[1]


def hard_vote(probs_stack: np.ndarray) -> np.ndarray:
    votes = probs_stack.argmax(axis=-1)
    out = np.zeros(votes.shape[1], dtype=np.int64)
    for i in range(votes.shape[1]):
        out[i] = np.bincount(votes[:, i], minlength=probs_stack.shape[-1]).argmax()
    return out
def prob_avg(probs_stack: np.ndarray) -> np.ndarray:
    return probs_stack.mean(axis=0).argmax(axis=-1)
def logit_avg(logits_stack: np.ndarray) -> np.ndarray:
    return logits_stack.mean(axis=0).argmax(axis=-1)
def weighted_prob(probs_stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = weights / weights.sum()
    return (probs_stack * w[:, None, None]).sum(axis=0).argmax(axis=-1)
def stacking(val_logits_list, val_labels, test_logits_list) -> np.ndarray | None:
    if any(v is None for v in val_logits_list):
        return None
    val_X  = np.concatenate(val_logits_list,  axis=1)
    test_X = np.concatenate(test_logits_list, axis=1)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(val_X, val_labels)
    return clf.predict(test_X)

def run_ensembles(members: list[dict]) -> pd.DataFrame:
    labels       = members[0]["labels"]
    probs_stack  = np.stack([m["probs"]  for m in members])
    logits_stack = np.stack([m["logits"] for m in members])
    accs         = np.array([m["test_acc"] for m in members])
    rows = [{"name": m["run"], "kind": "single",
             "test_acc": m["test_acc"], "macro_f1": m["macro_f1"]}
            for m in members]
    methods = {
        "ens: hard_vote":hard_vote(probs_stack),
        "ens: prob_avg":prob_avg(probs_stack),
        "ens: logit_avg":logit_avg(logits_stack),
        "ens: weighted_prob":weighted_prob(probs_stack, accs),
    }
    stack_pred = stacking([m["val_logits"]  for m in members],
                          members[0]["val_labels"]
                            if members[0]["val_labels"] is not None else None,
                          [m["logits"] for m in members])
    if stack_pred is not None:
        methods["ens: stacking"] = stack_pred

    for name, preds in methods.items():
        rows.append({
            "name": name, "kind": "ensemble",
            "test_acc": float(accuracy_score(labels, preds)),
            "macro_f1": float(f1_score(labels, preds, average="macro")),
            "preds":    preds,
        })
    return pd.DataFrame(rows)


def plot_acc_bar(df: pd.DataFrame, savename: str) -> None:
    df_sorted = df.sort_values("test_acc")
    x = np.arange(len(df_sorted)); w = 0.4
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w / 2, df_sorted.test_acc, w, label="test acc")
    ax.bar(x + w / 2, df_sorted.macro_f1, w, label="macro F1")
    for xi, a, f in zip(x, df_sorted.test_acc, df_sorted.macro_f1):
        ax.text(xi - w / 2, a, f"{a:.3f}", ha="center", va="bottom", fontsize=7)
        ax.text(xi + w / 2, f, f"{f:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(df_sorted.name, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 0.9); ax.legend()
    ax.set_title("ensemble vs single-model accuracy")
    fig.tight_layout(); fig.savefig(FIG_DIR / savename, dpi=120); plt.close(fig)


def plot_best_confusion(df: pd.DataFrame, members: list[dict], savename: str) -> str:
    ens = df[df.kind == "ensemble"].sort_values("test_acc", ascending=False).iloc[0]
    cm  = confusion_matrix(members[0]["labels"], ens["preds"], labels=list(range(7)))
    cmn = cm / cm.sum(1, keepdims=True).clip(1)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cmn, vmin=0, vmax=1)
    ax.set_xticks(range(7)); ax.set_yticks(range(7))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right"); ax.set_yticklabels(CLASSES)
    ax.set_title(f"{ens['name']}  (acc={ens['test_acc']:.3f})")
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.045)
    fig.tight_layout(); fig.savefig(FIG_DIR / savename, dpi=120); plt.close(fig)
    return ens["name"]


def plot_per_class_f1(df: pd.DataFrame, members: list[dict], savename: str) -> None:
    ens = df[df.kind == "ensemble"].sort_values("test_acc", ascending=False).iloc[0]
    labels = members[0]["labels"]
    ens_per_class = f1_score(labels, ens["preds"], average=None,
                             labels=list(range(7)))

    x = np.arange(7); n = len(members) + 1; w = 0.8 / n
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, m in enumerate(members):
        ax.bar(x + (i - n / 2) * w + w / 2, m["per_class_f1"], w,
               label=m["run"].split("_full_finetune")[0].split("_seed")[0])
    ax.bar(x + (n - 1 - n / 2) * w + w / 2, ens_per_class, w,
           label=ens["name"], color="black")
    ax.set_xticks(x); ax.set_xticklabels(CLASSES)
    ax.set_ylim(0, 1); ax.set_ylabel("macro F1")
    ax.set_title("per-class F1: members vs best ensemble")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(FIG_DIR / savename, dpi=120); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hetero", action="store_true")
    args = p.parse_args()

    member_names = list(CNN_RUNS)  #CNN
    if args.hetero: #+encoder
        for prefix in ("clip_vitl14_", "dinov2_vitl14_"):
            member_names.append(best_run_for(prefix))

    members = [load_run(name) for name in member_names]
    for name, m in zip(member_names, members):
        print(f"  loaded {name}  test_acc={m['test_acc']:.4f}")

    df = run_ensembles(members)
    df_save = df.drop(columns=["preds"], errors="ignore")
    df_save.to_csv(TABLES_DIR / "ensemble_summary.csv", index=False)
    plot_acc_bar(df_save, "ensemble_acc.png")
    best_name = plot_best_confusion(df, members, "ensemble_confusion.png")
    plot_per_class_f1(df, members, "ensemble_per_class.png")
    print(f"\nbest ensemble method: {best_name}")


if __name__ == "__main__":
    main()
