from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED    = PROJECT_ROOT / "data/processed"
FIG_DIR      = Path(__file__).resolve().parent / "figures"
TABLES_DIR   = Path(__file__).resolve().parent / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
SEED    = 42
RNG     = np.random.default_rng(SEED)


@dataclass
class RawSplits:
    images:np.ndarray
    labels:np.ndarray
    split:np.ndarray
    split_index: np.ndarray


def _load_one_split(name: str):
    obj  = torch.load(PROCESSED / f"{name}.pt", map_location="cpu", weights_only=False)
    imgs = obj["images"].numpy() if hasattr(obj["images"], "numpy") else np.asarray(obj["images"])
    lbls = obj["labels"].numpy() if hasattr(obj["labels"], "numpy") else np.asarray(obj["labels"])
    return imgs, lbls


def load_raw_splits() -> RawSplits:
    parts, splits, splitidx = [], [], []
    for name in ("train", "val", "test"):
        imgs, lbls = _load_one_split(name)
        parts.append((imgs, lbls))
        splits.extend([name] * len(imgs))
        splitidx.extend(range(len(imgs)))
    return RawSplits(
        images=np.concatenate([p[0] for p in parts]),
        labels=np.concatenate([p[1] for p in parts]),
        split=np.asarray(splits),
        split_index=np.asarray(splitidx),
    )


def _save(fig, name):
    fig.savefig(FIG_DIR / name)
    plt.close(fig)


def explore_dataset(raw: RawSplits) -> None:
    # Class distribution per split
    fig, ax = plt.subplots()
    x = np.arange(7)
    for sp in ("train", "val", "test"):
        m = raw.split == sp
        counts = [int(((raw.labels == c) & m).sum()) for c in range(7)]
        ax.plot(x, counts, marker="o", label=sp)
    ax.set_xticks(x); ax.set_xticklabels(CLASSES)
    ax.set_title("class distribution")
    ax.legend()
    _save(fig, "class_distribution.png")

    # Random samples per class (raw train)
    train_mask = raw.split == "train"
    n_per = 8
    fig, axes = plt.subplots(7, n_per)
    for ci in range(7):
        cand = np.where(train_mask & (raw.labels == ci))[0]
        for j, gi in enumerate(RNG.choice(cand, size=n_per, replace=False)):
            axes[ci, j].imshow(raw.images[gi], cmap="gray")
            axes[ci, j].axis("off")
        axes[ci, 0].set_ylabel(CLASSES[ci])
    fig.suptitle("random samples")
    _save(fig, "random_samples.png")

    # Per-class mean face 【orignal train】
    fig, axes = plt.subplots(1, 7)
    for ci in range(7):
        m = train_mask & (raw.labels == ci)
        axes[ci].imshow(raw.images[m].astype(np.float32).mean(0), cmap="gray")
        axes[ci].set_title(CLASSES[ci])
        axes[ci].axis("off")
    fig.suptitle("mean face per class")
    _save(fig, "mean_face.png")

    # Pixel intensity histogram per class
    fig, axes = plt.subplots(2, 4)
    for ci in range(7):
        ax = axes.ravel()[ci]
        m = train_mask & (raw.labels == ci)
        ax.hist(raw.images[m].ravel(), bins=64, range=(0, 255))
        ax.set_title(CLASSES[ci])
    axes.ravel()[7].axis("off")
    fig.suptitle("pixel intensity histogram")
    _save(fig, "pixel_histogram.png")

    # Per-image variance
    var = raw.images.reshape(len(raw.images), -1).astype(np.float32).var(axis=1)
    fig, axes = plt.subplots(1, 2)
    axes[0].hist(np.log10(var + 1e-3), bins=80)
    axes[0].set_xlabel("log10(pixel variance)")
    axes[0].set_title("variance histogram")
    axes[1].boxplot([var[raw.labels == c] for c in range(7)],
                    tick_labels=CLASSES, showfliers=False)
    axes[1].set_title("variance by class")
    _save(fig, "pixel_variance.png")

    # PCA eigenfaces
    from sklearn.decomposition import PCA
    train_imgs = raw.images[train_mask]
    X = train_imgs.reshape(len(train_imgs), -1).astype(np.float32) / 255.0
    pca = PCA(n_components=12, random_state=SEED).fit(X)
    fig, axes = plt.subplots(2, 6)
    for k, ax in enumerate(axes.ravel()):
        ax.imshow(pca.components_[k].reshape(48, 48), cmap="gray")
        ax.set_title(f"PC{k+1}")
        ax.axis("off")
    fig.suptitle("PCA eigenfaces")
    _save(fig, "eigenfaces.png")


def _per_image_stats(raw: RawSplits) -> pd.DataFrame:
    rows = []
    for gi, img in enumerate(raw.images):
        flat = img.ravel().astype(np.int64)
        rows.append({
            "global_index": gi,
            "split":        raw.split[gi],
            "split_index":  int(raw.split_index[gi]),
            "class_id":     int(raw.labels[gi]),
            "class_name":   CLASSES[int(raw.labels[gi])],
            "pixel_mean":   float(flat.mean()),
            "pixel_var":    float(flat.var()),
            "black_ratio":  float((flat == 0).mean()),
            "white_ratio":  float((flat == 255).mean()),
            "sha1":         hashlib.sha1(img.tobytes()).hexdigest(),
        })
    df = pd.DataFrame(rows)
    df["is_corrupt"]         = (df.pixel_var == 0).astype(int)
    df["is_black_saturated"] = (df.black_ratio >= 0.5).astype(int)
    df["is_white_saturated"] = (df.white_ratio >= 0.5).astype(int)
    df["is_lowvar"]          = (df.pixel_var < 50).astype(int)
    return df


def run_cleaning(raw: RawSplits, df_per: Optional[pd.DataFrame] = None) -> dict:
    if df_per is None:
        df_per = _per_image_stats(raw)

    groups = df_per.groupby("sha1")["global_index"].apply(list).reset_index()
    groups["group_size"]     = groups["global_index"].apply(len)
    groups = groups[groups.group_size > 1].copy()
    groups["class_ids"]      = groups["global_index"].apply(
        lambda gs: [int(df_per.loc[g, "class_id"]) for g in gs])
    groups["class_names"]    = groups["class_ids"].apply(lambda cs: [CLASSES[c] for c in cs])
    groups["label_conflict"] = groups["class_ids"].apply(lambda cs: int(len(set(cs)) > 1))
    redundant = sorted({gi for grp in groups.global_index for gi in grp[1:]})

    df_per["drop_dup"] = df_per["global_index"].isin(redundant).astype(int)
    df_per["keep"] = (~(df_per.is_corrupt.astype(bool) |
                        df_per.is_black_saturated.astype(bool) |
                        df_per.is_white_saturated.astype(bool) |
                        df_per.drop_dup.astype(bool))).astype(int)

    summary = {
        "total_images":              int(len(df_per)),
        "corrupt":                   int(df_per.is_corrupt.sum()),
        "black_saturated":           int(df_per.is_black_saturated.sum()),
        "white_saturated":           int(df_per.is_white_saturated.sum()),
        "duplicate_groups":          int(len(groups)),
        "duplicate_redundant":int(len(redundant)),
        "label_conflict——groups":     int(groups.label_conflict.sum()),
        "keet_after_cleaning":       int(df_per.keep.sum()),
        "removed_total":             int((df_per.keep == 0).sum()),
    }
    df_per.to_csv(TABLES_DIR / "image_quality_per_image.csv", index=False)
    groups.to_csv(TABLES_DIR / "duplicate_groups.csv", index=False)
    (TABLES_DIR / "cleaning_summary.json").write_text(json.dumps(summary, indent=2))
    return {"df_per": df_per, "duplicates": groups, "summary": summary}

#clean dataset
def _gallery(raw: RawSplits, indices, savename, n_cols=10, suptitle=None):
    indices = list(indices)[: n_cols * 3]
    if not indices:
        return
    n_rows = max(1, (len(indices) + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols)
    axes = np.atleast_2d(axes)
    for k, gi in enumerate(indices):
        ax = axes[k // n_cols, k % n_cols]
        ax.imshow(raw.images[int(gi)], cmap="gray")
        ax.axis("off")
    for k in range(len(indices), n_rows * n_cols):
        axes[k // n_cols, k % n_cols].axis("off")
    if suptitle: fig.suptitle(suptitle)
    _save(fig, savename)


def visualise_cleaning(raw: RawSplits, clean_out: dict) -> None:
    df_per   = clean_out["df_per"]
    dups     = clean_out["duplicates"]
    summary  = clean_out["summary"]

    fig, axes = plt.subplots(1, 2)
    axes[0].hist(df_per.black_ratio, bins=80)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("black_ratio")
    axes[0].set_title("black ratio distribution")
    axes[1].scatter(df_per.pixel_mean, df_per.black_ratio)
    axes[1].set_xlabel("pixel_mean")
    axes[1].set_ylabel("black_ratio")
    axes[1].set_title("black ratio vs pixel mean")
    _save(fig, "black_ratio.png")

    blk = df_per[df_per.is_black_saturated == 1].global_index.values
    wht = df_per[df_per.is_white_saturated == 1].global_index.values
    lv  = df_per[df_per.is_lowvar == 1].sort_values("pixel_var").global_index.values
    _gallery(raw, blk, "black_saturated.png", suptitle="black saturated")
    _gallery(raw, wht, "white_saturated.png", suptitle="white saturated")
    _gallery(raw, lv,  "low_variance.png",    suptitle="low variance")

    fig, ax = plt.subplots()
    ax.hist(dups.group_size, bins=range(2, int(dups.group_size.max()) + 2))
    ax.set_yscale("log")
    ax.set_xlabel("duplicate group size")
    ax.set_title("duplicate group sizes")
    _save(fig, "duplicate_sizes.png")

    conflicts = dups[dups.label_conflict == 1].sort_values("group_size", ascending=False).head(4)
    if len(conflicts):
        n_show = min(8, int(conflicts.group_size.max()))
        fig, axes = plt.subplots(len(conflicts), n_show)
        axes = np.atleast_2d(axes)
        for r, (_, row) in enumerate(conflicts.iterrows()):
            members = row.global_index[:n_show]
            classes = row.class_names[:n_show]
            for c, (gi, cn) in enumerate(zip(members, classes)):
                axes[r, c].imshow(raw.images[gi], cmap="gray")
                axes[r, c].axis("off")
                axes[r, c].set_title(cn)
            for c in range(len(members), n_show):
                axes[r, c].axis("off")
        fig.suptitle("duplicate label conflicts")
        _save(fig, "duplicate_conflicts.png")

    totals = {"Total": summary["total_images"],
              "Kept":  summary["kept_after_cleaning"],
              "Removed": summary["removed_total"]}
    causes = {"duplicate":       summary["duplicate_redundant_images"],
              "black-saturated": summary["black_saturated"],
              "white-saturated": summary["white_saturated"],
              "corrupt":         summary["corrupt"]}
    fig, axes = plt.subplots(1, 2)
    axes[0].bar(list(totals), list(totals.values()))
    axes[0].set_title("cleaning summary")
    axes[1].bar(list(causes), list(causes.values()))
    axes[1].set_title("removal causes")
    _save(fig, "cleaning_summary.png")

    train_mask = raw.split == "train"
    raw_counts = [int(((raw.labels == c) & train_mask).sum()) for c in range(7)]
    kept = [int(((df_per.split == "train") & (df_per.class_id == c) & (df_per.keep == 1)).sum())
            for c in range(7)]
    removed = [r - k for r, k in zip(raw_counts, kept)]
    x = np.arange(7)
    fig, axes = plt.subplots(1, 2)
    axes[0].plot(x, raw_counts, marker="o", label="raw")
    axes[0].plot(x, kept,        marker="o", label="cleaned")
    axes[0].set_xticks(x); axes[0].set_xticklabels(CLASSES)
    axes[0].set_title("class counts before vs after")
    axes[0].legend()
    axes[1].bar(x, removed)
    axes[1].set_xticks(x); axes[1].set_xticklabels(CLASSES)
    axes[1].set_title("removed per class")
    _save(fig, "removed_per_class.png")


def materialise_clean(raw: RawSplits, clean_out: dict) -> None:
    df_per = clean_out["df_per"]
    for sp in ("train", "val", "test"):
        out_pt = PROCESSED / f"{sp}_clean.pt"
        if out_pt.exists():
            continue
        keep_mask = (df_per.split == sp) & (df_per.keep == 1)
        keep_idx  = df_per.loc[keep_mask, "split_index"].astype(int).values
        m = raw.split == sp
        imgs = raw.images[m][keep_idx]
        lbls = raw.labels[m][keep_idx]
        torch.save({"images": torch.from_numpy(imgs).to(torch.uint8),
                    "labels": torch.from_numpy(lbls).to(torch.long)}, out_pt)
        print(f"wrote {out_pt}  N={len(keep_idx)}")


def build_loaders_from_clean(batch_size: int = 128, num_workers: int = 4):
    """Return (train, val, test) DataLoader trio over the cleaned splits."""
    import sys
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.data.datasets import build_loaders
    from src.utils.config  import load_yaml

    cfg = load_yaml(str(PROJECT_ROOT / "configs/base.yaml"))
    cfg["data"]["batch_size"]    = batch_size
    cfg["data"]["processed_dir"] = str(PROCESSED)
    cfg["num_workers"]           = num_workers
    return build_loaders(processed_dir=cfg["data"]["processed_dir"],
                         cfg=cfg, use_weighted_sampler=False)


def main():
    raw = load_raw_splits()
    print(f"raw images: {len(raw.images)}")

    explore_dataset(raw)
    print("exploration figures saved")

    clean = run_cleaning(raw)
    print(f"cleaning summary: {clean['summary']}")

    visualise_cleaning(raw, clean)
    print("cleaning figures saved")

    materialise_clean(raw, clean)

    train_loader, val_loader, test_loader, info = build_loaders_from_clean()
    print(f"loaders: train={info['train_size']} val={info['val_size']} test={info['test_size']}")
    print(f"batch shape: {next(iter(train_loader))[0].shape}")

if __name__ == "__main__":
    main()
