"""
This script generates pca-umap plots for evaluating the data distribution along positive-negative and train-test
"""
import os
import argparse
import warnings
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity

# UMAP
try:
    import umap
    HAS_UMAP = True
except Exception as e:
    HAS_UMAP = False
    _UMAP_ERR = e


# =========================
# CSVSequenceDataset
# =========================
class CSVSequenceDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.samples = self._load_samples(dataframe)

    def _load_samples(self, dataframe: pd.DataFrame):
        samples = []
        for _, row in dataframe.iterrows():
            code = self.extract_id(row["ID"])
            label = float(row["LABEL"])
            samples.append((code, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def extract_id(self, id_string):
        return str(id_string).split("_")[0]

    def __getitem__(self, idx):
        code, label = self.samples[idx]
        return code, torch.tensor(label, dtype=torch.float32)


# =========================
# Pooling helpers
# =========================
def mean_pool(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=0)

def median_pool(x: torch.Tensor) -> torch.Tensor:
    return x.median(dim=0).values


# =========================
# Embedding loading
# =========================
def load_pooled_embeddings(
    dataset: CSVSequenceDataset,
    embedding_path: str,
    pooling_fn,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Loads embeddings from embedding_path/{code}.pt.
    Applies pooling_fn if LxD, accepts D vectors as-is.
    Returns: ids_kept, X (n,d), y (n,)
    """
    ids, vecs, labels = [], [], []
    missing = 0

    for i in range(len(dataset)):
        code, label_tensor = dataset[i]
        code = str(code)
        p = os.path.join(embedding_path, f"{code}.pt")
        if not os.path.exists(p):
            missing += 1
            continue

        emb = torch.load(p, map_location="cpu")
        if emb.ndim == 2:
            v = pooling_fn(emb).detach().cpu().numpy()
        elif emb.ndim == 1:
            v = emb.detach().cpu().numpy()
        else:
            raise ValueError(f"Unexpected embedding shape for {code}: {tuple(emb.shape)}")

        ids.append(code)
        vecs.append(v)
        labels.append(int(label_tensor.item()))

    if missing > 0:
        warnings.warn(f"{missing} embeddings were missing under {embedding_path} and were skipped.")

    if len(vecs) == 0:
        raise ValueError(f"No embeddings found in {embedding_path} for any item in the dataset.")

    X = np.stack(vecs, axis=0)
    y = np.asarray(labels, dtype=int)
    return ids, X, y


# =========================
# PCA50 -> UMAP2 utility
# =========================
def pca50_umap2(
    X_train: np.ndarray, X_test: np.ndarray,
    fit_pca_on: str = "all",  # "all" or "train"
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduce to 50 dims with PCA, then to 2D with UMAP.
    Returns Z_train (n_train,2), Z_test (n_test,2).
    """
    if not HAS_UMAP:
        raise ImportError(f"umap-learn is required but couldn't be imported: {_UMAP_ERR}")

    # PCA to 50
    pca = PCA(n_components=min(50, min(X_train.shape[1], X_test.shape[1])))
    if fit_pca_on == "train":
        pca.fit(X_train)
        Xtr_50 = pca.transform(X_train)
        Xte_50 = pca.transform(X_test)
    else:
        X_all = np.vstack([X_train, X_test])
        pca.fit(X_all)
        Xtr_50 = pca.transform(X_train)
        Xte_50 = pca.transform(X_test)

    # UMAP to 2
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
        verbose=False,
    )
    Z_all = reducer.fit_transform(np.vstack([Xtr_50, Xte_50]).astype(np.float32))
    Z_train = Z_all[:len(Xtr_50)]
    Z_test = Z_all[len(Xtr_50):]
    return Z_train, Z_test


# =========================
# Density helpers
# =========================

def _silverman_bandwidth(Z: np.ndarray) -> float:
    """Silverman's rule of thumb for Gaussian KDE (2D)."""
    n, d = Z.shape
    if n < 2:
        return 1.0
    sigma = float(np.mean(Z.std(axis=0)))
    factor = (4.0 / (d + 2)) ** (1.0 / (d + 4)) * (n ** (-1.0 / (d + 4)))
    bw = max(1e-3, sigma * factor)
    return bw


def _pointwise_density(Z: np.ndarray, bandwidth: float | None = None) -> np.ndarray:
    if Z.size == 0:
        return np.array([])
    if bandwidth is None:
        bandwidth = _silverman_bandwidth(Z)
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(Z)
    log_d = kde.score_samples(Z)
    d = np.exp(log_d)
    # Normalize to [0,1] within this class to make gradients comparable
    d_min, d_max = float(d.min()), float(d.max())
    d_norm = (d - d_min) / (d_max - d_min + 1e-12)
    return d_norm


# =========================
# CSV + classic plots
# =========================

def save_pca50_umap2_plots_and_csv(
    ids_train: List[str], X_train: np.ndarray, y_train: np.ndarray,
    ids_test: List[str],  X_test: np.ndarray,  y_test: np.ndarray,
    out_prefix: str,
    pca_fit_on: str = "all",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    seed: int = 42,
):
    # Compute embeddings
    Z_tr, Z_te = pca50_umap2(
        X_train, X_test,
        fit_pca_on=pca_fit_on,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        seed=seed,
    )

    # Save coordinates to CSV
    df_coords = pd.DataFrame({
        "ID": ids_train + ids_test,
        "SPLIT": ["TRAIN"] * len(ids_train) + ["TEST"] * len(ids_test),
        "LABEL": list(y_train.astype(int)) + list(y_test.astype(int)),
        "UMAP1": np.concatenate([Z_tr[:, 0], Z_te[:, 0]]),
        "UMAP2": np.concatenate([Z_tr[:, 1], Z_te[:, 1]]),
    })
    df_coords.to_csv(f"{out_prefix}_coords.csv", index=False)

    # Plot 1: color by SPLIT (Train/Test)
    plt.figure(figsize=(7.2, 6.2))
    plt.scatter(Z_tr[:, 0], Z_tr[:, 1], s=18, alpha=0.85, label="Train")
    plt.scatter(Z_te[:, 0], Z_te[:, 1], s=22, alpha=0.85, label="Test")
    plt.title("PCA(50) ? UMAP(2): colored by split")
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_split.png", dpi=220)
    plt.close()

    # Plot 2: color by LABEL (two density gradients, no shapes, all points)
    # Combine train+test for density per class
    Z_all = np.vstack([Z_tr, Z_te])
    y_all = np.concatenate([y_train, y_test])

    cls0 = (y_all == 0)
    cls1 = (y_all == 1)
    Z0 = Z_all[cls0]
    Z1 = Z_all[cls1]

    d0 = _pointwise_density(Z0)
    d1 = _pointwise_density(Z1)

    plt.figure(figsize=(7.6, 6.4))
    sc0 = plt.scatter(Z0[:, 0], Z0[:, 1], s=10, alpha=0.95, c=d0, cmap="Blues", label="Class 0 density")
    sc1 = plt.scatter(Z1[:, 0], Z1[:, 1], s=10, alpha=0.95, c=d1, cmap="Reds", label="Class 1 density")
    plt.title("PCA(50) ? UMAP(2): label density (0 = Blues, 1 = Reds)")
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")

    # Two compact colorbars (one per class)
    cbar0 = plt.colorbar(sc0, shrink=0.8)
    cbar0.set_label("Class 0 density (rel.)")
    cbar1 = plt.colorbar(sc1, shrink=0.8)
    cbar1.set_label("Class 1 density (rel.)")

    plt.tight_layout()
    plt.savefig(f"{out_prefix}_label.png", dpi=240)
    plt.close()



# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(description="PCA50?UMAP-2D visualizations with density coloring")
    parser.add_argument("--csv_path_train", type=str, required=True, help="Path to training CSV (must have ID,LABEL)")
    parser.add_argument("--csv_path_test",  type=str, required=True, help="Path to testing CSV (must have ID,LABEL)")
    parser.add_argument("--embedding_path", type=str, required=True, help="Directory with {ID}.pt embeddings")
    parser.add_argument("--output_dir",     type=str, required=True, help="Directory to save plots/CSVs")
    parser.add_argument("--pca_fit_on",     type=str, default="all", choices=["all", "train"],
                        help="Fit PCA on 'all' (train+test) or only 'train' (default: all)")
    parser.add_argument("--umap_n_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist",    type=float, default=0.1)
    parser.add_argument("--umap_metric",      type=str, default="euclidean")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not HAS_UMAP:
        raise ImportError(f"umap-learn couldn't be imported: {_UMAP_ERR}\nInstall with: pip install umap-learn")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load CSVs
    df_train = pd.read_csv(args.csv_path_train).dropna(subset=["LABEL", "ID"])
    df_test  = pd.read_csv(args.csv_path_test).dropna(subset=["LABEL", "ID"])

    # Build datasets
    train_ds = CSVSequenceDataset(df_train)
    test_ds  = CSVSequenceDataset(df_test)

    # Load pooled embeddings for mean & median
    ids_tr_mean, X_tr_mean, y_tr = load_pooled_embeddings(train_ds, args.embedding_path, pooling_fn=mean_pool)
    ids_te_mean, X_te_mean, y_te = load_pooled_embeddings(test_ds,  args.embedding_path, pooling_fn=mean_pool)

    ids_tr_med,  X_tr_med,  y_tr_check = load_pooled_embeddings(train_ds, args.embedding_path, pooling_fn=median_pool)
    ids_te_med,  X_te_med,  y_te_check = load_pooled_embeddings(test_ds,  args.embedding_path, pooling_fn=median_pool)

    # Sanity: labels match across pooling
    assert np.array_equal(y_tr, y_tr_check), "Train labels mismatch between mean and median loads."
    assert np.array_equal(y_te, y_te_check), "Test labels mismatch between mean and median loads."

    # === PCA50?UMAP2 plots & coords (MEAN) ===
    out_prefix_mean = os.path.join(args.output_dir, "pca50_umap2_mean")
    save_pca50_umap2_plots_and_csv(
        ids_tr_mean, X_tr_mean, y_tr,
        ids_te_mean, X_te_mean, y_te,
        out_prefix=out_prefix_mean,
        pca_fit_on=args.pca_fit_on,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        seed=args.seed,
    )

    # === PCA50?UMAP2 plots & coords (MEDIAN) ===
    out_prefix_median = os.path.join(args.output_dir, "pca50_umap2_median")
    save_pca50_umap2_plots_and_csv(
        ids_tr_med, X_tr_med, y_tr,
        ids_te_med, X_te_med, y_te,
        out_prefix=out_prefix_median,
        pca_fit_on=args.pca_fit_on,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        seed=args.seed,
    )

    print("Saved:")
    for prefix in (out_prefix_mean, out_prefix_median):
        print(" -", prefix + "_split.png")
        print(" -", prefix + "_label.png")
        print(" -", prefix + "_coords.csv")


if __name__ == "__main__":
    main()
