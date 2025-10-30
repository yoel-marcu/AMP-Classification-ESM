import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

EMBED_SIZES = {
    "esm2_t6": 320,
    "esm2_t30": 640,
    "esm2_t36": 2560,
    "esmc_300m": 960,
    "esmc_600m": 1152
}

def _auto_zoom_limits(ax, x_vals, y_vals,
                      x_clip=(0.0, 1.0), y_clip=None,
                      pad=0.10, min_span_x=1e-3, min_span_y=1e-3):
    x = pd.to_numeric(pd.Series(x_vals), errors="coerce").dropna().to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(y_vals), errors="coerce").dropna().to_numpy(dtype=float)
    if x.size == 0 or y.size == 0:
        return

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    if x_max - x_min < min_span_x:
        c = 0.5 * (x_min + x_max)
        x_min, x_max = c - min_span_x/2, c + min_span_x/2
    if y_max - y_min < min_span_y:
        c = 0.5 * (y_min + y_max)
        y_min, y_max = c - min_span_y/2, c + min_span_y/2

    px = (x_max - x_min) * pad
    py = (y_max - y_min) * pad
    x_lo, x_hi = x_min - px, x_max + px
    y_lo, y_hi = y_min - py, y_max + py

    if x_clip is not None:
        x_lo = max(x_clip[0], x_lo)
        x_hi = min(x_clip[1], x_hi)
    if y_clip is not None:
        if y_clip[0] is not None:
            y_lo = max(y_clip[0], y_lo)
        if y_clip[1] is not None:
            y_hi = min(y_clip[1], y_hi)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)


def plot_pr_by_embedding(df, out_path):
    fig, ax = plt.subplots(figsize=(9, 7))

    embeds = sorted(
        df["embedding_name"].dropna().unique().tolist(),
        key=lambda e: EMBED_SIZES.get(e, float("inf"))
    )

    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9'])
    while len(colors) < len(embeds):
        colors += colors
    markers = ['o', 's', '^', 'D', 'P', 'X', 'v', '<', '>']
    while len(markers) < len(embeds):
        markers += markers

    handles = []
    for i, emb in enumerate(embeds):
        sub = df[df["embedding_name"] == emb]
        ax.scatter(sub["Final Precision"], sub["Final Recall"],
                   s=80, marker=markers[i], facecolors=colors[i], edgecolors='none', alpha=0.8,
                   label=f"{emb} ({EMBED_SIZES.get(emb, '?')})")
        handles.append(Line2D([0],[0], marker=markers[i], linestyle='', markerfacecolor=colors[i],
                              markeredgecolor='none', label=f"{emb} ({EMBED_SIZES.get(emb, '?')})", markersize=9))

    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")
    ax.set_title("Precision–Recall by ESM Version (pooling/loss/network ignored)")
    ax.grid(True, alpha=0.3)

    _auto_zoom_limits(ax,
        x_vals=df['Final Precision'].values,
        y_vals=df['Final Recall'].values,
        x_clip=(0.0, 1.0), y_clip=(0.0, 1.0), pad=0.20,
        min_span_x=0.02, min_span_y=0.02)

    ax.legend(handles=handles, title="Embedding (size)", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_mean_precision_vs_std_by_embedding(df, out_path):
    fig, ax = plt.subplots(figsize=(9, 7))

    embeds = sorted(
        df["embedding_name"].dropna().unique().tolist(),
        key=lambda e: EMBED_SIZES.get(e, float("inf"))
    )

    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9'])
    while len(colors) < len(embeds):
        colors += colors
    markers = ['o', 's', '^', 'D', 'P', 'X', 'v', '<', '>']
    while len(markers) < len(embeds):
        markers += markers

    handles = []
    x_vals_all = df['Mean CV Precision'].where(~df['Mean CV Precision'].isna(), df['Final Precision']).values
    for i, emb in enumerate(embeds):
        sub = df[df["embedding_name"] == emb].copy()
        x_vals = sub['Mean CV Precision'].where(~sub['Mean CV Precision'].isna(), sub['Final Precision']).values
        y_vals = sub['Std CV Precision'].fillna(0.0).values
        ax.scatter(x_vals, y_vals, s=80, marker=markers[i], facecolors=colors[i], edgecolors='none', alpha=0.8,
                   label=f"{emb} ({EMBED_SIZES.get(emb, '?')})")
        handles.append(Line2D([0],[0], marker=markers[i], linestyle='', markerfacecolor=colors[i],
                              markeredgecolor='none', label=f"{emb} ({EMBED_SIZES.get(emb, '?')})", markersize=9))

    ax.set_xlabel("Mean CV Precision")
    ax.set_ylabel("Std CV Precision")
    ax.set_title("Mean Precision vs CV Std by ESM Version (pooling/loss/network ignored)")
    ax.grid(True, alpha=0.3)

    _auto_zoom_limits(ax,
        x_vals=x_vals_all,
        y_vals=df['Std CV Precision'].fillna(0.0).values,
        x_clip=(0.0, 1.0), y_clip=(0.0, None), pad=0.20,
        min_span_x=0.02, min_span_y=0.005)

    ax.legend(handles=handles, title="Embedding (size)", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main(csv_path: str, outdir: str):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    for c in ["Final Precision","Final Recall","embedding_name","Mean CV Precision","Std CV Precision"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: '{c}'")

    pr_path = outdir / "pr_by_embedding.png"
    ps_path = outdir / "mean_precision_vs_std_by_embedding.png"

    plot_pr_by_embedding(df, pr_path)
    plot_mean_precision_vs_std_by_embedding(df, ps_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare ESM versions only (ignore pooling/loss/network).")
    parser.add_argument("--csv", required=True, help="Path to results CSV")
    parser.add_argument("--outdir", required=True, help="Directory to save plots")
    args = parser.parse_args()
    main(args.csv, args.outdir)
