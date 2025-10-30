"""
This script aims to analyze the charge and hydrophobicity distribution over the positive and negative sets.
Uses mainly methods that were found online (not orogonal code of mine)
"""
import os
import re
import argparse
import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

PKA = {
    "N_TERMINUS": 8.6,
    "C_TERMINUS": 3.6,
    "K": 10.54, "R": 12.48, "H": 6.04,
    "D": 3.90, "E": 4.07, "C": 8.18, "Y": 10.46,
}

STD_AA = set("ACDEFGHIKLMNPQRSTVWY")

def clean_seq(seq: str) -> str:
    s = str(seq).upper()
    s = re.sub(r"[^A-Z]", "", s)
    return "".join(ch for ch in s if ch in STD_AA)

def _frac_pos(pH: float, pKa: float) -> float:
    # Protonated fraction for bases (positive when protonated)
    return 1.0 / (1.0 + 10.0 ** (pH - pKa))

def _frac_neg(pH: float, pKa: float) -> float:
    # Deprotonated fraction for acids (negative when deprotonated)
    return 1.0 / (1.0 + 10.0 ** (pKa - pH))

def net_charge_at_pH(seq: str, pH: float) -> float:
    seq = clean_seq(seq)
    if not seq:
        return np.nan
    q_nterm = _frac_pos(pH, PKA["N_TERMINUS"])
    q_cterm = _frac_neg(pH, PKA["C_TERMINUS"])
    nK, nR, nH = seq.count("K"), seq.count("R"), seq.count("H")
    q_pos = nK * _frac_pos(pH, PKA["K"]) + nR * _frac_pos(pH, PKA["R"]) + nH * _frac_pos(pH, PKA["H"])
    nD, nE, nC, nY = seq.count("D"), seq.count("E"), seq.count("C"), seq.count("Y")
    q_neg = nD * _frac_neg(pH, PKA["D"]) + nE * _frac_neg(pH, PKA["E"]) + nC * _frac_neg(pH, PKA["C"]) + nY * _frac_neg(pH, PKA["Y"])
    return (q_nterm + q_pos) - (q_cterm + q_neg)

def cohen_d(x0, x1):
    x0 = np.asarray(x0, float); x1 = np.asarray(x1, float)
    if len(x0) < 2 or len(x1) < 2: return np.nan
    m0, m1 = x0.mean(), x1.mean()
    v0, v1 = x0.var(ddof=1), x1.var(ddof=1)
    sp = np.sqrt(((len(x0)-1)*v0 + (len(x1)-1)*v1) / (len(x0)+len(x1)-2))
    return (m1 - m0) / sp if sp > 0 else np.nan

def cliffs_delta(x0, x1):
    x0 = np.asarray(x0); x1 = np.asarray(x1)
    n0, n1 = len(x0), len(x1)
    if n0 == 0 or n1 == 0: return np.nan
    x0s, x1s = np.sort(x0), np.sort(x1)
    i = j = more = less = 0
    while i < n0 and j < n1:
        if x1s[j] > x0s[i]:
            more += (n1 - j); i += 1
        elif x1s[j] < x0s[i]:
            less += (n0 - i); j += 1
        else:
            i += 1; j += 1
    return (more - less) / float(n0 * n1)

def summarize_group(df, col):
    arr = df[col].to_numpy(float)
    if len(arr) == 0:
        return dict(n=0, mean=np.nan, std=np.nan, median=np.nan, p25=np.nan, p75=np.nan, min=np.nan, max=np.nan)
    return dict(
        n=int(len(arr)),
        mean=float(np.mean(arr)),
        std=float(np.std(arr, ddof=1)) if len(arr) > 1 else np.nan,
        median=float(np.median(arr)),
        p25=float(np.percentile(arr, 25)),
        p75=float(np.percentile(arr, 75)),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
    )

def compute_props(df: pd.DataFrame, split_name: str, pH: float,
                  col_id: str, col_seq: str, col_label: str) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        sid = r[col_id]; label = int(r[col_label])
        seq = clean_seq(r[col_seq])
        if not seq: continue
        pa = ProteinAnalysis(seq)
        gravy = pa.gravy()
        charge = net_charge_at_pH(seq, pH)
        L = len(seq)
        rows.append({
            "ID": sid, "SPLIT": split_name, "LABEL": label, "LENGTH": L,
            f"CHARGE_pH_{pH:g}": float(charge),
            "NET_CHARGE_PER_RES": float(charge / L),
            "GRAVY": float(gravy),
            "FRAC_KR": (seq.count("K")+seq.count("R"))/L,
            "FRAC_KRH": (seq.count("K")+seq.count("R")+seq.count("H"))/L,
        })
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(description="Net charge at pH and hydrophobicity (GRAVY) vs label.")
    ap.add_argument("--csv_path_train", required=True)
    ap.add_argument("--csv_path_test",  required=True)
    ap.add_argument("--output_dir",     required=True)
    ap.add_argument("--ph", type=float, default=7.5)
    ap.add_argument("--col_id", default="ID")
    ap.add_argument("--col_sequence", default="SEQUENCE")
    ap.add_argument("--col_label", default="LABEL")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df_tr = pd.read_csv(args.csv_path_train).dropna(subset=[args.col_id, args.col_sequence, args.col_label])
    df_te = pd.read_csv(args.csv_path_test ).dropna(subset=[args.col_id, args.col_sequence, args.col_label])

    feats_tr = compute_props(df_tr, "TRAIN", args.ph, args.col_id, args.col_sequence, args.col_label)
    feats_te = compute_props(df_te, "TEST",  args.ph, args.col_id, args.col_sequence, args.col_label)
    feats = pd.concat([feats_tr, feats_te], ignore_index=True)

    per_seq = os.path.join(args.output_dir, f"seq_props_pH{args.ph}.csv")
    feats.to_csv(per_seq, index=False)

    charge_col = f"CHARGE_pH_{args.ph:g}"

    parts = []
    for lab, g in feats.groupby("LABEL"):
        row = {"LABEL": int(lab)}
        for k, v in summarize_group(g, charge_col).items(): row[f"{charge_col}_{k}"] = v
        for k, v in summarize_group(g, "GRAVY").items(): row[f"GRAVY_{k}"] = v
        for k, v in summarize_group(g, "NET_CHARGE_PER_RES").items(): row[f"NET_CHARGE_PER_RES_{k}"] = v
        for k, v in summarize_group(g, "FRAC_KR").items(): row[f"FRAC_KR_{k}"] = v
        for k, v in summarize_group(g, "FRAC_KRH").items(): row[f"FRAC_KRH_{k}"] = v
        parts.append(row)
    pd.DataFrame(parts).sort_values("LABEL").to_csv(
        os.path.join(args.output_dir, f"summary_by_label_pH{args.ph}.csv"), index=False
    )

    parts2 = []
    for (split, lab), g in feats.groupby(["SPLIT", "LABEL"]):
        row = {"SPLIT": split, "LABEL": int(lab)}
        for k, v in summarize_group(g, charge_col).items(): row[f"{charge_col}_{k}"] = v
        for k, v in summarize_group(g, "GRAVY").items(): row[f"GRAVY_{k}"] = v
        parts2.append(row)
    pd.DataFrame(parts2).sort_values(["SPLIT", "LABEL"]).to_csv(
        os.path.join(args.output_dir, f"summary_by_split_label_pH{args.ph}.csv"), index=False
    )

    x0c = feats.loc[feats["LABEL"] == 0, charge_col].to_numpy(float)
    x1c = feats.loc[feats["LABEL"] == 1, charge_col].to_numpy(float)
    x0g = feats.loc[feats["LABEL"] == 0, "GRAVY"].to_numpy(float)
    x1g = feats.loc[feats["LABEL"] == 1, "GRAVY"].to_numpy(float)
    eff = pd.DataFrame([
        {"metric":"charge","pH":args.ph,"cohens_d":cohen_d(x0c,x1c),"cliffs_delta":cliffs_delta(x0c,x1c),
         "mean_label0":float(np.mean(x0c)) if len(x0c) else np.nan,
         "mean_label1":float(np.mean(x1c)) if len(x1c) else np.nan},
        {"metric":"gravy","cohens_d":cohen_d(x0g,x1g),"cliffs_delta":cliffs_delta(x0g,x1g),
         "mean_label0":float(np.mean(x0g)) if len(x0g) else np.nan,
         "mean_label1":float(np.mean(x1g)) if len(x1g) else np.nan},
    ])
    eff.to_csv(os.path.join(args.output_dir, f"effect_size_pH{args.ph}.csv"), index=False)

    print("Saved:")
    print(" -", per_seq)
    print(" -", os.path.join(args.output_dir, f"summary_by_label_pH{args.ph}.csv"))
    print(" -", os.path.join(args.output_dir, f"summary_by_split_label_pH{args.ph}.csv"))
    print(" -", os.path.join(args.output_dir, f"effect_size_pH{args.ph}.csv"))

if __name__ == "__main__":
    main()
