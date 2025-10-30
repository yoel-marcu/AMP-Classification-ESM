"""
This script plays our BLAST-MIS pipeline in one python script for run on our cluster. Compatible SLURM script is attached in this directory as well.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import random

from Bio import SeqIO
import pandas as pd
import numpy as np
import networkx as nx

# ---------------------------
# Greedy MIS utilities
# ---------------------------

def greedy_mis_from_order(G, order):
    selected, blocked = set(), set()
    for u in order:
        if u not in blocked:
            selected.add(u)
            blocked.add(u)
            blocked.update(G.neighbors(u))
    return selected

def greedy_mis_degree_ascending(G):
    order = sorted(G.nodes(), key=lambda u: G.degree(u))
    return greedy_mis_from_order(G, order)

def greedy_mis_degree_descending(G):
    order = sorted(G.nodes(), key=lambda u: G.degree(u), reverse=True)
    return greedy_mis_from_order(G, order)

def greedy_mis_random(G, nodes, rng):
    order = nodes[:]
    rng.shuffle(order)
    return greedy_mis_from_order(G, order)

# ---------------------------
# BLAST helpers
# ---------------------------

def _which(name):
    p = shutil.which(name)
    if not p:
        raise RuntimeError(f"Required binary '{name}' not found in PATH.")
    return p

def _run(cmd, **kw):
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kw)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"\n[ERROR] Command failed: {' '.join(cmd)}\n")
        sys.stderr.write(e.stderr.decode(errors='replace'))
        raise

def build_db(fasta, dbdir):
    _which("makeblastdb")
    dbbase = os.path.join(dbdir, "selfdb")
    _run(["makeblastdb", "-in", fasta, "-dbtype", "prot", "-out", dbbase])
    return dbbase

def run_self_blastp(fasta, dbbase, threads, out_tsv):
    _which("blastp")
    # Match your previous runs:
    # - outfmt 6 (default 12 columns)
    # - num_threads configurable
    cmd = [
        "blastp",
        "-query", fasta,
        "-db", dbbase,
        "-out", out_tsv,
        "-outfmt", "6",
        "-num_threads", str(threads),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"\n[ERROR] blastp failed.\n")
        sys.stderr.write(e.stderr.decode(errors='replace'))
        raise
    return out_tsv

# ---------------------------
# Graph construction (e-value only)
# ---------------------------

# Default outfmt 6 columns (12):
BLAST6_COLS = [
    "qseqid","sseqid","pident","length","mismatch","gapopen",
    "qstart","qend","sstart","send","evalue","bitscore"
]

def build_graph_from_blast6(tsv_path, evalue_cut=0.1, drop_self_hits=True):
    df = pd.read_csv(tsv_path, sep="\t", header=None, names=BLAST6_COLS)
    if drop_self_hits:
        df = df[df["qseqid"] != df["sseqid"]]
    # *** Only criterion: evalue < 0.1 ***
    df = df[df["evalue"] < evalue_cut]

    # one undirected edge per unordered pair
    u = np.where(df["qseqid"] < df["sseqid"], df["qseqid"], df["sseqid"])
    v = np.where(df["qseqid"] < df["sseqid"], df["sseqid"], df["qseqid"])
    pairs = pd.DataFrame({"u": u, "v": v}).drop_duplicates()

    G = nx.Graph()
    G.add_edges_from(map(tuple, pairs.to_numpy()))
    # include isolated nodes observed anywhere
    ids = set(df["qseqid"]).union(set(df["sseqid"]))
    G.add_nodes_from(ids)
    return G

# ---------------------------
# Main pipeline
# ---------------------------

def run_pipeline(
    fasta,
    out_fasta,
    out_report,
    threads=4,
    trials=5000,
    seed=42,
    evalue_cut=0.1,
    blast_bin_dir=None,
):
    # Optionally prepend BLAST bin dir to PATH (like your SLURM script)
    if blast_bin_dir:
        os.environ["PATH"] = blast_bin_dir + os.pathsep + os.environ.get("PATH", "")

    # Read sequences
    records = list(SeqIO.parse(fasta, "fasta"))
    if not records:
        raise RuntimeError("Input FASTA has no sequences.")
    id2rec = {r.id: r for r in records}

    tmpdir = tempfile.mkdtemp(prefix="mis_evalue_only_")
    try:
        dbbase = build_db(fasta, tmpdir)
        tsv = os.path.join(tmpdir, "self_vs_self.tsv")
        run_self_blastp(fasta, dbbase, threads, tsv)

        G = build_graph_from_blast6(tsv, evalue_cut=evalue_cut, drop_self_hits=True)
        # Ensure all input IDs are present even if no hits
        for r in records:
            if r.id not in G:
                G.add_node(r.id)

        # Heuristics  randomized MIS (like your approach)
        mis_asc = greedy_mis_degree_ascending(G)
        mis_desc = greedy_mis_degree_descending(G)
        best_heur = mis_asc if len(mis_asc) >= len(mis_desc) else mis_desc
        best = set(best_heur)

        rng = random.Random(seed)
        nodes = list(G.nodes())
        sizes = []
        for _ in range(trials):
            mis = greedy_mis_random(G, nodes, rng)
            sizes.append(len(mis))
            if len(mis) > len(best):
                best = mis

        # Write nonredundant FASTA
        with open(out_fasta, "w") as fh:
            for sid in sorted(best):
                SeqIO.write(id2rec[sid], fh, "fasta")

        # Tiny report
        comps = [len(c) for c in nx.connected_components(G)]
        comps.sort(reverse=True)
        report = {
            "input_fasta": os.path.abspath(fasta),
            "output_fasta": os.path.abspath(out_fasta),
            "blast": {
                "program": "blastp",
                "outfmt": 6,
                "num_threads": threads,
                "dbtype": "prot",
                "evalue_edge_rule": "< 0.1 (only criterion)",
            },
            "graph": {
                "num_nodes": G.number_of_nodes(),
                "num_edges": G.number_of_edges(),
                "num_components": len(comps),
                "largest_component_size": (comps[0] if comps else 0),
            },
            "mis": {
                "trials": trials,
                "best_size": len(best),
                "degree_asc_size": len(mis_asc),
                "degree_desc_size": len(mis_desc),
                "trial_size_mean": (float(np.mean(sizes)) if sizes else None),
            },
        }
        with open(out_report, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Nonredundant FASTA: {os.path.abspath(out_fasta)}")
        print(f"Report JSON       : {os.path.abspath(out_report)}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# ---------------------------
# CLI
# ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="MIS deduplication using only e-value < 0.1 as edge rule (blastp self vs self).")
    ap.add_argument("--fasta", required=True, help="Input FASTA")
    ap.add_argument("--out_fasta", required=True, help="Output deduplicated FASTA")
    ap.add_argument("--out_report", required=True, help="Output JSON report")
    ap.add_argument("--threads", type=int, default=4, help="Threads for blastp (default 4)")
    ap.add_argument("--trials", type=int, default=5000, help="Randomized MIS trials")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--evalue_cut", type=float, default=0.1, help="Edge rule: connect if evalue < this (default 0.1)")
    ap.add_argument("--blast_bin_dir", default=None, help="Optional path to BLAST+ bin (prepends to PATH)")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        fasta=args.fasta,
        out_fasta=args.out_fasta,
        out_report=args.out_report,
        threads=args.threads,
        trials=args.trials,
        seed=args.seed,
        evalue_cut=args.evalue_cut,
        blast_bin_dir=args.blast_bin_dir,
    )
