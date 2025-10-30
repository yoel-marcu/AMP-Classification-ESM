#!/bin/bash
#SBATCH --job-name=mis_blast
#SBATCH --output=mis_blast.%j.out
#SBATCH --error=mis_blast.%j.err
#SBATCH --time=90:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=8G

# ============================
# User-configurable parameters
# ============================

# Input / Output
FASTA="uniprot_unknown_50_150_refprot_random1percentSampling.fasta"
OUT_FASTA="uniprot_unknown_50_150_refprot_random1percentSampling_after_MISBLAST.fasta"
OUT_REPORT="uniprot_unknown_50_150_refprot_random1percentSampling_after_MISBLAST.json"

# Script & Python
SCRIPT="MaximalIndependentSetBLAST.py"
PYTHON="python"  # or absolute path to your Python if needed

# BLAST binaries (same location you used before)
BLAST_BIN="/sci/labs/asafle/alexlevylab/icore-data/tools/ncbi-blast-2.10.0+/bin"

# MIS / BLAST options
TRIALS=1000         # randomized MIS trials
SEED=42
EVAL_CUT=0.1        # edge rule: connect if evalue < 0.1

# ============================
# No edits typically required
# ============================

set -euo pipefail

# Make BLAST+ available (matches your previous runs)
export PATH="$BLAST_BIN:$PATH"

# Use all CPUs allocated by SLURM unless overridden
THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "===== Job metadata ====="
echo "Host:        $(hostname)"
echo "Start time:  $(date)"
echo "CPUs:        ${THREADS}"
echo "Mem (req):   ${SLURM_MEM_PER_NODE:-$SLURM_MEM_PER_CPU} per node"
echo "========================"

# Optional: show BLAST+ versions for provenance
echo "makeblastdb: $(makeblastdb -version | head -n1 || echo 'not found')"
echo "blastp:      $(blastp -version | head -n1 || echo 'not found')"

source /sci/labs/asafle/yoel.marcu2003/myenv/bin/activate


# Run the MIS-based deduplication
$PYTHON "$SCRIPT" \
  --fasta "$FASTA" \
  --out_fasta "$OUT_FASTA" \
  --out_report "$OUT_REPORT" \
  --threads "$THREADS" \
  --trials "$TRIALS" \
  --seed "$SEED" \
  --evalue_cut "$EVAL_CUT" \
  --blast_bin_dir "$BLAST_BIN"

echo "========================"
echo "Done at:     $(date)"
echo "Output FASTA: $(readlink -f "$OUT_FASTA" || echo "$OUT_FASTA")"
echo "Report JSON:  $(readlink -f "$OUT_REPORT" || echo "$OUT_REPORT")"
