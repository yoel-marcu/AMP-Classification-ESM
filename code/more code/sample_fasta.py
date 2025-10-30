"""
This script was used to randomly sample the unknown-function fasta.
"""
import sys
import random
from pathlib import Path

def read_fasta(filename):
    """Read FASTA file into a list of (header, sequence) tuples."""
    records = []
    header = None
    seq_lines = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if header:
                    records.append((header, "".join(seq_lines)))
                header = line
                seq_lines = []
            else:
                seq_lines.append(line)

        if header:
            records.append((header, "".join(seq_lines)))

    return records

def write_fasta(records, output_file):
    """Write a list of (header, sequence) tuples to a FASTA file."""
    with open(output_file, 'w') as f:
        for header, seq in records:
            f.write(f"{header}\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + "\n")

def main(input_fasta):
    records = read_fasta(input_fasta)
    sample_size = max(1, int(len(records) * 0.010))  

    sampled_records = random.sample(records, sample_size)

    input_path = Path(input_fasta)
    output_fasta = input_path.with_name(
        input_path.stem + "_random1percentSampling.fasta"
    )

    write_fasta(sampled_records, output_fasta)
    print(f"? Done! Wrote {sample_size} sequences to: {output_fasta}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python sample_fasta_1percent.py input.fasta")
        sys.exit(1)
    main(sys.argv[1])
