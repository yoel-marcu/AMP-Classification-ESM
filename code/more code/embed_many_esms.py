"""
This script embeds the sequences to the esm models used in the project for the experiments
"""
import json
import os
import argparse
import torch
import pandas as pd

from transformers import AutoTokenizer, EsmModel, EsmForMaskedLM
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig


def esm1_embedding(uniprot_code_to_seq, output_dir, model_name, device):
    """Helper for ESM-1 family embeddings"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name, output_hidden_states=True).to(device)

    for uniprot_code, sequence in uniprot_code_to_seq.items():
        inputs = tokenizer(sequence, return_tensors="pt",
                           truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        torch.save(outputs.hidden_states[-1][0].cpu(),
                   os.path.join(output_dir, f"{uniprot_code}.pt"))
        print(f"made {uniprot_code}.pt")


def esm2_embedding(uniprot_code_to_seq, output_dir, model_name, device):
    """Helper for ESM-2 family embeddings"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name, output_hidden_states=True).to(device)

    for uniprot_code, sequence in uniprot_code_to_seq.items():
        inputs = tokenizer(sequence, return_tensors="pt",
                           truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        torch.save(outputs.last_hidden_state[0].cpu(),
                   os.path.join(output_dir, f"{uniprot_code}.pt"))
        print(f"made {uniprot_code}.pt")


def main():
    parser = argparse.ArgumentParser(description="Embed dataset to different ESM versions")
    parser.add_argument('--data_path', required=True, help="Path to a CSV with columns: ID, SEQUENCE")
    parser.add_argument('--excluded_models', default="",
                        help="Comma-separated list of ESM versions to exclude. Example: esm2_t6_8M_UR50D,esm_msa1b_t12_100M_UR50S")
    parser.add_argument('--output_directory', required=True)
    args = parser.parse_args()

    # Remove rows with NaN or non-string sequences
    df = pd.read_csv(args.data_path)
    df = df.dropna(subset=["ID", "SEQUENCE"])
    df = df[df["SEQUENCE"].apply(lambda x: isinstance(x, str) and x.strip() != "")]
    uniprot_code_to_seq = {row["ID"]: row["SEQUENCE"] for _, row in df.iterrows()}

    all_models = {
        "esm2_t36_3B_UR50D",
        "esm2_t30_150M_UR50D",
        "esm2_t6_8M_UR50D",
    }

    excluded = set(args.excluded_models.split(",")) if args.excluded_models else set()
    models_to_run = all_models - excluded

    os.makedirs(args.output_directory, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name in models_to_run:
        print(f"Embedding with model: {model_name}")
        model_output_dir = os.path.join(args.output_directory, model_name + "/esm_embeddings")
        os.makedirs(model_output_dir, exist_ok=True)

        if model_name.startswith("esm2"):
            esm2_embedding(uniprot_code_to_seq, model_output_dir, "facebook/"+model_name, device)
        elif model_name.startswith("esm_msa1b") or model_name.startswith("esm1"):
            esm1_embedding(uniprot_code_to_seq, model_output_dir, "facebook/"+model_name, device)
        else:
            print(f"Model '{model_name}' not recognized for embedding. Skipping.")

if __name__ == "__main__":
    main()
