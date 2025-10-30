"""
This script embeds sequences to esmc embeddings (last layer)
"""

import json
import os
import argparse
import torch
import pandas as pd

# from transformers import AutoTokenizer, EsmModel, EsmForMaskedLM, EsmForProteinFolding
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
# import requests
# import csv
# from Bio.PDB import PDBParser



def esmc_embedding(uniprot_code_to_seq, output_dir, version, device):
    """Helper for ESMC model embeddings"""
    client = ESMC.from_pretrained("esmc_600m").to(device)
    for uniprot_code, sequence in uniprot_code_to_seq.items():
        protein = ESMProtein(sequence=sequence)
        protein_tensor = client.encode(protein)
        logits_output = client.logits(
            protein_tensor,
            LogitsConfig(sequence=True, return_embeddings=True)
        )
        torch.save(logits_output.embeddings.squeeze().cpu(),
                   os.path.join(output_dir, f"{uniprot_code}.pt"))
        print(f"made {uniprot_code}.pt")
        print(logits_output.embeddings.squeeze().shape)


# def esm1_embedding(uniprot_code_to_seq, output_dir, model_name, device):
#     """Helper for ESM-1 family embeddings"""
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = EsmForMaskedLM.from_pretrained(model_name, output_hidden_states=True).to(device)

#     for uniprot_code, sequence in uniprot_code_to_seq.items():
#         inputs = tokenizer(sequence, return_tensors="pt",
#                            truncation=True, max_length=1024).to(device)
#         with torch.no_grad():
#             outputs = model(**inputs)
#         torch.save(outputs.hidden_states[-1][0].cpu(),
#                    os.path.join(output_dir, f"{uniprot_code}.pt"))
#         print(f"made {uniprot_code}.pt")


# def esm2_embedding(uniprot_code_to_seq, output_dir, model_name, device):
#     """Helper for ESM-2 family embeddings"""
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = EsmModel.from_pretrained(model_name, output_hidden_states=True).to(device)

#     for uniprot_code, sequence in uniprot_code_to_seq.items():
#         inputs = tokenizer(sequence, return_tensors="pt",
#                            truncation=True, max_length=1024).to(device)
#         with torch.no_grad():
#             outputs = model(**inputs)
#         torch.save(outputs.last_hidden_state[0].cpu(),
#                    os.path.join(output_dir, f"{uniprot_code}.pt"))
#         print(f"made {uniprot_code}.pt")


def embed_esm(uniprot_code_to_seq, output_dir, version):
    """
    Unified embedding function with model family routing
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model family router
    #if version in ["esmc_300m", "esmc_650m", "esmc_3b"]:
    esmc_embedding(uniprot_code_to_seq, output_dir, version, device)

    # elif version in ["esm1v_t33_650M_UR90S_1", "esm1b_t33_650M_UR50S"]:
    #     esm1_embedding(uniprot_code_to_seq, output_dir, f"facebook/{version}", device)

    # elif version in ["esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D",
    #                  "esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D",
    #                  "esm2_t36_3B_UR50D"]:
    #     esm2_embedding(uniprot_code_to_seq, output_dir, f"facebook/{version}", device)

    #else:
    #    raise ValueError(f"Unsupported version: {version}\n"
    #                     "Available: esmc_300m|650m|3b, "
    #                     "esm1v_t33_650M_UR90S_1, esm1b_t33_650M_UR50S, "
    #                     "esm2_t[6|12|30|33|36]_*")




def embed_manager(sub_dict, range_values, output_dir):
    tool = sub_dict["tool"][0]
    tool_config = sub_dict["tool_config"][0]
    data_csv = sub_dict["data_csv"]

    print("sub_dict", sub_dict)

    df = pd.read_csv(data_csv)
    df = df.dropna(subset=['SEQUENCE'])
    selected_rows = df.iloc[:,:]  # +1 to include the end index

    uniprot_code_to_seq = {}
    # Correct way to iterate over DataFrame rows
    for _, row in selected_rows.iterrows():
        uniprot_code = row["ID"]
        sequence = row["SEQUENCE"]
        uniprot_code_to_seq[uniprot_code] = sequence
    print("uniprot_code_to_seq")
    print(uniprot_code_to_seq)
    if tool == "esm":
        embed_esm(uniprot_code_to_seq, output_dir, tool_config["version"])


def process_json(json_path, range_str=None, exp_key=None):
    """
    Reads a JSON file, extracts a sub-dictionary based on a given or first key,
    and processes a number range.

    Args:
        json_path (str): Path to the JSON file.
        exp_key (str, optional): Key to extract from JSON. If None, takes the first key.
        range_str (str, optional): Range string like "20-100".

    Returns:
        dict: Extracted sub-dictionary and parsed range values.
    """
    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Use the first key if no key is provided
    if exp_key is None:
        exp_key = next(iter(data))  # Get first key

    # Get the corresponding sub-dictionary
    if exp_key not in data:
        raise KeyError(f"exp_key '{exp_key}' not found in JSON.")

    sub_dict = data[exp_key]

    # Process range string
    if range_str:
        try:
            start, end = map(int, range_str.split("-"))
        except ValueError:
            raise ValueError(f"Invalid range format: {range_str}. Use 'start-end' format.")

        range_values = (start, end)
    else:
        range_values = None

    exp_dir_path = r"/sci/labs/asafle/yoel.marcu2003/Project_G/after_retreat/ESMC_600m/esmc_embeddings"
    os.makedirs(exp_dir_path, exist_ok=True)    


    embed_manager(sub_dict, range_values, exp_dir_path)


def main():
    parser = argparse.ArgumentParser(description='Embedding sequences.')
    parser.add_argument('--input_json', required=True, help='Path to the input filter JSON file.')
    parser.add_argument('--exp_key', required=False, help='')
    args = parser.parse_args()
    print("args:", args)
    process_json(json_path = args.input_json , exp_key = args.exp_key)


if __name__ == '__main__':
    main()
