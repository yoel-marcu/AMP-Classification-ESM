

# Sequence-Based Classification of Antimicrobial Small Proteins Using Deep Learning on ESM embedding

Utilities for:
- **Training** binary classifiers on **precomputed protein embeddings** (ESM / ESMC)
- **Running predictions** over **FASTA** sequences using trained models
- **Visualizing** results with scatter plots comparing embedding families

> ⚠️ This README covers only the *experiment*, *prediction*, and *plotting* scripts.  
> It **does not** include the Maximal Independent Set (BLAST) pipeline.

---

##  Requirements

- **Python ≥ 3.9**
- **PyTorch**
- **pandas**, **numpy**, **scikit-learn**, **matplotlib**, **biopython**
- (Optional) **CUDA** GPU acceleration

Install dependencies:
```bash
pip install torch pandas numpy scikit-learn matplotlib biopython
```

---

##  Repository Layout

```
.
├── CSVSequenceDataset.py          # Dataset class for (ID, SEQUENCE, LABEL) CSVs
├── TrainingExperimentConfig.py    # Dataclass holding experiment configs
├── Networks.py                    # Fixed/Dynamic MLP architectures
├── new_experiments.py             # Trains models & performs CV + test evaluation
├── use_models.py                  # Runs trained models on FASTA sequences
├── plot_compare_esms.py           # Generates scatter plots from results CSV
```


---

##  Data Formats

### Train and Test CSV
In order to use the experiment script you must have your train and test data separately in tables in the following format:

| Column     | Description                                                  |
| ---------- | ------------------------------------------------------------ |
| `ID`       | Protein identifier - must match embedding filename `{ID}.pt` |
| `SEQUENCE` | Protein sequence (string)                                    |
| `LABEL`    | Binary label (0 or 1)                                        |

> ⚠️ `CSVSequenceDataset.extract_id()` trims at the first underscore.
> Modify it if your embedding filenames retain the full ID.
> We don't supply the embeddings for each sequence here, but the scripts used to generate them are given in More Code.
> It is neccesary to have them properly in order to use the experiment script.

---

##  1) Training Experiments

`new_experiments.py` orchestrates multiple training runs across combinations of:

* **Embedding sources**
* **Pooling functions** (`mean`, `median`, `min`, `max`, or dual combinations)
* **Network types** (`FixedMLP`, `DynamicMLP`, etc.)
* **Weighted/unweighted losses**

Each experiment performs:

1. **5-fold cross-validation**
2. **Full-train evaluation on test set**
3. **Metric & model export**

### Configure Embedding Sources

Inside `new_experiments.py`, edit the dictionary so that your embedding directories match those in the dictionary:

```python
embedding_sources = {
    "esm2_t6":   ("/path/to/esm2_t6_8M_UR50D/esm_embeddings", 320),
    "esm2_t30":  ("/path/to/esm2_t30_150M_UR50D/esm_embeddings", 640),
    "esm2_t36":  ("/path/to/esm2_t36_3B_UR50D/esm_embeddings", 2560),
    "esmc_300m": ("/path/to/ESMC_300m/esmc_embeddings", 960),
    "esmc_600m": ("/path/to/ESMC_600m/esmc_embeddings", 1152),
}
```

Make sure each `input_dim` matches your embedding size.

### Run

```bash
python new_experiments.py \
  --csv_path_train /data/train.csv \
  --csv_path_test  /data/test.csv \
  --root           /results/experiments
```

### Output Structure

```
/results/experiments/
├── {experiment_name}/
│   ├── {experiment_name}_final_model.pth
│   └── {experiment_name}_experiment_summary.pkl
└── new_experiments_results_summary.csv
```

Each experiment folder contains:

* Model checkpoint (`.pth`)
* Summary of metrics
* Training configuration details

`new_experiments_results_summary.csv` aggregates all runs with:

* Final metrics (`AUC`, `Precision`, `Recall`, `F1`, etc.)
* Cross-validation mean/std metrics
* Paths to model checkpoints

---

##  2) Predictions over FASTA

`use_models.py` applies trained models to **FASTA sequences** and produces a **wide-format CSV**:

* Each **row** = sequence
* Each **column** = model
* Each **value** = predicted probability (`sigmoid(logit)`)

### Inputs

| Argument                    | Description                                              |
| --------------------------- | -------------------------------------------------------- |
| `--sequences_fasta`         | FASTA file of sequences to predict                       |
| `--models_csv`              | Path to training summary CSV                             |
| `--root`                    | Output directory                                         |
| `--batch_size`              | Number of sequences per batch (default: 64)              |
| `--output_wide`             | Output CSV name (default: `predictions_wide.csv`)        |
| `--old_root` / `--new_root` | Maps checkpoint paths from training environment to local |

By default, only models where:

* `embedding_name` contains `"esmc_300m"`
* `Pooling Functions` include `"mean_pool"` or `"median_pool"`
  are loaded. You can easily edit the code to match the models you wish to use.

### Run

```bash
python use_models.py \
  --sequences_fasta /in/sequences.fasta \
  --models_csv /results/experiments/new_experiments_results_summary.csv \
  --root /results/predictions \
  --batch_size 64 \
  --output_wide predictions_wide.csv \
  --old_root /remote/models \
  --new_root /local/models
```

### Output

```
/results/predictions/predictions_wide.csv
```

**Format:**

| Column   | Description           |         |                              |
| -------- | --------------------- | ------- | ---------------------------- |
| `seq_id` | Sequence identifier   |         |                              |
| `Fixed   | mean_pool             | modelX` | Probability output (sigmoid) |
| `Dynamic | mean_pool+median_pool | modelY` | ...                          |


---

##  3) Scatter Plot Visualization

`plot_compare_esms.py` visualizes performance across embedding families.

### Plots Generated

1. **`pr_by_embedding.png`** - Precision vs Recall per embedding
2. **`mean_precision_vs_std_by_embedding.png`** - Mean CV Precision vs Std CV Precision (stability)

### Inputs

* `--csv`: path to the experiment summary (`new_experiments_results_summary.csv`)
* `--outdir`: directory for output plots

### Run

```bash
python plot_compare_esms.py \
  --csv /results/experiments/new_experiments_results_summary.csv \
  --outdir /results/plots
```

### Output

```
/results/plots/
├── pr_by_embedding.png
└── mean_precision_vs_std_by_embedding.png
```

##  Acknowledgements

The protein embeddings used in this toolkit are derived from:

### Core Methods
```bibtex
What citations should i attach here?```

