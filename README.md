# DeltaRec

**History-efficient sequential recommendation across four backends.**

DeltaRec trains a history selector from teacher-derived CWI targets, then fine-tunes a recommender on selected histories. This project contains the **12 DeltaRec experiments in Table 1**: LinRec, HSTU, FuXi-Linear and BlossomRec on ML-20M, Amazon Books and KuaiRand-1K. Each configuration retains its original selection, grouping, loss and evaluation rules, including the 25% history retention setting and its backend-specific budget floors. Table 2 is outside the current release scope.

[Environment Setups](#environment-setups) · [Start Training](#start-training) · [Validation](#validation) · [Results](#results)

```text
deltarec/
├── deltarec/          # Internal layers, models, adaptors, data, metrics and utilities
├── configs/           # One configuration per backend × dataset
├── scripts/           # Table 1 runner and smoke check
├── train_*.py         # Four backend-specific training and validation entries
├── prepare_data.py    # Raw-data preprocessing and split construction
├── experiments.json   # Configuration, checkpoint, protocol and result bindings
└── results/           # Measured metrics and verification summaries
```

## Environment Setups

### Install dependencies

The verified environment is **Linux with an NVIDIA GPU**. The versions below describe the environment used for the supplied measurements.

| Component | Tested version / hardware |
|---|---|
| Python | 3.10.19 |
| PyTorch | 2.9.0+cu126 |
| CUDA runtime | 12.6 |
| Triton / FLA core | 3.5.0 / 0.5.1 |
| TorchRec / FBGEMM GPU | 1.4.0+cu126 / 1.4.0+cu126 |
| GPU | NVIDIA A100-SXM4-80GB |
| NVIDIA driver | 580.65.06 |

Run all commands from the project root:

```bash
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install pip==25.2
python -m pip install -r requirements.txt
python -m pip check
```

### Prepare data and checkpoints

The source ZIP contains code, configurations and result summaries. Obtain the companion archives separately and place them beside the project directory. Their filenames and SHA-256 checksums are recorded in [experiments.json](experiments.json).

| Archive | Contents |
|---|---|
| `deltarec_weights.zip` | Inference checkpoints and required teacher/initializer weights |
| `deltarec_frozen_assets.zip` | Frozen validation candidates and grouping assets |

```bash
python -m zipfile -e ../deltarec_weights.zip ..
python -m zipfile -e ../deltarec_frozen_assets.zip ..
```

Both archives contain a `deltarec/` prefix and populate this project's `checkpoints/` and `data/` directories. **No public anonymous download endpoint is currently available.** Access requirements and unresolved redistribution permissions are documented in [ASSETS.md](ASSETS.md).

Obtain the original raw datasets using the sources and exact versions listed in [ASSETS.md](ASSETS.md), then prepare the datasets needed for your experiment. All three are required for the full Table 1 run:

```bash
python prepare_data.py --dataset ml-20m \
  --raw data/raw/ml-20m/ratings.csv

python prepare_data.py --dataset amazon-books \
  --raw data/raw/amazon-books/ratings_Books.csv

python prepare_data.py --dataset kuairand-1k \
  --raw data/raw/KuaiRand-1K/data/log_standard_4_08_to_4_21_1k.csv \
        data/raw/KuaiRand-1K/data/log_standard_4_22_to_5_08_1k.csv \
  --user-features data/raw/KuaiRand-1K/data/user_features_1k.csv
```

ML-20M and Amazon Books require the supplied frozen top-100 candidate sets. These preserve the original retrieval results and ID mapping; replacing them with newly sampled negatives changes the evaluation protocol.

## Start Training

The training entries retain the existing staged workflow:

```text
Full-history warm-up → CWI targets → Selector distillation → Sparse fine-tuning
```

Start the LinRec × ML-20M experiment with its supplied configuration:

```bash
python train_linrec.py \
  --config configs/linrec_ml20m.json \
  --data-root data \
  --output outputs/linrec_ml20m \
  --device cuda:0
```

The other backend entries are `train_hstu.py`, `train_fuxilinear.py` and `train_blossomrec.py`; their dataset configurations are in `configs/`.

**Resume and reuse.** Rerun the same command with the same configuration and output directory to restore saved stages. Existing valid teacher, CWI and selector products are reused. `--start-stage` and `--stop-stage` accept `full-gdr`, `cwi`, `selector` and `sparse`; entering a later stage requires its preceding stage products. The inherited `full-gdr` CLI name also covers LinRec's native full-history warm-up. Some HSTU configurations start from supplied pretrained initializers; their short teacher continuations are not from-scratch training. See [ASSETS.md](ASSETS.md) for these distinctions.

The Table 1 runner can also launch an individual experiment:

```bash
bash scripts/table1.sh train linrec ml20m
```

Omit the backend and dataset arguments to launch all 12 training runs. This is a full training workload; the supplied checkpoints can be validated directly without repeating it.

## Validation

### Check the training pipeline

```bash
bash scripts/smoke_test.sh
```

The smoke check runs warm-up, CWI, selector training, sparse fine-tuning, checkpoint saving, restart and evaluation for all four backends. It uses small synthetic inputs to check the computation paths. Pass a backend name, for example `bash scripts/smoke_test.sh hstu`, to check one entry. These engineering checks do not establish paper-level quality from scratch.

### Evaluate a checkpoint

Run the complete ML-20M validation split using the released LinRec checkpoint:

```bash
python train_linrec.py \
  --config configs/linrec_ml20m.json \
  --data-root data \
  --evaluate \
  --checkpoint checkpoints/linrec_ml20m.pt \
  --output outputs/validation/linrec_ml20m \
  --device cuda:0
```

### Reproduce Table 1

```bash
# Evaluate all 12 backend × dataset combinations.
bash scripts/table1.sh eval

# Evaluate one combination in a separate output directory.
OUTPUT_ROOT=outputs/recheck bash scripts/table1.sh eval hstu kuairand
```

The runner accepts backend names `linrec`, `hstu`, `fuxilinear`, `blossomrec` and dataset names `ml20m`, `amazon`, `kuairand`; either position can be `all`. Override `PYTHON`, `DEVICE`, `DATA_ROOT`, `CHECKPOINT_ROOT` or `OUTPUT_ROOT` through environment variables when needed. Use a fresh output directory to keep separate evaluation runs.

| Dataset | Complete validation workload | Reported metrics |
|---|---:|---|
| ML-20M | 138,444 requests | HR@10, NDCG@10 |
| Amazon Books | 576,890 requests | HR@10, NDCG@10 |
| KuaiRand-1K | 36,007 slates / 1,152,224 exposures | Macro-GAUC |

Rating metrics include every validation request, including retrieval misses, with no forced positive injection. KuaiRand computes AUC from pooled raw logits for each eligible user, weights user AUCs by their valid exposure counts within each task, then takes an unweighted mean across eight tasks. Users with only one label class are excluded for that task. Exact metric and candidate definitions are recorded in [experiments.json](experiments.json).

## Results

The following results were **measured from the selected checkpoints on the complete validation splits**. Values are displayed to six decimal places; [table1.csv](results/table1.csv) retains the measured precision, paper targets, differences and reproduction status.

| Backend | ML-20M HR@10 | ML-20M NDCG@10 | Amazon Books HR@10 | Amazon Books NDCG@10 | KuaiRand-1K Macro-GAUC |
|---|---:|---:|---:|---:|---:|
| LinRec | 0.244099 | 0.134695 | 0.055948 | 0.034251 | 0.525358 |
| HSTU | 0.251575 | 0.138130 | 0.044672 | 0.024191 | 0.540525 |
| FuXi-Linear | 0.239375 | 0.128589 | 0.043110 | 0.023795 | 0.535928 |
| BlossomRec | 0.237959 | 0.129902 | 0.048501 | 0.029242 | 0.536548 |

Using the fixed absolute tolerance of `5e-5`, **3 combinations align, 5 measure better and 4 remain below target**. A combination is classified as better when at least one metric exceeds its target by more than the tolerance and none falls below its target by more than the tolerance. The remaining gaps are LinRec × KuaiRand-1K, HSTU × Amazon Books, HSTU × KuaiRand-1K and FuXi-Linear × KuaiRand-1K.

| Record | What it contains |
|---|---|
| [Experiment bindings](experiments.json) | Configurations, checkpoint checksums, protocols, targets and measured differences |
| [Preprocessing checks](results/preprocessing.json) | Three raw-data pipelines; 14 numeric outputs matching the reference files |
| [Checkpoint checks](results/checkpoint_parity.json) | Tensor preservation and fixed-input prediction comparisons, with their scope |
| [Final package verification](results/final_validation.json) | Archive extraction, asset checksums, clean-environment checks and a complete LinRec × ML-20M re-evaluation |

The extracted package passed dependency/import checks and the four-backend smoke pipeline. Its complete LinRec × ML-20M re-evaluation matched the primary measurements exactly. **Full-scale retraining of all experiments was not performed.**

**Known limitations and distribution status**

- HSTU × Amazon Books: the source checkpoint was selected using a 1,024-request validation shard. The result above uses all 576,890 validation requests and remains below the paper target.
- HSTU × KuaiRand-1K: a numerical difference between the original and clean environments remains unresolved; cross-environment prediction equivalence is not claimed.
- Redistribution permissions for some LinRec/BlossomRec components and derived dataset assets remain unresolved. This working bundle is **not yet cleared for reviewer upload**. Required attribution is retained in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); asset access and restrictions are described in [ASSETS.md](ASSETS.md).
