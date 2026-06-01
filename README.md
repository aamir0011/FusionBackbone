# Thesis Reproduction Workspace (IRENE)

This repository is organized to reproduce and extend experiments around:

> **IRENE** — *A transformer-based representation-learning model with unified processing of multimodal input for clinical diagnostics* (Nature BME, 2023)

## 1) Current Status

- ✅ IRENE source code cloned into `IRENE/`
- ✅ Python 3.12 virtual environment created in `.venv`
- ✅ Multiple multimodal datasets downloaded under `datasets/`
- ✅ Added dataset-preparation scripts to convert multimodal VQA datasets into IRENE-compatible format
- ✅ Added a full **training script** (`IRENE/train_irene.py`)
- ✅ Added `experiments/` folder for experiment artifacts and logs

## 2) Repository Layout

```text
thesis/
├── .venv/                         # Python 3.12 environment
├── README.md
├── experiments/                   # global experiment notes/logs
├── datasets/
│   ├── covid-chestxray-dataset/
│   ├── vqa-rad/                   # saved HF dataset
│   ├── path-vqa/                  # saved HF dataset
│   ├── slake/                     # saved HF dataset metadata
│   └── medical-cxr-vqa/           # additional medical VQA metadata
└── IRENE/
    ├── models/
    ├── irene.py                   # original inference script
    ├── train_irene.py             # NEW: training script
    ├── tools_prepare_covid_sample.py
    ├── tools_prepare_vqa_multimodal.py
    ├── data/
    │   ├── covid_images/
    │   ├── covid_sample.pkl
    │   ├── vqa_rad_images/
    │   ├── vqa_rad_train.pkl
    │   ├── vqa_rad_val.pkl
    │   └── vqa_rad_test.pkl
    └── experiments/
        └── smoke_vqarad/
            └── best_model.pth
```

## 3) Environment Setup

From `thesis/`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch torchvision pandas scikit-image scikit-learn matplotlib pillow tqdm ml-collections datasets pyarrow
```

## 4) Datasets (Multimodal)

Downloaded datasets include:

1. **COVID Chest X-ray Dataset**
   - image + metadata/clinical fields
2. **VQA-RAD**
   - image + question + answer
3. **Path-VQA**
   - image + question + answer

> Note: The original IRENE paper uses proprietary clinical multimodal data (CXR + unstructured complaint/history + structured labs/demographics) and internal curation. Public datasets are used here for reproducible approximation.

## 5) Converting Datasets to IRENE Format

### VQA-RAD -> IRENE format

```bash
source .venv/bin/activate
python IRENE/tools_prepare_vqa_multimodal.py \
  --dataset_disk_path datasets/vqa-rad \
  --out_data_dir IRENE/data/vqa_rad_images \
  --prefix IRENE/data/vqa_rad
```

This generates:

- `IRENE/data/vqa_rad_train.pkl`
- `IRENE/data/vqa_rad_val.pkl`
- `IRENE/data/vqa_rad_test.pkl`
- corresponding PNG images in `IRENE/data/vqa_rad_images/`

### COVID sample conversion

```bash
source .venv/bin/activate
python IRENE/tools_prepare_covid_sample.py \
  --src datasets/covid-chestxray-dataset \
  --out-data-dir IRENE/data/covid_images \
  --out-pkl-prefix IRENE/data/covid_sample \
  --limit 600
```

## 6) Training IRENE

Run from `thesis/IRENE`:

```bash
source ../.venv/bin/activate
python train_irene.py \
  --train_pkl data/vqa_rad_train.pkl \
  --val_pkl data/vqa_rad_val.pkl \
  --img_dir data/vqa_rad_images \
  --epochs 10 \
  --batch_size 16 \
  --out_dir experiments/vqa_rad_run1
```

Output:

- Best checkpoint at `experiments/<run_name>/best_model.pth`
- Per-epoch train/validation loss + validation AUROC in console logs

## 7) Smoke Test Completed

A one-epoch smoke run was completed successfully:

- Command: `train_irene.py` on VQA-RAD prepared split
- Result: training + validation executed end-to-end
- Checkpoint: `IRENE/experiments/smoke_vqarad/best_model.pth`

## 8) Experiment Tracking Convention

Store each experiment under:

- `IRENE/experiments/<dataset>_<date>_<tag>/`

Recommended files in each run folder:

- `config.json` (all hyperparameters)
- `train.log`
- `metrics.csv`
- `best_model.pth`
- `notes.md` (what changed and why)

## 9) Important Reproducibility Notes

- Original paper’s exact dataset and annotations are not publicly available.
- Current public-dataset pipeline is a reproducible approximation for methodology validation.
- VQA-derived labels are heuristic mappings to IRENE’s 8 disease slots.
- For strict result reproduction, obtain official dataset splits/checkpoints from authors if available.

## 10) Collaboration Checklist

When a collaborator joins:

1. Clone repo
2. Create Python 3.12 venv
3. Install requirements
4. Run one smoke experiment
5. Start new experiments under `IRENE/experiments/` with clear naming

---

## 11) Latest GPU Experiment Runs (Branch: `exp/vqarad-pathvqa-run1`)

All runs were executed on **GPU (`cuda:0`, RTX 3060)** with mixed precision (`--amp`).

| Run | Dataset | Epochs | Batch | Best Val AUROC | Checkpoint |
|---|---|---:|---:|---:|---|
| vqarad_run1 | VQA-RAD | 3 | 16 | 0.8121 | `experiments/irene_runs/vqarad_run1/best_model.pth` |
| pathvqa_run1 | Path-VQA | 2 | 12 | 0.3757 | `experiments/irene_runs/pathvqa_run1/best_model.pth` |
| mimic_run1 | MIMIC-CXR (itsanmolgupta) | 2 | 12 | 0.6284 | `experiments/irene_runs/mimic_run1/best_model.pth` |
| rexgradient_run1 | ReXGradient-160K | 2 | 12 | 0.7785 | `experiments/irene_runs/rexgradient_run1/best_model.pth` |

Structured summary CSV:

- `experiments/results_summary.csv`

### Requested additional datasets

- `itsanmolgupta/mimic-cxr-dataset` ✅ downloaded and used in run `mimic_run1`
- `rajpurkarlab/ReXGradient-160K` ✅ downloaded (with HF token access) and used in run `rexgradient_run1`

If needed, we can next add:

- full `requirements.txt`
- evaluation script for test split
- experiment registry (CSV/JSONL)
- optional Weights & Biases logging

## 12) Sequential Re-run (All Datasets, 1-epoch quick benchmark)

Runs executed one-by-one on GPU to avoid overload.

| Dataset | Best Val AUROC | Last Val Loss | Run Dir |
|---|---:|---:|---|
| vqarad | 0.47555953489684755 | 0.041864100669045 | `experiments/rerun_mm-modality-wise-gating_vqarad` |
| pathvqa | 0.4948422102667413 | 0.10861784324515611 | `experiments/rerun_mm-modality-wise-gating_pathvqa` |
| mimic | 0.45897209630575725 | 0.302647490054369 | `experiments/rerun_mm-modality-wise-gating_mimic` |
| rexgradient | 0.5950708886756622 | 0.4481514561921358 | `experiments/rerun_mm-modality-wise-gating_rexgradient` |

Raw CSV: `experiments/rerun_mm-modality-wise-gating_all_datasets.csv`

## 13) Full-scale sequential experiments (all datasets, one-by-one GPU queue)

- Approach used on this branch: **Modality-wise gating**
- Training plan: **10 epochs**, full train/val splits (no sample caps), sequential GPU runs

| Dataset | Best Val AUROC | Last Val Loss | Run Dir |
|---|---:|---:|---|
| vqarad | 0.9855783487710852 | 0.019038281581528923 | `experiments/fullscale_mm-modality-wise-gating_vqarad` |
| pathvqa | 0.7794761582726257 | 0.0671849969914183 | `experiments/fullscale_mm-modality-wise-gating_pathvqa` |
| mimic | 0.6734750146032351 | 0.20293055269532576 | `experiments/fullscale_mm-modality-wise-gating_mimic` |
| rexgradient | 0.8956396282336592 | 0.13902732627856695 | `experiments/fullscale_mm-modality-wise-gating_rexgradient` |

Raw CSV: `experiments/fullscale_mm-modality-wise-gating_all_datasets.csv`
