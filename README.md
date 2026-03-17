# EffiMiniVLM

This repository is a solution for [LoViF @ CVPR 2026: Challenge on Efficient VLM for Multimodal Creative Quality Scoring](https://www.codabench.org/competitions/13463/)

Lightweight multimodal regression pipeline built around:
- `EfficientNet-B0` for images
- `MiniLM-L6-H384` for text
- a small MLP fusion head for scalar score prediction

The current codebase is split so data preparation, data loading, training, inference, and efficiency metrics are easier to inspect independently.

## Project Layout

- `config.py`: central defaults for dataset preparation and training.
- `prepare_dataset.py`: builds cached raw and cleaned CSV files from the Amazon Reviews 2023 metadata.
- `data_pipeline.py`: dataset classes and collate function used by training and inference.
- `model.py`: multimodal regressor definition.
- `train.py`: training loop and validation/test evaluation.
- `inference.py`: submission generation helper.
- `metric_utils.py`: parameter counting, FLOPs estimation, and efficiency metrics.

## Configuration

Default settings live in `config.py`:
- `PREPARE_DATASET_DEFAULTS` controls dataset cache paths, sampling fraction, random seed, and preprocessing worker count.
- `SELECTED_CATEGORIES` lists the Amazon categories processed by `prepare_dataset.py`.
- `TRAIN_DEFAULTS` controls training, validation, and inference defaults used by `train.py`.

If you want to change the default experiment behavior, update `config.py` first.

## Setup

Install and sync dependencies with:

```bash
git clone https://github.com/yinloonkhor/CVPR2026-EffiMiniVLM.git
cd CVPR2026-EffiMiniVLM
uv sync
```

Also obtain the command to install PyTorch with CUDA based on your device through this [website](https://pytorch.org/get-started/locally/). An example of the command is shown below. The current `pyproject.toml` doesn't include torch and torchvision because each machine requires different CUDA version.

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

## Dataset Preparation

Build the cached CSV files with:

```bash
uv run python prepare_dataset.py \
  --raw-data-path data/raw.csv \
  --cleaned-data-path data/cleaned.csv \
  --frac 0.2 \
  --random-state 42
```

Or download the processed dataset from [Google Drive](https://drive.google.com/file/d/1Av9anhjgmeX0rpBOurZ0rE81jgp3El8w/view?usp=drive_link).

`prepare_dataset.py` loads product metadata from the Hugging Face dataset [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).

Behavior:
- if `cleaned.csv` already exists, the script loads it directly for basic analysis
- otherwise it downloads/processes the raw metadata and then writes both cached CSVs

## Training

Run training with:

```bash
uv run python train.py
```

`train.py` reads defaults from `TRAIN_DEFAULTS` in `config.py`.

Current training flow:
- loads `data/cleaned.csv`
- creates train/val/test splits
- builds the multimodal model from `model.py`
- uses dataset/collate code from `data_pipeline.py`
- evaluates on the held-out test split after training
- optionally generates a submission if `generate_submission` is enabled in `TRAIN_DEFAULTS`

## Download and Load Trained Model
The best trained model (EfficientNet-B0 + MiniLMv2-L6-H384) can be downloaded from [here](https://drive.google.com/file/d/11wZ2-I5TEzKZ6e-TbWsR6nR8vYQnnwRS/view?usp=drive_link).

Use it with the inference workflow documented below.

## Inference

Submission generation is implemented in `inference.py` via `generate_predictions(...)`.

Download the workshop test set from [Kirin0010/CVPR_workshop_efficiencyVLM](https://huggingface.co/datasets/Kirin0010/CVPR_workshop_efficiencyVLM)

Clone it with Git and Git LFS enabled, for example:

```bash
git lfs install
git clone https://huggingface.co/datasets/Kirin0010/CVPR_workshop_efficiencyVLM
unzip CVPR_workshop_efficiencyVLM/setB/setB.zip -d CVPR_workshop_efficiencyVLM/setB
```

The inference path in `inference.py` expects the downloaded test files to be available locally under `CVPR_workshop_efficiencyVLM/`.

Once the checkpoint and test files are available locally, you can generate a submission with:

```bash
uv run python inference.py \
  --model-path path/to/best_model.pt \
  --input-csv CVPR_workshop_efficiencyVLM/setB/input.csv \
  --images-dir CVPR_workshop_efficiencyVLM/setB \
  --output-csv submission.csv
```

It currently:
- loads test samples from `CVPR_workshop_efficiencyVLM/setB/input.csv`
- reads local images from `CVPR_workshop_efficiencyVLM/setB`
- writes `submission.csv`
- reports parameter count and FLOPs alongside predictions

## Notes

- `model.py` currently supports only `efficientnet_b0` for the image backbone.
- `data_pipeline.py` fetches training images from URLs inside the dataset class, which is convenient for iteration but can become the main runtime bottleneck during training.
- `metric_utils.py` provides both analytical and runtime FLOPs paths; the analytical image FLOPs estimate is still based on EfficientNet-B0.

## Acknowledgement
- Datasets are obtained from [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).
- Dataset preprocessing techniques are referred from [AmazonReviews2023](https://github.com/hyp1231/AmazonReviews2023).
