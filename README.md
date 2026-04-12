# EffiMiniVLM

This repository is a solution for [LoViF @ CVPR 2026: Challenge on Efficient VLM for Multimodal Creative Quality Scoring](https://www.codabench.org/competitions/13463/) co-hosted by **Snap Inc.** & **NTU** & **SYSU**. Our approach ranked 3rd (tied with the official Snap Inc. team with 10x lesser training data and 4x to 8x smaller model footprint) globally in the challenge! 🏅🎉 Our paper is accepted by CVPRW 2026! 🥳

Lightweight multimodal regression pipeline built around:
- `EfficientNet-B0` for images
- `MiniLM-L6-H384` for text
- a small MLP fusion head for scalar score prediction

The current codebase is split so data preparation, data loading, training, inference, and efficiency metrics are easier to inspect independently.

## Updates [20260413]

Our post-challenge analysis demonstrates promising results in scaling model capabilities, with our approach potentially outperforming even the 2nd and 3rd place teams while maintaining the smallest model footprint. The findings from this analysis and study will be published soon. Stay tuned!

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

These setup and usage commands assume Ubuntu, but they were tested in a Windows environment.

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

Then, activate the environment after downloading all the libraries.
```bash
source ./.venv/bin/activate
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

Or download the processed dataset from [Google Drive](https://drive.google.com/file/d/1Av9anhjgmeX0rpBOurZ0rE81jgp3El8w/view?usp=drive_link), and store in `data` directory.

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

## Runtime Metrics

You can benchmark runtime metrics after preparing the checkpoint and inference dataset with:

```bash
uv run python runtime_metrics.py \
  --model-path models/efficientnet_minilm/best/best_model.pt \
  --input-csv CVPR_workshop_efficiencyVLM/setB/input.csv \
  --images-dir CVPR_workshop_efficiencyVLM/setB \
  --device cuda \
  --timing-scope end_to_end \
  --warmup-batches 5 \
  --output-json runtime_metrics.json
```

`runtime_metrics.py` reports:
- inference latency in `ms/sample`
- throughput in `samples/s`
- optional input-token-normalized metrics in `ms/token` and `tokens/s`
- peak GPU memory usage on CUDA as `peak_vram_allocated_mb` and `peak_vram_reserved_mb`

Notes:
- For this project, `ms/sample` and `samples/s` are the primary runtime metrics because the model predicts one score per sample rather than generating output tokens.
- `--warmup-batches 5` is recommended for steady-state benchmarking so one-time startup overhead is excluded from the reported latency and throughput.
- If you run on CPU, GPU memory metrics are omitted automatically.

## Notes

- `model.py` currently supports only `efficientnet_b0` for the image backbone.
- `data_pipeline.py` fetches training images from URLs inside the dataset class, which is convenient for iteration but can become the main runtime bottleneck during training.
- `metric_utils.py` provides both analytical and runtime FLOPs paths; the analytical image FLOPs estimate is still based on EfficientNet-B0.

## Acknowledgement
- Datasets are obtained from [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).
- Dataset preprocessing techniques are referred from [AmazonReviews2023](https://github.com/hyp1231/AmazonReviews2023).

-------

## Cite this repository

If this repo helps your research, please kindly star this repo and cite our paper 😄 The preprint can be found [here](https://arxiv.org/pdf/2604.03172)!

```bash
@InProceedings{khor2026cvprw,
    author    = {Khor, Yin-Loon and Wong, Yi-Jie and Hum, Yan Chai}, 
    title     = {EffiMiniVLM: A Compact Dual-Encoder Regression Framework},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2026},
    pages     = {}
}

@inproceedings{zhang2026lovif,
    title   ={The 1st LoViF Challenge on Efficient VLM for Multimodal Creative Quality Scoring: Methods and Results},
    author  = {Zhang, Jusheng and Lyu, Qinhan and Li, Xin and Yang, Jing and Zhshchao and Ma, Sizhuo and Cao, Min and Wang, Jian and Leach, William and He, Ru and Jia, Yizhen and Cao, Sheng and Sui, Peimeng and Zhang, Hong and Khor, Yin-Loon and Wong, Yi-Jie and Hao, Yu},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2026},
    pages     = {}
  }
```
