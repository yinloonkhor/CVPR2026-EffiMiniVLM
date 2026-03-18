import argparse
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TRAIN_DEFAULTS
from data_pipeline import TestDataset, build_inference_collate_fn
from metric_utils import count_params, measure_flops
from model import MultimodalRegressor


def generate_predictions(
    model,
    config: dict,
    device,
    input_csv: str = "CVPR_workshop_efficiencyVLM/setB/input.csv",
    images_dir: str = "CVPR_workshop_efficiencyVLM/setB",
    output_csv: str = "submission.csv",
):
    """
    Run inference on the test set and write submission.csv with columns:
        item_id, score, params, flops
    """
    model.eval()

    total_params = count_params(model)
    params_millions = total_params / 1e6
    flops = measure_flops(
        model,
        batch_size=1,
        image_size=(224, 224),
        text_length=config.get("max_length", 256),
        device=str(device),
        with_images=True,
    )
    flops_giga = flops / 1e9
    print(f"  Model params : {params_millions:.1f}M")
    print(f"  Model FLOPs  : {flops_giga:.1f}G")

    test_dataset = TestDataset(
        csv_path=input_csv,
        images_dir=images_dir,
        num_images_per_sample=config.get("num_images_per_sample", 1),
    )

    collate_fn = build_inference_collate_fn(
        text_model_name=config.get(
            "text_model_name",
            "nreimers/MiniLM-L6-H384-uncased",
        ),
        max_length=config.get("max_length", 256),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get("batch_size", 16),
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        collate_fn=collate_fn,
    )

    all_item_ids = []
    all_scores = []

    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            has_images = batch["has_images"].to(device, non_blocking=True)

            preds = model(pixel_values, input_ids, attention_mask, has_images=has_images)
            preds = preds.float().squeeze(-1).clamp(1.0, 5.0)
            scores = preds.cpu().numpy()
            scores = [float(scores)] if scores.ndim == 0 else scores.tolist()

            all_item_ids.extend(batch["item_ids"])
            all_scores.extend(scores)

    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    n_total = len(all_scores)
    rt_per = elapsed / n_total if n_total > 0 else 0.0

    submission_df = pd.DataFrame({
        "item_id": all_item_ids,
        "score": [f"{s:.2f}" for s in all_scores],
        "params": [f"{params_millions:.1f}"] * n_total,
        "flops": [f"{flops_giga:.1f}"] * n_total,
    })
    submission_df.to_csv(output_csv, index=False)

    print("=" * 60)
    print(f"Submission saved to : {output_csv}")
    print(f"  Total predictions : {n_total}")
    print(f"  Score range       : [{min(all_scores):.2f}, {max(all_scores):.2f}]")
    print(f"  Params            : {params_millions:.1f}M")
    print(f"  FLOPs             : {flops_giga:.1f}G")
    print(f"  Runtime/image     : {rt_per:.4f}s  ({elapsed:.1f}s total)")
    print("=" * 60)

    return submission_df


def build_inference_model(config: dict, device):
    """Build the multimodal regressor from config for checkpoint loading."""
    model = MultimodalRegressor(
        image_model_name=config.get("image_model_name", "efficientnet_b0"),
        text_model_name=config.get(
            "text_model_name",
            "nreimers/MiniLM-L6-H384-uncased",
        ),
        hidden_dim=config.get("hidden_dim", 512),
        dropout=config.get("dropout", 0.2),
        freeze_image=config.get("freeze_image", False),
        freeze_text=config.get("freeze_text", False),
    ).to(device)
    return model


def load_model_from_path(model_path: str, config: dict, device):
    """Instantiate the model and load checkpoint weights from disk."""
    model = build_inference_model(config, device)
    print(f"Loading model from : {model_path}")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a trained checkpoint and generate a submission CSV.",
    )
    parser.add_argument(
        "--model-path",
        default=TRAIN_DEFAULTS.get("model_path"),
        help="Path to a trained .pt checkpoint.",
    )
    parser.add_argument(
        "--input-csv",
        default="CVPR_workshop_efficiencyVLM/setB/input.csv",
        help="Path to the workshop input CSV.",
    )
    parser.add_argument(
        "--images-dir",
        default="CVPR_workshop_efficiencyVLM/setB",
        help="Directory containing workshop images.",
    )
    parser.add_argument(
        "--output-csv",
        default="submission.csv",
        help="Where to write the generated submission CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = dict(TRAIN_DEFAULTS)

    if not args.model_path:
        raise ValueError(
            "No model path provided. Set TRAIN_DEFAULTS['model_path'] or pass --model-path."
        )

    config["model_path"] = args.model_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device      : {device}")

    model = load_model_from_path(args.model_path, config, device)
    generate_predictions(
        model=model,
        config=config,
        device=device,
        input_csv=args.input_csv,
        images_dir=args.images_dir,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
