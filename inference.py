import time

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_pipeline import TestDataset, build_collate_fn
from metric_utils import count_params, measure_flops


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

    collate_fn = build_collate_fn(
        text_model_name=config.get(
            "text_model_name",
            "nreimers/MiniLM-L6-H384-uncased",
        ),
        max_length=config.get("max_length", 256),
    )

    def test_collate_fn(batch):
        for item in batch:
            item.setdefault("average_rating", 0.0)
        result = collate_fn(batch)
        result["item_ids"] = [item["item_id"] for item in batch]
        return result

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get("batch_size", 16),
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        collate_fn=test_collate_fn,
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
