import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import TRAIN_DEFAULTS
from data_pipeline import TestDataset, build_collate_fn
from inference import load_model_from_path


def build_test_loader(
    config: dict,
    input_csv: str,
    images_dir: str,
    batch_size: int,
    num_workers: int,
):
    dataset = TestDataset(
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

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": test_collate_fn,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    return DataLoader(dataset, **loader_kwargs)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        "pixel_values": batch["pixel_values"].to(device, non_blocking=True),
        "input_ids": batch["input_ids"].to(device, non_blocking=True),
        "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
        "has_images": batch["has_images"].to(device, non_blocking=True),
    }


def synchronize_if_needed(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory_if_needed(device: torch.device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def collect_memory_stats(device: torch.device) -> dict:
    if device.type != "cuda":
        return {
            "peak_vram_allocated_mb": None,
            "peak_vram_reserved_mb": None,
        }

    return {
        "peak_vram_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        "peak_vram_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024 ** 2),
    }


def run_forward_pass(model, batch_on_device: dict):
    preds = model(
        batch_on_device["pixel_values"],
        batch_on_device["input_ids"],
        batch_on_device["attention_mask"],
        has_images=batch_on_device["has_images"],
    )
    return preds.float().squeeze(-1).clamp(1.0, 5.0)


def warmup_model(model, loader, device: torch.device, warmup_batches: int):
    if warmup_batches <= 0:
        return

    iterator = iter(loader)
    with torch.inference_mode():
        for _ in range(warmup_batches):
            try:
                batch = next(iterator)
            except StopIteration:
                break
            batch_on_device = move_batch_to_device(batch, device)
            _ = run_forward_pass(model, batch_on_device)
    synchronize_if_needed(device)


def benchmark_runtime(
    model,
    loader,
    device: torch.device,
    timing_scope: str,
    warmup_batches: int,
    max_batches: int | None,
):
    warmup_model(model, loader, device, warmup_batches)

    iterator = iter(loader)
    for _ in range(warmup_batches):
        try:
            next(iterator)
        except StopIteration:
            break

    reset_peak_memory_if_needed(device)
    synchronize_if_needed(device)

    total_samples = 0
    total_tokens = 0
    measured_batches = 0
    measured_seconds = 0.0

    with torch.inference_mode():
        if timing_scope == "end_to_end":
            start = time.perf_counter()

        while True:
            if max_batches is not None and measured_batches >= max_batches:
                break

            try:
                batch = next(iterator)
            except StopIteration:
                break

            batch_size = int(batch["input_ids"].shape[0])
            token_count = int(batch["attention_mask"].sum().item())

            if timing_scope == "model":
                batch_on_device = move_batch_to_device(batch, device)
                synchronize_if_needed(device)
                batch_start = time.perf_counter()
                _ = run_forward_pass(model, batch_on_device)
                synchronize_if_needed(device)
                measured_seconds += time.perf_counter() - batch_start
            else:
                batch_on_device = move_batch_to_device(batch, device)
                _ = run_forward_pass(model, batch_on_device)

            total_samples += batch_size
            total_tokens += token_count
            measured_batches += 1

        if timing_scope == "end_to_end":
            synchronize_if_needed(device)
            measured_seconds = time.perf_counter() - start

    if measured_batches == 0 or total_samples == 0:
        raise ValueError(
            "No batches were measured. Reduce --warmup-batches or check the input dataset."
        )

    memory_stats = collect_memory_stats(device)

    results = {
        "device": str(device),
        "timing_scope": timing_scope,
        "warmup_batches": warmup_batches,
        "measured_batches": measured_batches,
        "num_samples": total_samples,
        "num_input_tokens": total_tokens,
        "total_runtime_seconds": measured_seconds,
        "latency_ms_per_sample": (measured_seconds / total_samples) * 1000.0,
        "throughput_samples_per_sec": total_samples / measured_seconds,
        "latency_ms_per_token": (
            (measured_seconds / total_tokens) * 1000.0 if total_tokens > 0 else None
        ),
        "throughput_tokens_per_sec": (
            total_tokens / measured_seconds if total_tokens > 0 else None
        ),
        "avg_input_tokens_per_sample": total_tokens / total_samples,
    }
    results.update(memory_stats)
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark inference latency, throughput, and CUDA memory usage.",
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
        "--batch-size",
        type=int,
        default=TRAIN_DEFAULTS.get("batch_size", 32),
        help="Batch size used during benchmarking.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=TRAIN_DEFAULTS.get("num_workers", 8),
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=5,
        help="Number of warmup batches to run before timing.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional cap on the number of measured batches.",
    )
    parser.add_argument(
        "--timing-scope",
        choices=["end_to_end", "model"],
        default="end_to_end",
        help=(
            "Use 'end_to_end' to include loading, preprocessing, host-to-device copies, "
            "and model forward. Use 'model' to isolate forward-pass timing after tensors "
            "are on the target device."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Benchmark device, for example 'cuda', 'cuda:0', or 'cpu'.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write the measured metrics as JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.model_path:
        raise ValueError(
            "No model path provided. Set TRAIN_DEFAULTS['model_path'] or pass --model-path."
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available in this environment.")

    config = dict(TRAIN_DEFAULTS)
    config["model_path"] = args.model_path

    print(f"Using device      : {device}")
    print(f"Timing scope      : {args.timing_scope}")
    print(f"Batch size        : {args.batch_size}")
    print(f"Warmup batches    : {args.warmup_batches}")

    model = load_model_from_path(args.model_path, config, device)
    loader = build_test_loader(
        config=config,
        input_csv=args.input_csv,
        images_dir=args.images_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    results = benchmark_runtime(
        model=model,
        loader=loader,
        device=device,
        timing_scope=args.timing_scope,
        warmup_batches=args.warmup_batches,
        max_batches=args.max_batches,
    )

    print("=" * 60)
    print(f"Measured batches           : {results['measured_batches']}")
    print(f"Measured samples           : {results['num_samples']}")
    print(f"Measured input tokens      : {results['num_input_tokens']}")
    print(f"Total runtime              : {results['total_runtime_seconds']:.4f}s")
    print(f"Latency                    : {results['latency_ms_per_sample']:.4f} ms/sample")
    if results["latency_ms_per_token"] is not None:
        print(f"Latency                    : {results['latency_ms_per_token']:.6f} ms/token")
    print(
        f"Throughput                 : {results['throughput_samples_per_sec']:.4f} samples/s"
    )
    if results["throughput_tokens_per_sec"] is not None:
        print(
            f"Throughput                 : {results['throughput_tokens_per_sec']:.4f} tokens/s"
        )
    if results["peak_vram_allocated_mb"] is not None:
        print(
            "Peak VRAM allocated        : "
            f"{results['peak_vram_allocated_mb']:.2f} MB"
        )
        print(
            "Peak VRAM reserved         : "
            f"{results['peak_vram_reserved_mb']:.2f} MB"
        )
    else:
        print("Peak VRAM                  : omitted (CUDA not in use)")
    print("=" * 60)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"Saved metrics JSON to : {output_path}")


if __name__ == "__main__":
    main()
