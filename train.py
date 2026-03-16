import os
import random
import math
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import get_cosine_schedule_with_warmup

from data_pipeline import MultimodalDataset, build_collate_fn
from inference import generate_predictions
from model import MultimodalRegressor

from config import TRAIN_DEFAULTS

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------
# Data loading & splitting
# -----------------------------------------------------------------------
def load_and_split(file_path, random_state=42, test_size=0.2, stratify_threshold=50):
    """Load dataset from CSV and return train / val / test DataFrames."""
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} samples from {file_path}")

    # Merge rare classes into a single "RARE" stratum
    vc = df["average_rating"].value_counts()
    rare = vc[vc <= stratify_threshold].index
    df["strat_label"] = df["average_rating"].astype(str)
    df.loc[df["average_rating"].isin(rare), "strat_label"] = "RARE"

    # 80 / 10 / 10 split
    train_df, temp_df = train_test_split(
        df, test_size=test_size, random_state=random_state,
        stratify=df["strat_label"],
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=random_state,
        stratify=temp_df["strat_label"],
    )

    print(f"  Train : {len(train_df):>6,} samples")
    print(f"  Val   : {len(val_df):>6,} samples")
    print(f"  Test  : {len(test_df):>6,} samples")

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# -----------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------
def evaluate(model, loader, criterion, device):
    """Run one pass over *loader* and return (avg_loss, mae, rmse, plcc, std_pred)."""
    model.eval()

    total_loss = 0.0
    total_abs  = 0.0
    total_sq   = 0.0
    n          = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            pixel_values   = batch["pixel_values"].to(device, non_blocking=True)
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels         = batch["labels"].to(device, non_blocking=True).float()
            has_images     = batch["has_images"].to(device, non_blocking=True)

            preds = model(
                pixel_values, input_ids, attention_mask, has_images=has_images
            ).float().squeeze(-1)
            loss = criterion(preds, labels)

            bs = labels.size(0)
            total_loss += loss.item() * bs
            diff        = preds - labels
            total_abs  += diff.abs().sum().item()
            total_sq   += (diff * diff).sum().item()
            n          += bs

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / max(1, n)
    mae      = total_abs  / max(1, n)
    rmse     = math.sqrt(total_sq / max(1, n))

    preds_np  = torch.cat(all_preds).numpy()
    labels_np = torch.cat(all_labels).numpy()

    if preds_np.size < 2 or np.std(preds_np) == 0.0 or np.std(labels_np) == 0.0:
        plcc = 0.0
    else:
        plcc = float(np.corrcoef(preds_np, labels_np)[0, 1])

    std_pred  = float(np.std(preds_np))
    std_label = float(np.std(labels_np))

    return avg_loss, mae, rmse, plcc, std_pred, std_label


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    config = TRAIN_DEFAULTS.copy()

    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f" Multimodal Regression — {config['image_model_name']} + {config['text_model_name']}")
    print("=" * 60)
    print(f"  Device  : {device}")
    print(f"  Seed    : {config['seed']}")
    print(f"  Epochs  : {config['num_epochs']}")
    print(f"  Batch   : {config['batch_size']}  (accum steps: {config['gradient_accumulation_steps']})")
    print(f"  LR      : {config['learning_rate']}")
    print(f"  Freeze  : image={config['freeze_image']}  text={config['freeze_text']}")
    print("-" * 60)

    # ---- Dataset ----
    print("Loading dataset ...")
    train_df, val_df, test_df = load_and_split(config["train_file"], random_state=config["seed"])
    print("  rating_number stats (train):")
    global_mean_log_rating = float(np.log1p(train_df["rating_number"].fillna(0)).mean())
    print(f"  global mean log(1 + rating_number): {global_mean_log_rating:.4f}")

    train_dataset = MultimodalDataset(train_df, num_images_per_sample=config["num_images_per_sample"])
    val_dataset   = MultimodalDataset(val_df, num_images_per_sample=config["num_images_per_sample"])
    test_dataset  = MultimodalDataset(test_df, num_images_per_sample=config["num_images_per_sample"])

    # ---- Model ----
    model = MultimodalRegressor(
        image_model_name=config["image_model_name"],
        text_model_name=config["text_model_name"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
        freeze_image=config["freeze_image"],
        freeze_text=config["freeze_text"],
    ).to(device)

    print("-" * 60)
    print("Model summary:")
    print(f"  Image backbone : {config['image_model_name']}  (dim={model.image_dim})")
    print(f"  Text backbone  : {config['text_model_name']}  (dim={model.text_dim})")
    print(f"  Fused dim      : {model.image_dim + model.text_dim}")
    print(f"  Total params   : {sum(p.numel() for p in model.parameters()):>12,}")
    print(f"  Trainable      : {sum(p.numel() for p in model.parameters() if p.requires_grad):>12,}")
    print("-" * 60)

    # ---- Data loaders ----
    collate_fn = build_collate_fn(
        text_model_name=config["text_model_name"],
        max_length=config["max_length"],
        global_mean_log_rating=global_mean_log_rating,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"],
        shuffle=True,  num_workers=config["num_workers"],
        collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"],
        shuffle=False, num_workers=config["num_workers"],
        collate_fn=collate_fn, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"],
        shuffle=False, num_workers=config["num_workers"],
        collate_fn=collate_fn, pin_memory=True,
    )

    # ---- Loss / optimiser / scheduler ----
    criterion      = nn.SmoothL1Loss(reduction="none")   # per-sample, for weighted training
    eval_criterion = nn.SmoothL1Loss()                   # unweighted, for fair evaluation
    optimizer   = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
    )

    accum_steps             = config["gradient_accumulation_steps"]
    updates_per_epoch       = math.ceil(len(train_loader) / accum_steps)
    total_steps             = updates_per_epoch * config["num_epochs"]
    warmup_steps            = max(1, int(total_steps * config["warmup_ratio"]))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print("Scheduler:")
    print(f"  Total optimizer steps : {total_steps}")
    print(f"  Warmup steps          : {warmup_steps}")
    print("=" * 60)

    # ---- Paths ----
    best_dir = os.path.join(config["save_dir"], config["save_path"], "best")
    ckpt_dir = os.path.join(config["save_dir"], config["save_path"], "checkpoints")
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_plcc          = float("-inf")
    epochs_no_improve      = 0
    training_history       = []

    # ====================================================================
    # Training loop
    # ====================================================================
    if config["train"]:
        for epoch in range(config["num_epochs"]):
            model.train()
            optimizer.zero_grad(set_to_none=True)

            running_loss     = 0.0
            seen             = 0
            all_train_preds  = []
            all_train_labels = []

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}")
            for step, batch in enumerate(pbar):
                pixel_values   = batch["pixel_values"].to(device, non_blocking=True)
                input_ids      = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels         = batch["labels"].to(device, non_blocking=True)
                has_images     = batch["has_images"].to(device, non_blocking=True)

                preds    = model(
                    pixel_values, input_ids, attention_mask, has_images=has_images
                ).squeeze(-1)
                all_train_preds.append(preds.detach().cpu())
                all_train_labels.append(labels.detach().cpu())
                weights  = batch["weights"].to(device, non_blocking=True)
                raw_loss = (criterion(preds, labels) * weights).mean()
                loss     = raw_loss / accum_steps
                loss.backward()

                running_loss += raw_loss.item() * labels.size(0)
                seen         += labels.size(0)

                do_step = (
                    (step + 1) % accum_steps == 0
                    or (step + 1) == len(train_loader)
                )
                if do_step:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config["max_grad_norm"]
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                pbar.set_postfix(
                    train_loss=running_loss / max(1, seen),
                    lr=scheduler.get_last_lr()[0],
                )

            # ---- Validation ----
            val_loss, val_mae, val_rmse, val_plcc, val_std_pred, val_std_label = evaluate(
                model, val_loader, eval_criterion, device
            )
            train_loss_avg  = running_loss / max(1, seen)
            train_preds_np  = torch.cat(all_train_preds).numpy()
            train_labels_np = torch.cat(all_train_labels).numpy()
            train_std_pred  = float(np.std(train_preds_np))
            train_std_label = float(np.std(train_labels_np))
            if train_preds_np.size < 2 or np.std(train_preds_np) == 0.0 or np.std(train_labels_np) == 0.0:
                train_plcc = 0.0
            else:
                train_plcc = float(np.corrcoef(train_preds_np, train_labels_np)[0, 1])

            print(
                f"[Epoch {epoch + 1:02d}]"
                f"  Train: {train_loss_avg:.4f}"
                f"  PLCC: {train_plcc:.4f}"
                f"  std(y_pred): {train_std_pred:.4f}"
                f"  std(y_true): {train_std_label:.4f}"
                f"  Val: {val_loss:.4f}"
                f"  MAE: {val_mae:.4f}"
                f"  RMSE: {val_rmse:.4f}"
                f"  PLCC: {val_plcc:.4f}"
                f"  std(y_pred): {val_std_pred:.4f}"
                f"  std(y_true): {val_std_label:.4f}"
            )

            training_history.append({
                "epoch":           epoch + 1,
                "train_loss":      train_loss_avg,
                "train_plcc":      train_plcc,
                "train_std_pred":  train_std_pred,
                "train_std_label": train_std_label,
                "val_loss":        val_loss,
                "val_plcc":        val_plcc,
                "val_mae":         val_mae,
                "val_rmse":        val_rmse,
                "val_std_pred":    val_std_pred,
                "val_std_label":   val_std_label,
                "lr":              optimizer.param_groups[0]["lr"],
            })

            # ---- Checkpoint ----
            ckpt_path = os.path.join(ckpt_dir, f"epoch{epoch + 1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Checkpoint saved : {ckpt_path}")

            # ---- Best model ----
            if val_plcc > best_val_plcc:
                best_val_plcc = val_plcc
                epochs_no_improve = 0
                best_path = os.path.join(best_dir, "best_model.pt")
                torch.save(model.state_dict(), best_path)
                print(f"  Best model saved  (val_plcc={best_val_plcc:.4f})")
            else:
                epochs_no_improve += 1
                print(
                    f"  No improvement  "
                    f"({epochs_no_improve}/{config['patience']} patience epochs)"
                )

            if epochs_no_improve >= config["patience"]:
                print(
                    f"\nEarly stopping after epoch {epoch + 1} "
                    f"({config['patience']} epochs without improvement)."
                )
                break

        print("=" * 60)

    # ---- Save training history ----
    if training_history:
        hist_path = os.path.join(config["save_dir"], config["save_path"], "training_history.csv")
        pd.DataFrame(training_history).to_csv(hist_path, index=False)
        print(f"Training history saved to: {hist_path}")

    # ====================================================================
    # Final evaluation on test split
    # ====================================================================
    best_path = os.path.join(best_dir, "best_model.pt")
    print(f"\nLoading best model from {best_path} ...")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

    test_loss, test_mae, test_rmse, test_plcc, test_std_pred, test_std_label = evaluate(
        model, test_loader, eval_criterion, device
    )
    print("=" * 60)
    print("Test results:")
    print(
        f"  Loss : {test_loss:.4f}"
        f"  MAE  : {test_mae:.4f}"
        f"  RMSE : {test_rmse:.4f}"
        f"  PLCC : {test_plcc:.4f}"
        f"  std(y_pred): {test_std_pred:.4f}"
        f"  std(y_true): {test_std_label:.4f}"
    )
    print("=" * 60)

    # ====================================================================
    # Generate submission
    # ====================================================================
    if config.get("generate_submission", False):
        generate_predictions(
            model,
            config,
            device,
            input_csv="CVPR_workshop_efficiencyVLM/setB/input.csv",
            images_dir="CVPR_workshop_efficiencyVLM/setB",
            output_csv="submission.csv",
        )


if __name__ == "__main__":
    main()
