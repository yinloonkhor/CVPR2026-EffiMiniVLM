import os
from io import BytesIO

import pandas as pd
import requests
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer


# ImageNet normalisation constants used by torchvision EfficientNet models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_collate_fn(
    text_model_name: str = "nreimers/MiniLM-L6-H384-uncased",
    max_length: int = 256,
    image_size: int = 224,
    global_mean_log_rating: float = 1.0,
):
    """
    Returns a collate_fn that:
      - Tokenises text with the configured transformer tokenizer.
      - Preprocesses images with EfficientNet-compatible transforms
        (resize → tensor → ImageNet normalise).
      - Packs everything into a dict with keys:
          pixel_values, input_ids, attention_mask, labels, has_images
    """
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)

    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    black_image = Image.new("RGB", (image_size, image_size), (0, 0, 0))

    def collate_fn(batch):
        texts = [item["question"] for item in batch]
        text_enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        pixel_list = []
        for item in batch:
            img_list = item.get("image_urls", [])
            pil_img = img_list[0] if len(img_list) > 0 else black_image # Only 1 image per sample for now
            pixel_list.append(image_transform(pil_img))
        pixel_values = torch.stack(pixel_list, dim=0)

        if "average_rating" in batch[0]:
            labels = torch.tensor(
                [item["average_rating"] for item in batch],
                dtype=torch.float32,
            )
        else:
            labels = torch.zeros(len(batch), dtype=torch.float32)

        raw_counts = torch.tensor(
            [item.get("rating_number", 0.0) for item in batch],
            dtype=torch.float32,
        )
        raw_weights = torch.log1p(raw_counts)
        weights = (
            raw_weights / global_mean_log_rating
            if global_mean_log_rating > 0.0
            else torch.ones_like(raw_weights)
        )
        weights = weights.clamp(max=5.0)

        has_images = torch.tensor(
            [1.0 if item.get("has_images", False) else 0.0 for item in batch],
            dtype=torch.float32,
        ).unsqueeze(1)

        return {
            "pixel_values": pixel_values,
            "input_ids": text_enc["input_ids"],
            "attention_mask": text_enc["attention_mask"],
            "labels": labels,
            "has_images": has_images,
            "weights": weights,
        }

    return collate_fn


class MultimodalDataset(Dataset):
    """
    Training / validation / test dataset that loads images from URLs.
    Text fields are concatenated into labeled lines.
    """

    def __init__(self, df: pd.DataFrame, num_images_per_sample: int = 1):
        self.df = df
        self.num_images_per_sample = num_images_per_sample

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        parts = []
        if pd.notna(row.get("main_category")) and str(row["main_category"]).strip():
            parts.append(f"Main category: {row['main_category']}")
        if pd.notna(row.get("title")) and str(row["title"]).strip():
            parts.append(f"Title: {row['title']}")
        if pd.notna(row.get("features")) and str(row["features"]).strip():
            parts.append(f"Features: {row['features']}")
        if pd.notna(row.get("description")) and str(row["description"]).strip():
            parts.append(f"Description: {row['description']}")
        text = "\n".join(parts)

        hi_res = row["images_hi_res"].split(" | ") if pd.notna(row.get("images_hi_res")) else []
        large = row["images_large"].split(" | ") if pd.notna(row.get("images_large")) else []
        urls = (hi_res + large)[: self.num_images_per_sample]

        loaded = []
        for url in urls:
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content))

                # Handle palette/transparency images safely
                if img.mode in ("P", "LA") or (
                    img.mode == "RGBA" or "transparency" in img.info
                ):
                    img = img.convert("RGBA").convert("RGB")
                else:
                    img = img.convert("RGB")

                if img.width >= 10 and img.height >= 10:
                    loaded.append(img)
            except Exception:
                continue

        return {
            "question": text,
            "image_urls": loaded,
            "has_images": len(loaded) > 0,
            "average_rating": row["average_rating"],
            "rating_number": float(row["rating_number"]) if pd.notna(row.get("rating_number")) else 0.0,
        }


class TestDataset(Dataset):
    """
    Inference dataset that loads images from local disk (no labels).
    """

    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        num_images_per_sample: int = 1,
    ):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.num_images_per_sample = num_images_per_sample
        print(f"Loaded {len(self.df)} test samples from {csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item_id = row["parent_asin"]

        parts = []
        if pd.notna(row.get("main_category")) and str(row["main_category"]).strip():
            parts.append(f"Main category: {row['main_category']}")
        if pd.notna(row.get("title")) and str(row["title"]).strip():
            parts.append(f"Title: {row['title']}")
        if pd.notna(row.get("features")) and str(row["features"]).strip():
            parts.append(f"Features: {row['features']}")
        if pd.notna(row.get("description")) and str(row["description"]).strip():
            parts.append(f"Description: {row['description']}")
        text = "\n".join(parts)

        loaded = []
        if pd.notna(row.get("image_paths")) and str(row["image_paths"]).strip():
            paths = str(row["image_paths"]).split(";")
            paths = [p.strip() for p in paths if p.strip()][: self.num_images_per_sample]
            for rel_path in paths:
                full = os.path.join(self.images_dir, rel_path)
                try:
                    if os.path.exists(full):
                        img = Image.open(full).convert("RGB")
                        if img.width >= 10 and img.height >= 10:
                            loaded.append(img)
                except Exception:
                    continue

        return {
            "item_id": item_id,
            "question": text,
            "image_urls": loaded,
            "has_images": len(loaded) > 0,
        }
