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


class MultimodalCollator:
    """Picklable collator for multimodal batches."""

    def __init__(
        self,
        text_model_name: str,
        max_length: int,
        image_size: int,
        global_mean_log_rating: float,
    ):
        self.text_model_name = text_model_name
        self.max_length = max_length
        self.image_size = image_size
        self.global_mean_log_rating = global_mean_log_rating
        self._tokenizer = None
        self._image_transform = None
        self._black_image = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tokenizer"] = None
        state["_image_transform"] = None
        state["_black_image"] = None
        return state

    def _ensure_assets(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
        if self._image_transform is None:
            self._image_transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        if self._black_image is None:
            self._black_image = Image.new(
                "RGB",
                (self.image_size, self.image_size),
                (0, 0, 0),
            )

    def __call__(self, batch):
        self._ensure_assets()

        texts = [item["question"] for item in batch]
        text_enc = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        pixel_list = []
        for item in batch:
            img_list = item.get("image_urls", [])
            pil_img = img_list[0] if len(img_list) > 0 else self._black_image
            pixel_list.append(self._image_transform(pil_img))
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
            raw_weights / self.global_mean_log_rating
            if self.global_mean_log_rating > 0.0
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


class InferenceCollator:
    """Picklable wrapper that preserves item ids for inference loaders."""

    def __init__(self, base_collator: MultimodalCollator):
        self.base_collator = base_collator

    def __call__(self, batch):
        normalized_batch = []
        for item in batch:
            normalized_item = dict(item)
            normalized_item.setdefault("average_rating", 0.0)
            normalized_batch.append(normalized_item)

        result = self.base_collator(normalized_batch)
        result["item_ids"] = [item["item_id"] for item in batch]
        return result


def build_collate_fn(
    text_model_name: str = "nreimers/MiniLM-L6-H384-uncased",
    max_length: int = 256,
    image_size: int = 224,
    global_mean_log_rating: float = 1.0,
):
    """Returns a picklable collator that tokenises text and preprocesses images."""
    return MultimodalCollator(
        text_model_name=text_model_name,
        max_length=max_length,
        image_size=image_size,
        global_mean_log_rating=global_mean_log_rating,
    )


def build_inference_collate_fn(
    text_model_name: str = "nreimers/MiniLM-L6-H384-uncased",
    max_length: int = 256,
    image_size: int = 224,
):
    """Returns a picklable collator for inference and benchmarking loaders."""
    return InferenceCollator(
        build_collate_fn(
            text_model_name=text_model_name,
            max_length=max_length,
            image_size=image_size,
        )
    )


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
