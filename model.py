"""
Multimodal regression model with a torchvision EfficientNet image encoder and a
Hugging Face transformer text encoder.

Architecture:
  Image branch : EfficientNet-B0 → pooled feature vector
  Text branch  : Transformer CLS token → hidden-size feature vector
  Fusion       : concat(image, text) → MLP head → scalar(s)

Missing-image handling: `has_images` zeros the image branch so the model
degrades to text-only inference when no valid image is available.
"""

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoModel


class MultimodalRegressor(nn.Module):
    """
    EfficientNet + transformer concatenation regression model.

    Args:
        image_model_name: torchvision EfficientNet variant to load.
                          Supported: "efficientnet_b0" (1280 dim).
        text_model_name:  HuggingFace model name for the text backbone.
                          Hidden size is read from the loaded config.
        hidden_dim:       Hidden size of the MLP fusion head.
        dropout:          Dropout probability used throughout the MLP head.
        freeze_image:     If True, freeze all parameters of the image backbone.
        freeze_text:      If True, freeze all parameters of the text backbone.
        num_outputs:      Output size (1 for single-target regression).
    """

    def __init__(
        self,
        image_model_name: str = "efficientnet_b0",
        text_model_name: str = "nreimers/MiniLM-L6-H384-uncased",
        hidden_dim: int = 512,
        dropout: float = 0.2,
        freeze_image: bool = False,
        freeze_text: bool = False,
        num_outputs: int = 1,
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # Image backbone — EfficientNet-B0 via torchvision
        # Strip the classifier head; features + avgpool → 1280-dim vector
        # ------------------------------------------------------------------
        if image_model_name == "efficientnet_b0":
            _efn = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            self.image_backbone = nn.Sequential(
                _efn.features,   # conv backbone  → (B, 1280, 7, 7)
                _efn.avgpool,    # adaptive pool  → (B, 1280, 1, 1)
                nn.Flatten(1),   # flatten         → (B, 1280)
            )
            self.image_dim = 1280
        else:
            raise ValueError(
                f"Unsupported image_model_name: {image_model_name}. "
                "Only 'efficientnet_b0' is supported."
            )

        if freeze_image:
            for param in self.image_backbone.parameters():
                param.requires_grad = False

        # ------------------------------------------------------------------
        # Text backbone — configurable HuggingFace encoder
        # ------------------------------------------------------------------
        self.text_backbone = AutoModel.from_pretrained(
            text_model_name, dtype=torch.float32,
            # text_model_name,
        )
        self.text_dim = self.text_backbone.config.hidden_size  # 384

        if freeze_text:
            for param in self.text_backbone.parameters():
                param.requires_grad = False

        # ------------------------------------------------------------------
        # MLP regression head
        # Input: concat([img_feat, text_feat]) → (B, image_dim + text_dim)
        # ------------------------------------------------------------------
        fused_dim = self.image_dim + self.text_dim  # 1280 + 384 = 1664

        self.head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_outputs),
        )

        # Store for external inspection (e.g. metric_utils)
        self.hidden_dim = hidden_dim
        self.num_outputs = num_outputs

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        has_images: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values:   (B, 3, H, W)  — preprocessed image tensors.
            input_ids:      (B, L)         — token IDs.
            attention_mask: (B, L)         — attention mask.
            has_images:     (B, 1) float   — 1.0 if sample has a valid image,
                                             0.0 otherwise.  If None, all
                                             samples are assumed to have images.
        Returns:
            logits: (B, num_outputs)
        """

        # ---- Image branch ----
        img_feat = self.image_backbone(pixel_values)   # (B, image_dim)

        # Zero-out image features for samples without images
        if has_images is not None:
            img_feat = img_feat * has_images  # has_images: (B, 1) broadcasts

        # ---- Text branch ----
        text_out = self.text_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # CLS token — first token of last hidden state
        text_feat = text_out.last_hidden_state[:, 0, :]  # (B, text_dim)

        # ---- Fusion ----
        fused = torch.cat([img_feat, text_feat], dim=-1)  # (B, 1664)

        # ---- Head ----
        out = self.head(fused)  # (B, num_outputs)
        return out


# ----------------------------------------------------------------------
# Quick smoke-test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultimodalRegressor(
        freeze_image=True,
        freeze_text=False,
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params     : {total:,}")
    print(f"Trainable params : {trainable:,}")

    # Dummy batch
    B, L = 2, 64
    dummy_pixels = torch.randn(B, 3, 224, 224, device=device)
    dummy_ids = torch.randint(0, 30522, (B, L), device=device)
    dummy_mask = torch.ones(B, L, dtype=torch.long, device=device)
    dummy_has_img = torch.tensor([[1.0], [0.0]], device=device)

    with torch.no_grad():
        out = model(dummy_pixels, dummy_ids, dummy_mask, has_images=dummy_has_img)
    print(f"Output shape     : {out.shape}")   # expect (2, 1)
    print("Smoke test passed!")
