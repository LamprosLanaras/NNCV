"""
SegFormer-B5 architecture for the Baseline and AugSegFormer models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation


class SegFormerB5(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        n_classes: int = 19,
        use_pretrained_weights: bool = False,
    ):
        super().__init__()

        if in_channels != 3:
            raise ValueError(f"SegFormer requires 3 input channels, got {in_channels}.")

        if use_pretrained_weights:
            # Used during training to leverage ImageNet-1k pre-trained weights
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/mit-b5",
                num_labels=n_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            # Used during inference/evaluation when internet access is unavailable
            config = SegformerConfig(
                num_channels=in_channels,
                num_labels=n_classes,
                depths=[3, 6, 40, 3],
                hidden_sizes=[64, 128, 320, 512],
                decoder_hidden_size=768,
            )
            self.model = SegformerForSemanticSegmentation(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        outputs = self.model(pixel_values=x)
        logits = outputs.logits

        # Restore logits to the original image resolution
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(
                logits,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )

        return logits