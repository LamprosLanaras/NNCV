"""
Unified Prediction Pipeline for the Peak Performance Track.
Supports: 'baseline', 'augsegformer', 'uperformer', and 'auxlovasz_uperformer'.
Automatically routes the correct architecture, weight loading, and Test-Time Augmentation (TTA) strategy.
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.transforms.v2 import Compose, ToImage, ToDtype, Normalize

# Import all architectures
from models.peak_performance.segformer import SegFormerB5
from models.peak_performance.uperformer import UperFormer
from models.peak_performance.aux_lovasz_uperformer import AuxLovaszUperFormer


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Inference for Peak Performance Models.")
    parser.add_argument("--variant", type=str, default="auxlovasz_uperformer", 
                        choices=["baseline", "augsegformer", "uperformer", "auxlovasz_uperformer"],
                        help="Which model variant to evaluate.")
    parser.add_argument("--weights_path", type=str, default="/app/model.pt",
                        help="Path to the trained .pt weights file.")
    parser.add_argument("--input_dir", type=str, default="/data",
                        help="Directory containing input images.")
    parser.add_argument("--output_dir", type=str, default="/output",
                        help="Directory to save predicted masks.")
    
    return parser.parse_args()


def preprocess(img: Image.Image) -> torch.Tensor:
    transform = Compose([
        ToImage(),
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return transform(img.convert("RGB")).unsqueeze(0)


def predict_single_scale(model: torch.nn.Module, img_tensor: torch.Tensor) -> torch.Tensor:
    """Standard forward pass for SegFormer Baseline."""
    logits = model(img_tensor)
    return F.softmax(logits, dim=1)


def predict_multi_scale_tta(model: torch.nn.Module, img_tensor: torch.Tensor, scales: list = [0.75, 1.0, 1.25]) -> torch.Tensor:
    """Multi-scale + Horizontal Flip Test-Time Augmentation."""
    _, _, h, w = img_tensor.shape
    device = img_tensor.device
    final_prob = torch.zeros((1, 19, h, w), device=device)
    
    for scale in scales:
        target_h, target_w = int(h * scale), int(w * scale)
        scaled_img = F.interpolate(img_tensor, size=(target_h, target_w), mode="bilinear", align_corners=False)
        
        # 1. Predict on scaled image
        logits = model(scaled_img)
        logits_restored = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        prob = F.softmax(logits_restored, dim=1)
        
        # 2. Predict on flipped scaled image
        img_flipped = torch.flip(scaled_img, dims=[3]) 
        logits_flipped = model(img_flipped)
        logits_flipped_restored = F.interpolate(logits_flipped, size=(h, w), mode="bilinear", align_corners=False)
        prob_flipped = F.softmax(logits_flipped_restored, dim=1)
        prob_flipped_back = torch.flip(prob_flipped, dims=[3])
        
        # Accumulate
        final_prob += (prob + prob_flipped_back) / 2.0
        
    return final_prob / len(scales)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing {args.variant} on {device}...")

    # 1. Dynamically instantiate architecture
    if args.variant in ["baseline", "augsegformer"]:
        model = SegFormerB5(in_channels=3, n_classes=19, use_pretrained_weights=False)
    elif args.variant == "uperformer":
        model = UperFormer(in_channels=3, n_classes=19, use_pretrained_weights=False)
    elif args.variant == "auxlovasz_uperformer":
        model = AuxLovaszUperFormer(in_channels=3, n_classes=19, use_pretrained_weights=False)

    # 2. Load Weights (Safely filtering auxiliary layers if they exist)
    state_dict = torch.load(args.weights_path, map_location=device, weights_only=True)
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("auxlayer.")}
    
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    # 3. Inference Loop
    image_files = list(Path(args.input_dir).glob("*.png"))
    print(f"Found {len(image_files)} images.")
    
    with torch.no_grad():
        for img_path in image_files:
            img = Image.open(img_path)
            img_tensor = preprocess(img).to(device)

            # Route the TTA strategy
            if args.variant == "baseline":
                prob_map = predict_single_scale(model, img_tensor)
            else:
                # All advanced models use multi-scale TTA
                prob_map = predict_multi_scale_tta(model, img_tensor)

            # Postprocess and save
            pred_max = torch.argmax(prob_map, dim=1, keepdim=True) 
            seg_pred = pred_max.cpu().detach().numpy().squeeze() 
            
            out_path = Path(args.output_dir) / img_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(seg_pred.astype(np.uint8)).save(out_path)
            
    print("Inference complete.")


if __name__ == "__main__":
    main()