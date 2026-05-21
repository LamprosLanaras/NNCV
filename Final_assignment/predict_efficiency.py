"""
Unified Prediction Pipeline for the Efficiency Track.
Supports: 'fastscnn' (Baseline), 'c_fastscnn' (Compressed), and 'kd_c_fastscnn' (Knowledge Distillation).
"""
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision.transforms.v2 import (
    Compose, 
    ToImage, 
    Resize, 
    ToDtype, 
    Normalize,
    InterpolationMode,
)

# Import both architectures
from models.efficiency.fast_scnn_baseline import FastSCNN
from models.efficiency.c_fast_scnn import CFastSCNN


def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for Efficiency models.")
    
    parser.add_argument("--variant", type=str, default="kd_c_fastscnn", 
                        choices=["fastscnn", "c_fastscnn", "kd_c_fastscnn"],
                        help="Which model variant to evaluate.")
    parser.add_argument("--weights_path", type=str, default="/app/model.pt",
                        help="Path to the trained .pt weights file.")
    parser.add_argument("--input_dir", type=str, default="/data",
                        help="Directory containing input images.")
    parser.add_argument("--output_dir", type=str, default="/output",
                        help="Directory to save predicted masks.")
    
    return parser.parse_args()


def preprocess(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB")
    
    # Downscale the image to 256x512 for high-speed inference
    transform = Compose([
        ToImage(),
        Resize(size=(256, 512), interpolation=InterpolationMode.BILINEAR), 
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    img_tensor = transform(img)
    return img_tensor.unsqueeze(0)


def postprocess(pred: torch.Tensor, original_shape: tuple) -> np.ndarray:
    # Extract class with highest logit score
    pred_max = torch.argmax(pred, dim=1, keepdim=True).to(torch.float32) 
    
    # Restore original resolution using nearest neighbor to preserve discrete class labels
    prediction = Resize(size=original_shape, interpolation=InterpolationMode.NEAREST)(pred_max)
    prediction_numpy = prediction.cpu().detach().numpy().squeeze() 
    
    return prediction_numpy


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing {args.variant} on {device}...")
    
    # 1. Dynamically instantiate the correct architecture
    if args.variant == "fastscnn":
        model = FastSCNN(n_classes=19)
    else:
        # Both C-FastSCNN and KD-C-FastSCNN use the compressed architecture
        model = CFastSCNN(n_classes=19)
        
    # 2. Load Weights
    state_dict = torch.load(args.weights_path, map_location=device, weights_only=True)
    
    # Filter out auxiliary layer weights and unexpected fusion biases
    state_dict = {
        k: v for k, v in state_dict.items()
        if not k.startswith("auxlayer.") and k not in [
            "feature_fusion.conv_lower_res.0.bias",
            "feature_fusion.conv_higher_res.0.bias"
        ]
    }

    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    # 3. Inference Loop
    image_files = list(Path(args.input_dir).glob("*.png"))
    print(f"Found {len(image_files)} images to process.")
    
    with torch.no_grad():
        for img_path in image_files:
            img = Image.open(img_path)
            original_shape = np.array(img).shape[:2]

            img_tensor = preprocess(img).to(device)
            pred = model(img_tensor)
            
            seg_pred = postprocess(pred, original_shape)

            out_path = Path(args.output_dir) / img_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(seg_pred.astype(np.uint8)).save(out_path)
            
    print("Inference complete.")


if __name__ == "__main__":
    main()