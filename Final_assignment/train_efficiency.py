"""
Unified training script for the Efficiency Track.
Supports: 'fastscnn' (Baseline), 'c_fastscnn' (Compressed), and 'kd_c_fastscnn' (Knowledge Distillation).
"""
import os
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.datasets import Cityscapes
from torchvision.utils import make_grid
from torchvision.transforms import v2
from torchvision.transforms.v2 import InterpolationMode

from models.efficiency.fast_scnn_baseline import FastSCNN
from models.efficiency.c_fast_scnn import CFastSCNN
from models.peak_performance.segformer import SegFormerB5 # Teacher model

id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}

def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid.get(x, 255))

train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)
    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id
        for i in range(3):
            color_image[:, i][mask] = color[i]
    return color_image


class JointTransforms:
    def __init__(self, is_train=True):
        if is_train:
            self.spatial = v2.Compose([
                v2.RandomResizedCrop(size=(512, 1024), scale=(0.5, 2.0)),
                v2.RandomHorizontalFlip(p=0.5),
            ])
            self.color = v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        else:
            self.spatial = v2.Resize((512, 1024), interpolation=InterpolationMode.BILINEAR)
            self.color = nn.Identity()

        self.normalize = v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def __call__(self, img, target):
        img = tv_tensors.Image(img)
        target = tv_tensors.Mask(target)
        img, target = self.spatial(img, target)
        img = self.color(img)
        img = v2.functional.to_dtype(img, torch.float32, scale=True)
        img = self.normalize(img)
        target = v2.functional.to_dtype(target, torch.int64)
        return img, target.as_subclass(torch.Tensor)


def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """Calculates the KL Divergence between Student and Teacher probability distributions."""
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    
    # KLDivergence loss expects input in log-space and target in prob-space
    kl_div = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    
    # Scale by T^2 as proposed by Hinton et al.
    return kl_div * (temperature ** 2)


def get_args_parser():
    parser = ArgumentParser("Unified Training Script for Efficiency Track")
    parser.add_argument("--variant", type=str, default="kd_c_fastscnn", choices=["fastscnn", "c_fastscnn", "kd_c_fastscnn"], help="Model variant to train")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to data")
    parser.add_argument("--batch-size", type=int, default=12, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.045, help="Base learning rate")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--experiment-id", type=str, default="efficiency-run", help="WandB ID")
    
    # Distillation specific arguments
    parser.add_argument("--teacher-weights", type=str, default="./weights/baseline.pt", help="Path to Teacher (SegFormer) weights")
    parser.add_argument("--temperature", type=float, default=2.0, help="KD Temperature")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for KD loss (vs standard CE loss)")
    return parser


def main(args):
    wandb.init(project="5lsm0-cityscapes-segmentation", name=args.experiment_id, config=vars(args))
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = Cityscapes(args.data_dir, split="train", mode="fine", target_type="semantic", transforms=JointTransforms(is_train=True))
    valid_dataset = Cityscapes(args.data_dir, split="val", mode="fine", target_type="semantic", transforms=JointTransforms(is_train=False))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 1. Instantiate the Student Model
    print(f"Initializing {args.variant} on {device}...")
    if args.variant == "fastscnn":
        model = FastSCNN(n_classes=19, aux=True).to(device)
    else:
        model = CFastSCNN(n_classes=19, aux=True).to(device)

    # 2. Instantiate the Teacher Model (ONLY if using KD)
    teacher_model = None
    if args.variant == "kd_c_fastscnn":
        print(f"Initializing Teacher Model (SegFormerB5) from {args.teacher_weights}...")
        teacher_model = SegFormerB5(in_channels=3, n_classes=19, use_pretrained_weights=False).to(device)
        teacher_model.load_state_dict(torch.load(args.teacher_weights, map_location=device, weights_only=True), strict=True)
        teacher_model.eval() # Freeze teacher
        for param in teacher_model.parameters():
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=4e-5)
    total_iters = args.epochs * len(train_dataloader)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_iters, power=0.9)
    scaler = GradScaler()

    best_valid_loss = float("inf")
    current_best_model_path = None

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04} - Variant: {args.variant}")

        # --- Training Loop ---
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):
            labels = convert_to_train_id(labels)
            images, labels = images.to(device), labels.to(device)
            labels = labels.squeeze(1)

            optimizer.zero_grad()

            with autocast(dtype=torch.bfloat16):
                outputs, aux_outputs = model(images)
                
                # Standard Cross-Entropy Loss
                loss_ce_main = criterion(outputs, labels)
                loss_ce_aux = criterion(aux_outputs, labels)
                standard_loss = loss_ce_main + (0.4 * loss_ce_aux)

                if args.variant == "kd_c_fastscnn":
                    # Generate Soft Labels from Teacher
                    with torch.no_grad():
                        teacher_logits = teacher_model(images)
                    
                    # Calculate Distillation Loss
                    kd_loss = distillation_loss(outputs, teacher_logits, temperature=args.temperature)
                    
                    # Blend losses using alpha
                    loss = (1.0 - args.alpha) * standard_loss + (args.alpha * kd_loss)
                    
                    if i % 50 == 0:
                        wandb.log({"train_loss_standard": standard_loss.item(), "train_loss_kd": kd_loss.item()})
                else:
                    loss = standard_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if i % 50 == 0:
                wandb.log({
                    "train_loss_total": loss.item(),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch": epoch + 1,
                }, step=epoch * len(train_dataloader) + i)

        # --- Validation Loop ---
        model.eval()
        with torch.no_grad():
            losses = []
            for i, (images, labels) in enumerate(valid_dataloader):
                labels = convert_to_train_id(labels)
                images, labels = images.to(device), labels.to(device)
                labels = labels.squeeze(1)

                # Validation always uses single output (aux head is disabled in eval mode)
                outputs = model(images)
                loss = criterion(outputs, labels)
                losses.append(loss.item())

                if i == 0:
                    predictions = outputs.softmax(1).argmax(1).unsqueeze(1)
                    labels_vis = labels.unsqueeze(1)

                    predictions = convert_train_id_to_color(predictions)
                    labels_vis = convert_train_id_to_color(labels_vis)

                    predictions_img = make_grid(predictions.cpu(), nrow=4).permute(1, 2, 0).numpy()
                    labels_img = make_grid(labels_vis.cpu(), nrow=4).permute(1, 2, 0).numpy()

                    wandb.log({
                        "predictions": [wandb.Image(predictions_img)],
                        "labels": [wandb.Image(labels_img)],
                    }, step=(epoch + 1) * len(train_dataloader) - 1)

            valid_loss = sum(losses) / len(losses)
            wandb.log({"valid_loss": valid_loss}, step=(epoch + 1) * len(train_dataloader) - 1)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if current_best_model_path and os.path.exists(current_best_model_path):
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(output_dir, f"best_{args.variant}-epoch={epoch:04}-val_loss={valid_loss:04.4f}.pt")
                torch.save(model.state_dict(), current_best_model_path)

    print("Training complete!")
    torch.save(model.state_dict(), os.path.join(output_dir, f"final_{args.variant}-epoch={args.epochs:04}.pt"))
    wandb.finish()

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)