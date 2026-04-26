"""
Unified training script for the Peak Performance Track.
Supports all progressive variants: 'baseline', 'augsegformer', 'uperformer', 
and the final 'auxlovasz_uperformer'.
"""
import os
import random
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.datasets import Cityscapes
from torchvision.utils import make_grid
from torchvision.transforms import v2
from torchvision.transforms.v2 import InterpolationMode
from transformers import get_linear_schedule_with_warmup

# Ensure these files exist in your models/peak_performance/ directory!
from models.peak_performance.segformer import SegFormerB5
from models.peak_performance.uperformer import UperFormer
from models.peak_performance.aux_lovasz_uperformer import AuxLovaszUperFormer

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


class IndependentChannelNoise(nn.Module):
    def __init__(self, scale_limit=0.15, shift_limit=0.1, p=0.5):
        super().__init__()
        self.scale_limit = scale_limit
        self.shift_limit = shift_limit
        self.p = p

    def forward(self, img):
        if random.random() > self.p:
            return img
        c = img.shape[0]
        for i in range(c):
            scale = 1.0 + (random.random() * 2 - 1.0) * self.scale_limit
            shift = (random.random() * 2 - 1.0) * self.shift_limit
            img[i] = img[i] * scale + shift
        return torch.clamp(img, 0.0, 1.0)


class JointTransforms:
    def __init__(self, is_train=True, use_augmentations=False):
        self.is_train = is_train
        if is_train and use_augmentations:
            self.spatial = v2.Compose([
                v2.RandomResizedCrop(size=(768, 768), scale=(0.5, 2.0)),
                v2.RandomHorizontalFlip(p=0.5),
            ])
            self.color_jitter = v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
            self.anti_shortcut = IndependentChannelNoise(scale_limit=0.15, shift_limit=0.1, p=0.5)
        elif is_train and not use_augmentations:
            self.spatial = v2.RandomCrop(size=(768, 768))
            self.color_jitter = nn.Identity()
            self.anti_shortcut = nn.Identity()
        else:
            self.spatial = v2.Resize((1024, 2048), interpolation=InterpolationMode.BILINEAR)
            self.color_jitter = nn.Identity()
            self.anti_shortcut = nn.Identity()

        self.normalize = v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def __call__(self, img, target):
        img = tv_tensors.Image(img)
        target = tv_tensors.Mask(target)
        img, target = self.spatial(img, target)
        img = self.color_jitter(img)
        img = v2.functional.to_dtype(img, torch.float32, scale=True)
        img = self.anti_shortcut(img)
        img = self.normalize(img)
        target = v2.functional.to_dtype(target, torch.int64)
        return img, target.as_subclass(torch.Tensor)

# --- LOSS FUNCTIONS ---

def ohem_cross_entropy(logits, targets, ignore_index=255, keep_ratio=0.7):
    loss = F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction='none').view(-1)
    valid_loss = loss[targets.view(-1) != ignore_index]
    if valid_loss.numel() > 0:
        k = int(keep_ratio * valid_loss.numel())
        if k > 0:
            topk_loss, _ = torch.topk(valid_loss, k)
            return topk_loss.mean()
        return valid_loss.mean()
    return loss.sum() * 0.0

def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_softmax_flat(probs, labels, classes='present'):
    if probs.numel() == 0:
        return probs * 0.
    C = probs.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()
        if classes == 'present' and fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probs[:, 0]
        else:
            class_pred = probs[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    return sum(losses) / len(losses)

def lovasz_softmax(logits, labels, ignore_index=255):
    probs = F.softmax(logits, dim=1)
    B, C, H, W = probs.shape
    probs = probs.permute(0, 2, 3, 1).reshape(-1, C) 
    labels = labels.view(-1)
    
    valid_mask = (labels != ignore_index)
    probs = probs[valid_mask]
    labels = labels[valid_mask]
    
    if probs.numel() == 0:
        return logits.sum() * 0.
    return lovasz_softmax_flat(probs, labels)

# --- METRICS ---

class MetricTracker:
    def __init__(self, num_classes=19):
        self.num_classes = num_classes
        self.hist = torch.zeros((num_classes, num_classes))

    def update(self, preds, labels):
        valid_mask = (labels != 255) & (labels >= 0) & (labels < self.num_classes)
        preds = preds[valid_mask]
        labels = labels[valid_mask]
        k = labels * self.num_classes + preds
        bincount = torch.bincount(k, minlength=self.num_classes**2)
        self.hist += bincount.reshape(self.num_classes, self.num_classes).cpu()

    def get_scores(self):
        tp = torch.diag(self.hist)
        fp = self.hist.sum(dim=0) - tp
        fn = self.hist.sum(dim=1) - tp
        valid_classes = self.hist.sum(dim=1) > 0
        iou = tp[valid_classes] / (tp[valid_classes] + fp[valid_classes] + fn[valid_classes] + 1e-6)
        dice = 2 * tp[valid_classes] / (2 * tp[valid_classes] + fp[valid_classes] + fn[valid_classes] + 1e-6)
        return iou.mean().item(), dice.mean().item()

# --- MAIN RUNNER ---

def get_args_parser():
    parser = ArgumentParser("Unified Training Script for Peak Performance Track")
    parser.add_argument("--variant", type=str, default="auxlovasz_uperformer", 
                        choices=["baseline", "augsegformer", "uperformer", "auxlovasz_uperformer"], 
                        help="Model variant to train")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.00006)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-id", type=str, default="peak-performance-run")
    parser.add_argument("--ohem-keep-ratio", type=float, default=0.7)
    return parser

def main(args):
    wandb.init(project="5lsm0-cityscapes-segmentation", name=args.experiment_id, config=vars(args))
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_aug = (args.variant in ["augsegformer", "uperformer", "auxlovasz_uperformer"])
    
    train_dataset = Cityscapes(args.data_dir, split="train", mode="fine", target_type="semantic", transforms=JointTransforms(is_train=True, use_augmentations=use_aug))
    valid_dataset = Cityscapes(args.data_dir, split="val", mode="fine", target_type="semantic", transforms=JointTransforms(is_train=False))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Dynamically instantiate architecture
    if args.variant in ["baseline", "augsegformer"]:
        model = SegFormerB5(in_channels=3, n_classes=19, use_pretrained_weights=True).to(device)
    elif args.variant == "uperformer":
        model = UperFormer(in_channels=3, n_classes=19, use_pretrained_weights=True).to(device)
    elif args.variant == "auxlovasz_uperformer":
        model = AuxLovaszUperFormer(in_channels=3, n_classes=19, use_pretrained_weights=True).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_dataloader) * args.epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_combined_score = 0.0
    current_best_model_path = None

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04} - Variant: {args.variant}")

        model.train()
        for i, (images, labels) in enumerate(train_dataloader):
            labels = convert_to_train_id(labels)
            images, labels = images.to(device), labels.to(device)
            labels = labels.long().squeeze(1)

            optimizer.zero_grad()
            outputs = model(images)

            # Route loss function dynamically based on variant
            if args.variant == "auxlovasz_uperformer":
                main_logits, aux_logits = outputs
                
                main_loss_ohem = ohem_cross_entropy(main_logits, labels, ignore_index=255, keep_ratio=args.ohem_keep_ratio)
                main_loss_lovasz = lovasz_softmax(main_logits, labels, ignore_index=255)
                main_loss = (0.5 * main_loss_ohem) + (0.5 * main_loss_lovasz)

                aux_loss_ohem = ohem_cross_entropy(aux_logits, labels, ignore_index=255, keep_ratio=args.ohem_keep_ratio)
                aux_loss_lovasz = lovasz_softmax(aux_logits, labels, ignore_index=255)
                aux_loss = (0.5 * aux_loss_ohem) + (0.5 * aux_loss_lovasz)

                loss = main_loss + (0.4 * aux_loss)
                
                if i % 50 == 0:
                    wandb.log({"train_loss_main": main_loss.item(), "train_loss_aux": aux_loss.item()})
                    
            elif args.variant in ["augsegformer", "uperformer"]:
                loss = ohem_cross_entropy(outputs, labels, ignore_index=255, keep_ratio=args.ohem_keep_ratio)
            else:
                loss = F.cross_entropy(outputs, labels, ignore_index=255)

            loss.backward()
            optimizer.step()
            scheduler.step()

            if i % 50 == 0:
                wandb.log({
                    "train_loss_total": loss.item(),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch": epoch + 1,
                }, step=epoch * len(train_dataloader) + i)

        model.eval()
        metric_tracker = MetricTracker(num_classes=19)
        val_losses = []
        
        with torch.no_grad():
            for i, (images, labels) in enumerate(valid_dataloader):
                labels = convert_to_train_id(labels)
                images, labels = images.to(device), labels.to(device)
                labels = labels.long().squeeze(1)

                # Eval mode automatically disables aux head, returning only main logits
                outputs = model(images) 
                
                loss_ce = F.cross_entropy(outputs, labels, ignore_index=255)
                val_losses.append(loss_ce.item())

                preds = outputs.argmax(dim=1)
                metric_tracker.update(preds, labels)

                if i == 0:
                    preds_vis = convert_train_id_to_color(preds.unsqueeze(1))
                    labels_vis = convert_train_id_to_color(labels.unsqueeze(1))
                    wandb.log({
                        "predictions": [wandb.Image(make_grid(preds_vis.cpu(), nrow=4).permute(1, 2, 0).numpy())],
                        "labels": [wandb.Image(make_grid(labels_vis.cpu(), nrow=4).permute(1, 2, 0).numpy())],
                    }, step=(epoch + 1) * len(train_dataloader) - 1)

            mean_iou, mean_dice = metric_tracker.get_scores()
            combined_score = (mean_iou + mean_dice) / 2.0
            avg_val_loss = sum(val_losses) / len(val_losses)

            wandb.log({
                "valid_loss": avg_val_loss,
                "valid_mIoU": mean_iou,
                "valid_mDice": mean_dice,
                "valid_combined_score": combined_score
            }, step=(epoch + 1) * len(train_dataloader) - 1)

            if combined_score > best_combined_score:
                print(f"New best combined score! mIoU: {mean_iou:.4f}, mDice: {mean_dice:.4f}")
                best_combined_score = combined_score
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(output_dir, f"best_{args.variant}-epoch={epoch:04}-score={combined_score:.4f}.pt")
                torch.save(model.state_dict(), current_best_model_path)

    print("Training complete!")
    torch.save(model.state_dict(), os.path.join(output_dir, f"final_{args.variant}-epoch={args.epochs:04}.pt"))
    wandb.finish()

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)