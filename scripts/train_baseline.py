#!/usr/bin/env python3
"""
Baseline Training Script for Evidence Fusion Model
Supports single dataset training with configurable options
"""

import argparse
import os
import sys
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from transformers import XLMRobertaTokenizer
from tqdm import tqdm
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import UnifiedDataLoader
from src.data.dataset import MultimodalDataset
from src.models.experiments.evidence_fusion import EvidenceFusionModel
from src.utils.metrics import compute_metrics


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train Evidence Fusion Model")

    # Dataset
    parser.add_argument("--dataset", type=str, default="MR2",
                       choices=["AMG", "DGM4", "FineFake", "MMFakeBench", "MR2"],
                       help="Dataset to use")
    parser.add_argument("--data_root", type=str, default="data",
                       help="Root directory of datasets")
    parser.add_argument("--language", type=str, default=None,
                       help="Filter by language (en/zh), None for all")
    parser.add_argument("--exclude_unverified", action="store_true",
                       help="Exclude unverified samples (label=2) from MR2")

    # Model
    parser.add_argument("--text_model", type=str, default="xlm-roberta-base",
                       help="Text encoder model name")
    parser.add_argument("--image_model", type=str, default="openai/clip-vit-base-patch16",
                       help="Image encoder model name")
    parser.add_argument("--num_fusion_layers", type=int, default=3,
                       help="Number of fusion layers")
    parser.add_argument("--use_evidence", action="store_true",
                       help="Enable evidence image fusion")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate")

    # Training
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size per GPU")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="Warmup ratio")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Max gradient norm for clipping")
    parser.add_argument("--mixed_precision", action="store_true",
                       help="Use automatic mixed precision training")

    # Data loading
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of dataloader workers")
    parser.add_argument("--pin_memory", action="store_true",
                       help="Pin memory for faster GPU transfer")
    parser.add_argument("--prefetch_factor", type=int, default=2,
                       help="Prefetch factor for dataloader")

    # Checkpointing
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Output directory for checkpoints and logs")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                       help="Evaluate every N steps")
    parser.add_argument("--logging_steps", type=int, default=50,
                       help="Log every N steps")

    # Other
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")

    return parser.parse_args()


def create_data_filter(language=None, exclude_unverified=False):
    """Create data filter function"""
    def filter_fn(item):
        # Language filter
        if language is not None:
            if item.language != language:
                return False

        # Exclude unverified (label=2)
        if exclude_unverified and item.label == 2:
            return False

        return True

    return filter_fn


def prepare_dataloaders(args, tokenizer):
    """Prepare train/val/test dataloaders"""

    # Create data filter
    filter_fn = create_data_filter(
        language=args.language,
        exclude_unverified=args.exclude_unverified
    )

    # Load datasets
    loader = UnifiedDataLoader(data_root=args.data_root)

    print(f"\nLoading {args.dataset} dataset...")
    train_data = loader.load_dataset(args.dataset, "train", filter_fn=filter_fn)
    val_data = loader.load_dataset(args.dataset, "val", filter_fn=filter_fn)
    test_data = loader.load_dataset(args.dataset, "test", filter_fn=filter_fn)

    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")

    # Determine number of classes
    unique_labels = set([item.label for item in train_data + val_data + test_data])
    num_classes = len(unique_labels)
    print(f"Number of classes: {num_classes} (labels: {sorted(unique_labels)})")

    # Create datasets
    train_dataset = MultimodalDataset(
        train_data,
        text_tokenizer=tokenizer,
        max_text_length=512,
        use_ocr=True
    )

    val_dataset = MultimodalDataset(
        val_data,
        text_tokenizer=tokenizer,
        max_text_length=512,
        use_ocr=True
    )

    test_dataset = MultimodalDataset(
        test_data,
        text_tokenizer=tokenizer,
        max_text_length=512,
        use_ocr=True
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None
    )

    return train_loader, val_loader, test_loader, num_classes


def train_epoch(model, train_loader, optimizer, criterion, device, scaler=None, max_grad_norm=1.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        # Forward pass with mixed precision
        if scaler is not None:
            with autocast(device_type='cuda', dtype=torch.float16):
                logits = model(input_ids, attention_mask, pixel_values)
                loss = criterion(logits, labels)

            # Backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids, attention_mask, pixel_values)
            loss = criterion(logits, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        # Collect predictions
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(train_loader)
    metrics = compute_metrics(all_labels, all_preds)

    return avg_loss, metrics


def evaluate(model, data_loader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['image'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask, pixel_values)
            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    metrics = compute_metrics(all_labels, all_preds)

    return avg_loss, metrics


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.text_model}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)

    # Prepare dataloaders
    train_loader, val_loader, test_loader, num_classes = prepare_dataloaders(args, tokenizer)

    # Create model
    print(f"\nInitializing model...")
    print(f"  Text encoder: {args.text_model}")
    print(f"  Image encoder: {args.image_model}")
    print(f"  Fusion layers: {args.num_fusion_layers}")
    print(f"  Number of classes: {num_classes}")

    model = EvidenceFusionModel(
        num_classes=num_classes,
        text_model_name=args.text_model,
        image_model_name=args.image_model,
        num_fusion_layers=args.num_fusion_layers,
        use_evidence=args.use_evidence,
        dropout=args.dropout
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Setup loss
    criterion = nn.CrossEntropyLoss()

    # Setup mixed precision
    scaler = GradScaler(device='cuda') if args.mixed_precision else None
    if args.mixed_precision:
        print("\nMixed precision training enabled")

    # Training loop
    best_val_f1 = 0.0
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}\n")

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        print("-" * 60)

        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, args.max_grad_norm
        )

        print(f"\nTraining - Loss: {train_loss:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, "
              f"F1: {train_metrics['f1_macro']:.4f}")

        # Validate
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        print(f"Validation - Loss: {val_loss:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1_macro']:.4f}")

        # Save best model
        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            checkpoint_path = output_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'args': vars(args)
            }, checkpoint_path)
            print(f"Saved best model (F1: {best_val_f1:.4f})")

    # Test on best model
    print(f"\n{'='*60}")
    print("Testing on Best Model")
    print(f"{'='*60}\n")

    checkpoint = torch.load(output_dir / "best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision_macro']:.4f}")
    print(f"  Recall: {test_metrics['recall_macro']:.4f}")
    print(f"  F1 (Macro): {test_metrics['f1_macro']:.4f}")
    print(f"  F1 (Weighted): {test_metrics['f1_weighted']:.4f}")

    # Save results (convert numpy types to Python native types)
    def convert_to_python_type(obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: convert_to_python_type(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_python_type(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    results = {
        'test_loss': float(test_loss),
        'test_metrics': convert_to_python_type(test_metrics),
        'best_val_f1': float(best_val_f1)
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir / 'results.json'}")
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
