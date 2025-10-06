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
import yaml
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import UnifiedDataLoader
from src.data.dataset import MultimodalDataset
from src.models import MultimodalModel
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


def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file"""
    if not Path(config_path).exists():
        print(f"Warning: Config file {config_path} not found, using command-line defaults")
        return None

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


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
    parser.add_argument("--num_heads", type=int, default=8,
                       help="Number of attention heads in fusion layers")
    parser.add_argument("--use_evidence", action="store_true",
                       help="Enable evidence image fusion")
    parser.add_argument("--caption_max_length", type=int, default=128,
                       help="Max length for evidence captions")
    parser.add_argument("--dropout", type=float, default=0.3,
                       help="Dropout rate")
    parser.add_argument("--unfreeze_clip_layers", type=int, default=3,
                       help="Number of CLIP layers to unfreeze from the end")
    parser.add_argument("--hidden_dim", type=int, default=768,
                       help="Hidden dimension for fusion")
    parser.add_argument("--classifier_hidden_dim", type=int, default=512,
                       help="Hidden dimension in final classifier")

    # Training
    parser.add_argument("--num_epochs", type=int, default=8,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size per GPU")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate (for fusion and text layers)")
    parser.add_argument("--clip_lr", type=float, default=1e-6,
                       help="Learning rate for unfrozen CLIP layers")
    parser.add_argument("--weight_decay", type=float, default=0.1,
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

    # Load config file (default: config/config.yaml)
    config = load_config()
    if config is not None:
        print(f"Loading configuration from: config/config.yaml\n")

        # Override args with config values (command-line args take precedence if explicitly set)
        # Dataset
        if args.dataset == "MR2":  # If using default, load from config
            args.dataset = config['dataset']['name']
        args.data_root = config['dataset']['data_root']
        if args.language is None:
            args.language = config['dataset']['language']
        if not args.exclude_unverified:
            args.exclude_unverified = config['dataset']['exclude_unverified']

        # Model
        if args.text_model == "xlm-roberta-base":
            args.text_model = config['model']['text_model']
        if args.image_model == "openai/clip-vit-base-patch16":
            args.image_model = config['model']['image_model']
        if args.hidden_dim == 768:
            args.hidden_dim = config['model']['hidden_dim']
        if args.num_fusion_layers == 3:
            args.num_fusion_layers = config['model']['num_fusion_layers']
        if args.num_heads == 8:
            args.num_heads = config['model']['num_heads']
        if not args.use_evidence:
            args.use_evidence = config['model']['use_evidence']
        if args.caption_max_length == 128:
            args.caption_max_length = config['model']['caption_max_length']
        if args.dropout == 0.3:
            args.dropout = config['model']['dropout']
        if args.unfreeze_clip_layers == 3:
            args.unfreeze_clip_layers = config['model']['unfreeze_clip_layers']
        if args.classifier_hidden_dim == 512:
            args.classifier_hidden_dim = config['model']['classifier_hidden_dim']

        # Training
        if args.num_epochs == 8:
            args.num_epochs = config['training']['num_epochs']
        if args.batch_size == 16:
            args.batch_size = config['training']['batch_size']
        if args.learning_rate == 2e-5:
            args.learning_rate = config['training']['learning_rate']
        if args.clip_lr == 1e-6:
            args.clip_lr = config['training']['clip_lr']
        if args.weight_decay == 0.1:
            args.weight_decay = config['training']['weight_decay']
        if args.warmup_ratio == 0.1:
            args.warmup_ratio = config['training']['warmup_ratio']
        if args.max_grad_norm == 1.0:
            args.max_grad_norm = config['training']['max_grad_norm']
        if not args.mixed_precision:
            args.mixed_precision = config['training']['mixed_precision']

        # Experiment
        if args.seed == 42:
            args.seed = config['experiment']['seed']
        if args.device == "cuda":
            args.device = config['experiment']['device']
        if args.num_workers == 4:
            args.num_workers = config['experiment']['num_workers']
        if args.pin_memory:
            args.pin_memory = config['experiment']['pin_memory']
        if args.prefetch_factor == 2:
            args.prefetch_factor = config['experiment']['prefetch_factor']
        if args.output_dir == "outputs":
            args.output_dir = config['experiment']['output_dir']
        if args.logging_steps == 50:
            args.logging_steps = config['experiment']['logging_steps']
        if args.eval_steps == 500:
            args.eval_steps = config['experiment']['eval_steps']
        if args.save_steps == 500:
            args.save_steps = config['experiment']['save_steps']

    # Set seed
    set_seed(args.seed)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Create output directory with dataset + language + timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    language_str = args.language if args.language else "all"
    output_dirname = f"{args.dataset}_{language_str}_{timestamp}"
    output_dir = Path("outputs") / output_dirname
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

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
    print(f"  CLIP unfrozen layers: last {args.unfreeze_clip_layers} layers")

    model = MultimodalModel(
        num_classes=num_classes,
        text_model_name=args.text_model,
        image_model_name=args.image_model,
        num_fusion_layers=args.num_fusion_layers,
        num_heads=args.num_heads,
        use_evidence=args.use_evidence,
        caption_max_length=args.caption_max_length,
        dropout=args.dropout,
        unfreeze_clip_layers=args.unfreeze_clip_layers,
        hidden_dim=args.hidden_dim,
        classifier_hidden_dim=args.classifier_hidden_dim
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Setup optimizer with layered learning rates
    clip_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'image_encoder.clip.vision_model.encoder.layers' in name:
            # Unfrozen CLIP layers
            clip_params.append(param)
        else:
            # Fusion, text encoder, classifier
            other_params.append(param)

    param_groups = [
        {'params': clip_params, 'lr': args.clip_lr},
        {'params': other_params, 'lr': args.learning_rate}
    ]

    print(f"\nLayered learning rates:")
    print(f"  CLIP unfrozen layers: {args.clip_lr} ({sum(p.numel() for p in clip_params):,} params)")
    print(f"  Fusion/Text/Classifier: {args.learning_rate} ({sum(p.numel() for p in other_params):,} params)")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # Setup loss
    criterion = nn.CrossEntropyLoss()

    # Setup mixed precision
    scaler = GradScaler(device='cuda') if args.mixed_precision else None
    if args.mixed_precision:
        print("\nMixed precision training enabled")

    # Training loop
    best_val_metric = 0.0
    eval_metric_name = "f1"  # Evaluation metric (f1 = f1_macro by default)
    checkpoint_history = []  # Track saved checkpoints
    patience_counter = 0  # Early stopping counter

    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}")
    print(f"Evaluation strategy: After each epoch")
    print(f"Evaluation metric: F1 (macro)")
    print(f"Checkpoint limit: Keep top 2 models")
    print(f"Early stopping: Enabled (patience=3)")
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
              f"F1: {train_metrics['f1']:.4f}")

        # Evaluate after each epoch
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        print(f"Validation - Loss: {val_loss:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}")

        # Save checkpoint based on eval metric
        current_metric = val_metrics[eval_metric_name]

        if current_metric > best_val_metric:
            best_val_metric = current_metric
            patience_counter = 0  # Reset early stopping counter

            # Create checkpoint
            checkpoint_path = output_dir / f"checkpoint_epoch{epoch+1}_f1{current_metric:.4f}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'val_f1': current_metric,
                'args': vars(args)
            }, checkpoint_path)

            checkpoint_history.append({
                'path': checkpoint_path,
                'metric': current_metric,
                'epoch': epoch
            })

            # Sort by metric (descending) and keep only top 2
            checkpoint_history = sorted(checkpoint_history, key=lambda x: x['metric'], reverse=True)

            # Remove old checkpoints beyond limit (keep top 2)
            if len(checkpoint_history) > 2:
                for old_ckpt in checkpoint_history[2:]:
                    if old_ckpt['path'].exists():
                        old_ckpt['path'].unlink()
                        print(f"Removed old checkpoint: {old_ckpt['path'].name}")
                checkpoint_history = checkpoint_history[:2]

            print(f"✓ Saved checkpoint (F1: {current_metric:.4f}) - Rank: 1/{len(checkpoint_history)}")
        else:
            patience_counter += 1
            print(f"  No improvement (Best F1: {best_val_metric:.4f}) - Patience: {patience_counter}/3")

            # Early stopping
            if patience_counter >= 3:
                print(f"\n⚠ Early stopping triggered after {epoch + 1} epochs (no improvement for 3 epochs)")
                break

    # Test on best model
    print(f"\n{'='*60}")
    print("Testing on Best Model")
    print(f"{'='*60}\n")

    if len(checkpoint_history) > 0:
        # Load the best checkpoint (rank 1)
        best_checkpoint = checkpoint_history[0]
        print(f"Loading best checkpoint: {best_checkpoint['path'].name}")
        print(f"  Epoch: {best_checkpoint['epoch'] + 1}")
        print(f"  Val F1: {best_checkpoint['metric']:.4f}\n")

        checkpoint = torch.load(best_checkpoint['path'], weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Warning: No checkpoint saved, using final model state\n")

    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")

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

    # Save comprehensive results
    results = {
        'test_loss': float(test_loss),
        'test_metrics': convert_to_python_type(test_metrics),
        'best_val_f1': float(best_val_metric),
        'best_epoch': best_checkpoint['epoch'] + 1 if len(checkpoint_history) > 0 else args.num_epochs,
        'total_epochs': args.num_epochs,
        'eval_metric': 'f1',
        'saved_checkpoints': [
            {
                'path': str(ckpt['path'].name),
                'epoch': ckpt['epoch'] + 1,
                'val_f1': float(ckpt['metric'])
            }
            for ckpt in checkpoint_history
        ]
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir / 'results.json'}")
    print(f"\nSaved checkpoints:")
    for i, ckpt in enumerate(checkpoint_history, 1):
        print(f"  {i}. {ckpt['path'].name} (F1: {ckpt['metric']:.4f})")

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
