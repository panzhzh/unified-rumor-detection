#!/usr/bin/env python3
"""
Export FineFake DataFrame to JSON format

This script converts the FineFake.pkl file to JSON files (train/val/test splits)
for easier viewing and analysis.
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path


def convert_to_serializable(obj):
    """Convert numpy types to Python native types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        try:
            if pd.isna(obj):
                return None
        except (ValueError, TypeError):
            pass
    return str(obj) if obj is not None else None


def export_finefake_to_json():
    """Export FineFake DataFrame to JSON files"""

    # Load pickle file
    pkl_path = Path("data/FineFake/FineFake.pkl")
    print(f"Loading {pkl_path}...")
    df = pd.read_pickle(pkl_path)
    print(f"Loaded {len(df)} items")

    # Calculate split indices (same as data_loader.py)
    total = len(df)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)

    splits = {
        "train": df.iloc[:train_end],
        "val": df.iloc[train_end:val_end],
        "test": df.iloc[val_end:]
    }

    output_dir = Path("data/FineFake")

    for split_name, split_df in splits.items():
        print(f"\nProcessing {split_name} split ({len(split_df)} items)...")

        # Convert DataFrame to list of dicts
        data = []
        for idx, row in split_df.iterrows():
            item = {
                "id": str(idx),  # Use DataFrame index as ID
                "text": str(row['text']) if pd.notna(row['text']) else "",
                "image_path": str(row['image_path']) if pd.notna(row['image_path']) else None,
                "label": int(row['label']),  # 0=fake, 1=real
                "fine_grained_label": int(row['fine-grained label']),
                "topic": str(row['topic']) if pd.notna(row['topic']) else None,
                "platform": str(row['platform']) if pd.notna(row['platform']) else None,
                "author": convert_to_serializable(row['author']),
                "date": convert_to_serializable(row['date']),
                "entity_id": convert_to_serializable(row['entity_id']),
                "description": convert_to_serializable(row['description']),
                "relation": convert_to_serializable(row['relation']),
                "comment": convert_to_serializable(row['comment'])
            }

            # Note: knowledge_embedding is excluded (very large array)
            data.append(item)

        # Save to JSON
        output_path = output_dir / f"{split_name}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✓ Saved {len(data)} items to {output_path}")
        print(f"  Label distribution: 0(fake)={sum(1 for d in data if d['label']==0)}, "
              f"1(real)={sum(1 for d in data if d['label']==1)}")

    print("\n" + "="*60)
    print("Export Complete!")
    print("="*60)
    print(f"Files created:")
    for split_name in splits.keys():
        print(f"  - data/FineFake/{split_name}.json")
    print("\nNote: knowledge_embedding field was excluded due to size")
    print("Original pickle file preserved at: data/FineFake/FineFake.pkl")


if __name__ == "__main__":
    export_finefake_to_json()
