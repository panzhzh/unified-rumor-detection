"""
Comprehensive Data Analysis Script
Validates the unified data loader and analyzes all datasets

Output: outputs/data_analysis/
- All analysis results in one place
- All plots in English
- Multi-perspective analysis
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict, List

from src.data import UnifiedDataLoader
from src.config import AVAILABLE_DATASETS

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Output directory
OUTPUT_DIR = project_root / "outputs" / "data_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class DatasetAnalyzer:
    """Unified dataset analyzer using single data loader entry point"""

    def __init__(self, data_root: str = "data"):
        """Initialize with unified data loader - THE ONLY DATA ENTRY POINT"""
        self.loader = UnifiedDataLoader(data_root=data_root)
        self.datasets = AVAILABLE_DATASETS
        self.analysis_results = {}

    def load_all_data(self):
        """Load all datasets through unified loader"""
        print("=" * 70)
        print("Loading All Datasets via Unified Data Loader")
        print("=" * 70)

        all_data = {}
        for dataset_name in self.datasets:
            print(f"\n[{dataset_name}] Loading...")
            try:
                # Load all splits
                train_data = self.loader.load_dataset(dataset_name, "train")
                val_data = self.loader.load_dataset(dataset_name, "val")
                test_data = self.loader.load_dataset(dataset_name, "test")

                all_data[dataset_name] = {
                    'train': train_data,
                    'val': val_data,
                    'test': test_data
                }

                total = len(train_data) + len(val_data) + len(test_data)
                print(f"  ✓ Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
                print(f"  ✓ Total: {total} items")

            except Exception as e:
                print(f"  ✗ Error loading {dataset_name}: {e}")
                all_data[dataset_name] = None

        return all_data

    def analyze_basic_statistics(self, all_data: Dict):
        """Analysis 1: Basic Statistics"""
        print("\n" + "=" * 70)
        print("Analysis 1: Basic Dataset Statistics")
        print("=" * 70)

        stats_data = []

        for dataset_name, splits in all_data.items():
            if splits is None:
                continue

            for split_name, data in splits.items():
                if not data:
                    continue

                # Calculate statistics
                total = len(data)
                fake_count = sum(1 for item in data if item.label == 1)
                real_count = sum(1 for item in data if item.label == 0)
                unverified_count = sum(1 for item in data if item.label == 2)

                has_text = sum(1 for item in data if item.has_text())
                has_image = sum(1 for item in data if item.has_image())
                has_ocr = sum(1 for item in data if item.has_ocr())
                is_multimodal = sum(1 for item in data if item.is_multimodal())

                stats_data.append({
                    'Dataset': dataset_name,
                    'Split': split_name,
                    'Total': total,
                    'Real': real_count,
                    'Fake': fake_count,
                    'Unverified': unverified_count,
                    'Fake_Ratio': fake_count / total if total > 0 else 0,
                    'Has_Text': has_text,
                    'Has_Image': has_image,
                    'Has_OCR': has_ocr,
                    'Multimodal': is_multimodal
                })

        df = pd.DataFrame(stats_data)

        # Save to CSV
        csv_path = OUTPUT_DIR / "basic_statistics.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved statistics to: {csv_path}")

        # Print summary
        print("\nDataset Summary:")
        print(df.to_string(index=False))

        return df

    def plot_dataset_sizes(self, df: pd.DataFrame):
        """Visualization 1: Dataset sizes comparison"""
        print("\nGenerating dataset size comparison plots...")

        # Aggregate by dataset
        dataset_totals = df.groupby('Dataset')['Total'].sum().sort_values(ascending=False)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Plot 1: Total samples per dataset
        ax1 = axes[0]
        bars = ax1.bar(range(len(dataset_totals)), dataset_totals.values,
                       color=sns.color_palette("husl", len(dataset_totals)))
        ax1.set_xticks(range(len(dataset_totals)))
        ax1.set_xticklabels(dataset_totals.index, rotation=45, ha='right')
        ax1.set_ylabel('Number of Samples', fontsize=12)
        ax1.set_title('Total Sample Count by Dataset', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, dataset_totals.values)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{int(value)}', ha='center', va='bottom', fontsize=10)

        # Plot 2: Split distribution
        split_data = df.pivot_table(index='Dataset', columns='Split', values='Total', aggfunc='sum', fill_value=0)
        split_data = split_data[['train', 'val', 'test']]  # Ensure order

        ax2 = axes[1]
        split_data.plot(kind='bar', stacked=True, ax=ax2,
                       color=['#3498db', '#2ecc71', '#e74c3c'])
        ax2.set_ylabel('Number of Samples', fontsize=12)
        ax2.set_title('Train/Val/Test Split Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Dataset', fontsize=12)
        ax2.legend(title='Split', loc='upper right')
        ax2.grid(axis='y', alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        save_path = OUTPUT_DIR / "dataset_sizes.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved plot: {save_path}")

    def plot_label_distribution(self, all_data: Dict):
        """Visualization 2: Label distribution (Real vs Fake)"""
        print("\nGenerating label distribution plots...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, (dataset_name, splits) in enumerate(all_data.items()):
            if splits is None or idx >= 5:
                continue

            ax = axes[idx]

            # Combine all splits
            all_items = []
            for split_data in splits.values():
                all_items.extend(split_data)

            # Count labels
            labels = [item.label for item in all_items]
            unique_labels, counts = np.unique(labels, return_counts=True)

            # Label names
            label_names = []
            for label in unique_labels:
                if label == 0:
                    label_names.append('Real')
                elif label == 1:
                    label_names.append('Fake')
                else:
                    label_names.append('Unverified')

            # Plot
            colors = ['#2ecc71' if l == 0 else '#e74c3c' if l == 1 else '#f39c12'
                     for l in unique_labels]
            bars = ax.bar(label_names, counts, color=colors, alpha=0.8, edgecolor='black')

            ax.set_title(f'{dataset_name}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Count', fontsize=10)
            ax.grid(axis='y', alpha=0.3)

            # Add percentage labels
            total = sum(counts)
            for bar, count in zip(bars, counts):
                percentage = count / total * 100
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01,
                       f'{int(count)}\n({percentage:.1f}%)',
                       ha='center', va='bottom', fontsize=9)

        # Remove empty subplot
        if len(all_data) < 6:
            fig.delaxes(axes[5])

        plt.suptitle('Label Distribution Across Datasets', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()

        save_path = OUTPUT_DIR / "label_distribution.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved plot: {save_path}")

    def plot_modality_analysis(self, df: pd.DataFrame):
        """Visualization 3: Modality analysis (Text/Image/Multimodal)"""
        print("\nGenerating modality analysis plots...")

        # Aggregate by dataset
        modality_data = df.groupby('Dataset')[['Has_Text', 'Has_Image', 'Has_OCR', 'Multimodal']].sum()

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Modality availability heatmap
        ax1 = axes[0]
        sns.heatmap(modality_data.T, annot=True, fmt='d', cmap='YlGnBu',
                   ax=ax1, cbar_kws={'label': 'Sample Count'})
        ax1.set_title('Modality Availability by Dataset', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Modality Type', fontsize=12)
        ax1.set_xlabel('Dataset', fontsize=12)

        # Plot 2: Multimodal percentage
        ax2 = axes[1]
        dataset_totals = df.groupby('Dataset')['Total'].sum()
        multimodal_counts = df.groupby('Dataset')['Multimodal'].sum()
        multimodal_ratio = (multimodal_counts / dataset_totals * 100).sort_values(ascending=False)

        bars = ax2.barh(range(len(multimodal_ratio)), multimodal_ratio.values,
                       color=sns.color_palette("rocket_r", len(multimodal_ratio)))
        ax2.set_yticks(range(len(multimodal_ratio)))
        ax2.set_yticklabels(multimodal_ratio.index)
        ax2.set_xlabel('Multimodal Percentage (%)', fontsize=12)
        ax2.set_title('Multimodal Data Percentage', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

        # Add percentage labels
        for i, (bar, value) in enumerate(zip(bars, multimodal_ratio.values)):
            ax2.text(value + 1, bar.get_y() + bar.get_height()/2,
                    f'{value:.1f}%', va='center', fontsize=10)

        plt.tight_layout()
        save_path = OUTPUT_DIR / "modality_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved plot: {save_path}")

    def validate_data_loader(self, all_data: Dict):
        """Validation: Check data loader correctness"""
        print("\n" + "=" * 70)
        print("Validation: Unified Data Loader Correctness Check")
        print("=" * 70)

        validation_results = []

        for dataset_name, splits in all_data.items():
            if splits is None:
                continue

            print(f"\n[{dataset_name}] Validating...")

            checks = {
                'has_unique_ids': True,
                'has_valid_labels': True,
                'has_dataset_name': True,
                'has_split_info': True,
                'text_field_correct': True,
                'image_field_correct': True,
                'label_in_range': True
            }

            all_ids = set()

            for split_name, data in splits.items():
                for item in data:
                    # Check 1: Unique IDs
                    if item.id in all_ids:
                        checks['has_unique_ids'] = False
                    all_ids.add(item.id)

                    # Check 2: Valid labels (0, 1, or 2)
                    if item.label not in [0, 1, 2]:
                        checks['has_valid_labels'] = False
                        checks['label_in_range'] = False

                    # Check 3: Dataset name matches
                    if item.dataset_name != dataset_name:
                        checks['has_dataset_name'] = False

                    # Check 4: Split info matches
                    if item.split != split_name:
                        checks['has_split_info'] = False

                    # Check 5: Text field consistency
                    if item.has_text() and (item.text is None or len(item.text.strip()) == 0):
                        checks['text_field_correct'] = False

                    # Check 6: Image field consistency
                    if item.has_image() and item.image_path is None:
                        checks['image_field_correct'] = False

            # Print results
            all_passed = all(checks.values())
            status = "✓ PASS" if all_passed else "✗ FAIL"
            print(f"  {status}")

            for check_name, passed in checks.items():
                symbol = "✓" if passed else "✗"
                print(f"    {symbol} {check_name.replace('_', ' ').title()}")

            validation_results.append({
                'Dataset': dataset_name,
                **checks,
                'All_Passed': all_passed
            })

        # Save validation report
        val_df = pd.DataFrame(validation_results)
        csv_path = OUTPUT_DIR / "validation_report.csv"
        val_df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved validation report: {csv_path}")

        # Summary
        total_datasets = len(validation_results)
        passed_datasets = sum(1 for r in validation_results if r['All_Passed'])

        print(f"\nValidation Summary:")
        print(f"  Total datasets: {total_datasets}")
        print(f"  Passed: {passed_datasets}")
        print(f"  Failed: {total_datasets - passed_datasets}")

        if passed_datasets == total_datasets:
            print("\n✅ ALL DATASETS PASSED VALIDATION!")
            print("✅ UNIFIED DATA LOADER IS WORKING CORRECTLY!")
        else:
            print("\n⚠️  Some datasets failed validation. Check the report.")

        return val_df

    def run_analysis(self):
        """Run complete analysis pipeline"""
        print("\n" + "=" * 70)
        print("UNIFIED DATASET ANALYSIS")
        print("Data Entry Point: UnifiedDataLoader (SINGLE SOURCE OF TRUTH)")
        print("=" * 70)

        # Step 1: Load all data via unified loader
        all_data = self.load_all_data()

        # Step 2: Basic statistics
        df = self.analyze_basic_statistics(all_data)

        # Step 3: Visualizations
        self.plot_dataset_sizes(df)
        self.plot_label_distribution(all_data)
        self.plot_modality_analysis(df)

        # Step 4: Validation
        validation_df = self.validate_data_loader(all_data)

        # Final summary
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"Output directory: {OUTPUT_DIR}")
        print("\nGenerated files:")
        for file in OUTPUT_DIR.iterdir():
            print(f"  - {file.name}")
        print("\n✅ Data loader validation complete!")


if __name__ == "__main__":
    analyzer = DatasetAnalyzer(data_root="data")
    analyzer.run_analysis()