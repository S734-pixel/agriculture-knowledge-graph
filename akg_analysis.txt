# akg_analysis.py
# Quantitative EDA for Crop_recommendationV2 dataset
# - Clean, robust, script-friendly (no notebook-only features)
# - Matplotlib only (no seaborn)
# - Saves all figures to outputs/ and also shows them

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams.update({"figure.autolayout": True})

# -----------------------------
# Config
# -----------------------------
# Set this path to your dataset. Supports .csv, .xlsx
DATA_PATH = r"C:\Users\sidds\Downloads\Crop_recommendationV2.csv.xlsx"
OUTPUT_DIR = "outputs"

# -----------------------------
# Helpers
# -----------------------------
def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)

def load_dataset(path: str) -> pd.DataFrame:
    path_lower = path.lower()
    if path_lower.endswith(".csv"):
        df = pd.read_csv(path)
    elif path_lower.endswith(".xlsx") or path_lower.endswith(".xls"):
        # Requires openpyxl for .xlsx
        # If openpyxl isn't installed, install via: pip install openpyxl
        df = pd.read_excel(path)
    else:
        raise ValueError("Unsupported file type. Use .csv or .xlsx")
    return df

def numeric_columns(df: pd.DataFrame):
    return df.select_dtypes(include=[np.number]).columns.tolist()

def save_and_show(fig, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=150)
    print(f"[Saved] {filepath}")
    plt.show()

def boxplot_by_label(df: pd.DataFrame, value_col: str, label_col: str = "label", title: str = ""):
    labels = sorted(df[label_col].unique())
    data = [df.loc[df[label_col] == lab, value_col].dropna().values for lab in labels]
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(data, showfliers=True, patch_artist=False)
    ax.set_title(title if title else f"{value_col} by {label_col}")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel(value_col)
    return fig

def histogram(fig_title: str, series: pd.Series, bins: int = 30):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    ax.hist(series.dropna().values, bins=bins)
    ax.set_title(fig_title)
    ax.set_xlabel(series.name)
    ax.set_ylabel("Count")
    return fig

def correlation_heatmap(df_numeric: pd.DataFrame, title: str = "Correlation Heatmap"):
    corr = df_numeric.corr()
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    cax = ax.imshow(corr.values, interpolation="nearest", aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    fig.colorbar(cax)
    return fig, corr

# -----------------------------
# Main analysis
# -----------------------------
def main():
    ensure_output_dir(OUTPUT_DIR)

    # 1) Load
    df = load_dataset(DATA_PATH)

    # 2) Basic info
    print("=== DATASET INFO ===")
    print(df.info())
    print("\n=== FIRST 5 ROWS ===")
    print(df.head())

    # 3) Clean/prepare references
    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column for crop names, but it was not found.")

    num_cols = numeric_columns(df)
    if not num_cols:
        raise ValueError("No numeric columns found for quantitative analysis.")

    # 4) Descriptive statistics (numeric only)
    print("\n=== DESCRIPTIVE STATISTICS (numeric) ===")
    desc = df[num_cols].describe()
    print(desc)

    # Save descriptive stats
    desc_path = os.path.join(OUTPUT_DIR, "descriptive_statistics.csv")
    desc.to_csv(desc_path, index=True)
    print(f"[Saved] {desc_path}")

    # 5) Histograms (as previously shown: temperature, humidity, ph, rainfall, soil_moisture, crop_density)
    # Only plot columns that actually exist
    hist_targets = ["temperature", "humidity", "ph", "rainfall", "soil_moisture", "crop_density"]
    for col in hist_targets:
        if col in df.columns and col in num_cols:
            fig = histogram(f"{col.capitalize()} Distribution", df[col])
            save_and_show(fig, f"hist_{col}.png")
        else:
            print(f"[Skip] Column '{col}' not found or not numeric; histogram not generated.")

    # 6) Correlation heatmap (numeric only)
    fig, corr = correlation_heatmap(df[num_cols], "Correlation Heatmap of Numeric Features")
    save_and_show(fig, "correlation_heatmap.png")

    # Also save correlation matrix as CSV
    corr_path = os.path.join(OUTPUT_DIR, "correlation_matrix.csv")
    corr.to_csv(corr_path, index=True)
    print(f"[Saved] {corr_path}")

    # 7) Crop-wise analysis (means per label)
    crop_means = df.groupby("label")[num_cols].mean().sort_index()
    print("\n=== CROP-WISE MEANS (numeric) ===")
    print(crop_means.head())

    crop_means_path = os.path.join(OUTPUT_DIR, "crop_wise_means.csv")
    crop_means.to_csv(crop_means_path, index=True)
    print(f"[Saved] {crop_means_path}")

    # 8) Boxplots by crop type for: temperature, rainfall, soil_moisture
    box_targets = ["temperature", "rainfall", "soil_moisture"]
    for col in box_targets:
        if col in df.columns and col in num_cols:
            fig = boxplot_by_label(df, value_col=col, label_col="label",
                                   title=f"{col.capitalize()} Distribution by Crop Type")
            save_and_show(fig, f"box_{col}_by_label.png")
        else:
            print(f"[Skip] Column '{col}' not found or not numeric; boxplot not generated.")

    print("\nAll analyses completed successfully.")
    print(f"Charts saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
