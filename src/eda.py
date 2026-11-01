import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def set_dark_style():
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except Exception:
        plt.style.use("dark_background")


def generate_eda_plots(df: pd.DataFrame, output_dir: str):
    """Generate and persist publication-quality exploratory visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    set_dark_style()

    # 1. Target Price Distribution
    plt.figure(figsize=(9, 5), dpi=150)
    sns.histplot(df["MedHouseVal"] * 100000, kde=True, color="#00D2BE", bins=40)
    plt.title("California Median House Value Distribution ($)", fontsize=14, fontweight="bold")
    plt.xlabel("House Price ($)", fontsize=11)
    plt.ylabel("Block Group Count", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "price_distribution.png"))
    plt.close()

    # 2. Correlation Matrix Heatmap
    plt.figure(figsize=(10, 8), dpi=150)
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, linewidths=0.5)
    plt.title("Correlation Matrix of Housing Features", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
    plt.close()

    # 3. Geospatial Price Map
    plt.figure(figsize=(10, 8), dpi=150)
    scatter = plt.scatter(
        df["Longitude"],
        df["Latitude"],
        c=df["MedHouseVal"] * 100000,
        cmap="plasma",
        alpha=0.4,
        s=df["Population"] / 100,
        label="Population Density"
    )
    plt.colorbar(scatter, label="Median House Value ($)")
    plt.title("California Geospatial Housing Value & Population Map", fontsize=14, fontweight="bold")
    plt.xlabel("Longitude", fontsize=11)
    plt.ylabel("Latitude", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "geospatial_price_map.png"))
    plt.close()

    print(f"[EDA] Generated 3 visual plots in {output_dir}")
