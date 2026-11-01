import os
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import joblib


def evaluate_models(models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series, output_dir: str, model_save_path: str):
    """Evaluate candidate regressors, plot diagnostics, and save champion model."""
    os.makedirs(output_dir, exist_ok=True)
    records = []
    predictions = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        records.append({
            "Model": name,
            "RMSE ($100k)": rmse,
            "MAE ($100k)": mae,
            "R2 Score": r2
        })
        predictions[name] = y_pred

    comparison_df = pd.DataFrame(records).sort_values(by="R2 Score", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 70)
    print(" [LEADERBOARD] CALIFORNIA HOUSING REGRESSION BENCHMARK:")
    print("=" * 70)
    print(comparison_df.to_string(index=False, formatters={
        "RMSE ($100k)": "{:.4f}".format,
        "MAE ($100k)": "{:.4f}".format,
        "R2 Score": "{:.4f}".format,
    }))
    print("=" * 70)

    best_name = comparison_df.iloc[0]["Model"]
    best_model = models[best_name]

    # Save champion model
    joblib.dump({"model_name": best_name, "model": best_model}, model_save_path)
    print(f"[Evaluation] Champion Model '{best_name}' serialized to {model_save_path}")

    # Plot 1: Actual vs Predicted (Champion)
    plt.figure(figsize=(8, 6), dpi=150)
    best_preds = predictions[best_name]
    plt.scatter(y_test * 100000, best_preds * 100000, alpha=0.3, color="#00D2BE", s=15)
    plt.plot([0, 500000], [0, 500000], color="#FF5252", linestyle="--", linewidth=2, label="Ideal Fit (y=x)")
    plt.title(f"Actual vs Predicted Prices ({best_name})", fontsize=13, fontweight="bold")
    plt.xlabel("Actual Price ($)", fontsize=11)
    plt.ylabel("Predicted Price ($)", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "actual_vs_predicted.png"))
    plt.close()

    # Plot 2: Residuals Distribution
    plt.figure(figsize=(8, 5), dpi=150)
    residuals = (y_test - best_preds) * 100000
    sns.histplot(residuals, kde=True, color="#FFA726", bins=40)
    plt.axvline(0, color="#FF5252", linestyle="--", linewidth=1.5)
    plt.title(f"Residuals Error Distribution ({best_name})", fontsize=13, fontweight="bold")
    plt.xlabel("Prediction Error / Residual ($)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residuals_plot.png"))
    plt.close()

    # Plot 3: Feature Importance
    if hasattr(best_model, "feature_importances_"):
        plt.figure(figsize=(9, 6), dpi=150)
        fi = pd.Series(best_model.feature_importances_, index=X_test.columns).sort_values(ascending=True)
        fi.plot(kind="barh", color="#58A6FF", edgecolor="#30363D")
        plt.title(f"Feature Importance Weights ({best_name})", fontsize=13, fontweight="bold")
        plt.xlabel("Importance Weight", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "feature_importance.png"))
        plt.close()

    return comparison_df, best_name
