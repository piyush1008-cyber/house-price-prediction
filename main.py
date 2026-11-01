import os
import sys
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from data_loader import load_housing_data
from eda import generate_eda_plots
from feature_engineering import prepare_datasets
from model_training import train_regressors
from evaluation import evaluate_models
from predict import HousingPricePredictor


def main():
    parser = argparse.ArgumentParser(description="California Housing Price Regression Pipeline")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA plot generation")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, "data")
    vis_dir = os.path.join(project_root, "visualizations")
    model_dir = os.path.join(project_root, "models")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 70)
    print("      CALIFORNIA HOUSING PRICE REGRESSION PIPELINE")
    print("      Author: Peeyoosh Kangle")
    print("=" * 70)

    # 1. Load Data
    t0 = time.time()
    df = load_housing_data()

    # 2. EDA
    if not args.skip_eda:
        print("\n>>> Running Exploratory Data Analysis & generating charts...")
        generate_eda_plots(df, vis_dir)

    # 3. Feature Engineering
    print("\n>>> Transforming features & splitting train/test sets...")
    X_train, X_test, y_train, y_test, pipeline = prepare_datasets(df, model_dir)

    # 4. Train Models
    models = train_regressors(X_train, y_train)

    # 5. Evaluate & Save Best Model
    model_save_path = os.path.join(model_dir, "best_model.joblib")
    comparison_df, best_model_name = evaluate_models(models, X_test, y_test, vis_dir, model_save_path)

    # 6. Inference Demo
    print("\n>>> Demonstrating Real-Time Property Valuation Inference:")
    sample_block = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.9841,
        "AveBedrms": 1.0238,
        "Population": 322.0,
        "AveOccup": 2.5555,
        "Latitude": 37.88,
        "Longitude": -122.23
    }
    predictor = HousingPricePredictor(model_save_path, os.path.join(model_dir, "feature_pipeline.joblib"))
    val = predictor.predict_single(sample_block)
    print(f"   [INFERENCE] Sample Property Valuation: ${val:,.2f} USD (Champion: {best_model_name})")

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f" Pipeline executed successfully in {elapsed:.2f} seconds!")
    print("=" * 70)


if __name__ == "__main__":
    main()
