import os
from typing import Dict, Any
import pandas as pd
import joblib


class HousingPricePredictor:
    def __init__(self, model_path: str = None, pipeline_path: str = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = model_path or os.path.join(base_dir, "models", "best_model.joblib")
        self.pipeline_path = pipeline_path or os.path.join(base_dir, "models", "feature_pipeline.joblib")

        artifact = joblib.load(self.model_path)
        self.model = artifact["model"]
        self.model_name = artifact["model_name"]
        self.pipeline = joblib.load(self.pipeline_path)

    def predict_single(self, house_data: Dict[str, Any]) -> float:
        df_single = pd.DataFrame([house_data])
        X_trans = self.pipeline.transform(df_single)
        pred_scaled = float(self.model.predict(X_trans)[0])
        # MedHouseVal is in $100k units
        return round(pred_scaled * 100000, 2)


if __name__ == "__main__":
    sample_house = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.9841,
        "AveBedrms": 1.0238,
        "Population": 322.0,
        "AveOccup": 2.5555,
        "Latitude": 37.88,
        "Longitude": -122.23
    }
    predictor = HousingPricePredictor()
    price = predictor.predict_single(sample_house)
    print(f"Predicted House Value: ${price:,.2f} USD (Model: {predictor.model_name})")
