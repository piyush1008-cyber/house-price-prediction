import os
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


class HousingFeaturePipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df = df.copy()
        # Domain engineered ratios
        df["RoomsPerHousehold"] = df["AveRooms"] / np.maximum(df["AveOccup"], 0.1)
        df["BedroomsPerRoom"] = df["AveBedrms"] / np.maximum(df["AveRooms"], 0.1)

        y = df["MedHouseVal"]
        X = df.drop(columns=["MedHouseVal"])

        self.feature_names = X.columns.tolist()
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=self.feature_names, index=X.index)
        return X_scaled, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["RoomsPerHousehold"] = df["AveRooms"] / np.maximum(df["AveOccup"], 0.1)
        df["BedroomsPerRoom"] = df["AveBedrms"] / np.maximum(df["AveRooms"], 0.1)

        if "MedHouseVal" in df.columns:
            X = df.drop(columns=["MedHouseVal"])
        else:
            X = df

        X_aligned = X.reindex(columns=self.feature_names, fill_value=0)
        return pd.DataFrame(self.scaler.transform(X_aligned), columns=self.feature_names, index=X_aligned.index)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)


def prepare_datasets(df: pd.DataFrame, model_dir: str):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    pipeline = HousingFeaturePipeline()
    X_train, y_train = pipeline.fit_transform(train_df)
    X_test = pipeline.transform(test_df)
    y_test = test_df["MedHouseVal"]

    pipeline.save(os.path.join(model_dir, "feature_pipeline.joblib"))
    print(f"[FeatureEngineering] Engineered {X_train.shape[1]} features across {len(X_train)} training records.")
    return X_train, X_test, y_train, y_test, pipeline
