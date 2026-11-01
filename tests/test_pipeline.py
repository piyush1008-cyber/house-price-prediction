import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from data_loader import load_housing_data
from feature_engineering import HousingFeaturePipeline
from model_training import train_regressors


def test_data_loader():
    df = load_housing_data()
    assert len(df) == 20640
    assert "MedHouseVal" in df.columns
    assert df.isnull().sum().sum() == 0


def test_feature_pipeline():
    df = load_housing_data().head(100)
    pipeline = HousingFeaturePipeline()
    X_scaled, y = pipeline.fit_transform(df)
    assert X_scaled.shape[0] == 100
    assert "RoomsPerHousehold" in X_scaled.columns
    assert "BedroomsPerRoom" in X_scaled.columns

    # Test transform on unseen single record
    single = df.iloc[[0]].drop(columns=["MedHouseVal"])
    trans_single = pipeline.transform(single)
    assert trans_single.shape == (1, X_scaled.shape[1])


def test_model_training_fast():
    df = load_housing_data().head(200)
    pipeline = HousingFeaturePipeline()
    X_scaled, y = pipeline.fit_transform(df)
    models = train_regressors(X_scaled, y)
    assert "Linear Regression" in models
    assert "Random Forest" in models
    preds = models["Linear Regression"].predict(X_scaled)
    assert len(preds) == 200
