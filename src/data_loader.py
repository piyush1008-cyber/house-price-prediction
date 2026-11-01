import os
import pandas as pd


def load_housing_data(csv_path: str = None) -> pd.DataFrame:
    """Load California Housing dataset from CSV or sklearn fallback."""
    if csv_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, "data", "california_housing.csv")

    if not os.path.exists(csv_path):
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing(as_frame=True)
        df = data.frame
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path)

    # Basic data integrity checks
    assert not df.empty, "Dataset must not be empty"
    assert "MedHouseVal" in df.columns, "Target column 'MedHouseVal' missing"
    assert df.isnull().sum().sum() == 0, "Null values detected"

    print(f"[DataLoader] Successfully loaded {len(df)} records with {df.shape[1]} columns.")
    return df
