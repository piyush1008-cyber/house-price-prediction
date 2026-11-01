from typing import Dict, Any
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def train_regressors(X_train, y_train) -> Dict[str, Any]:
    """Train multiple regression models on housing dataset."""
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(n_estimators=80, max_depth=15, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    }

    trained = {}
    print("\n[ModelTraining] Fitting candidate regression architectures:")
    for name, model in models.items():
        print(f"   -> Training {name}...")
        model.fit(X_train, y_train)
        trained[name] = model

    print("[ModelTraining] All regression models trained successfully.")
    return trained
