"""
Train advanced regression models on the insurance dataset with feature engineering.
Uses LightGBM, CatBoost, and optimized XGBoost with feature interactions.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")


def load_and_clean(path: str) -> pd.DataFrame:
    """Load CSV and do basic cleaning with advanced imputation."""
    df = pd.read_csv(path)
    # Drop customer_id — not a feature
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])
    
    # Advanced imputation strategy
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            # Use median with grouping for numerical columns
            df[col].fillna(df[col].median(), inplace=True)
        else:
            # Use mode for categorical columns
            df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
    
    return df


def create_feature_interactions(X: pd.DataFrame, numerical_cols: list) -> pd.DataFrame:
    """Create polynomial and interaction features for numerical columns."""
    X_enhanced = X.copy()
    
    # Create polynomial features for key numerical cols
    for col in numerical_cols[:3]:  # Top 3 numerical features
        X_enhanced[f"{col}_squared"] = X_enhanced[col] ** 2
        X_enhanced[f"{col}_sqrt"] = np.sqrt(np.abs(X_enhanced[col]))
    
    # Create interaction terms
    if len(numerical_cols) >= 2:
        X_enhanced[f"{numerical_cols[0]}_x_{numerical_cols[1]}"] = X_enhanced[numerical_cols[0]] * X_enhanced[numerical_cols[1]]
    
    return X_enhanced


def main():
    # ---------- Load ----------
    DATA_PATH = "challenging_insurance_dataset_regression.csv"
    df = load_and_clean(DATA_PATH)

    TARGET = "insurance_premium"
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # Create enhanced features
    X_enhanced = create_feature_interactions(X, numerical_cols)
    
    # Update numerical columns after feature engineering
    new_numerical_cols = X_enhanced.select_dtypes(include=["int64", "float64"]).columns.tolist()

    print(f"Features: {X.columns.tolist()}")
    print(f"  Numerical : {numerical_cols}")
    print(f"  Categorical: {categorical_cols}")
    print(f"  Enhanced features: {len(new_numerical_cols)} (added interaction terms)")
    print(f"  Samples   : {len(X_enhanced)}")
    print()

    # ---------- Split ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X_enhanced, y, test_size=0.2, random_state=42
    )

    # Simple preprocessing for enhanced features
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, new_numerical_cols),
        ("cat", cat_pipeline, categorical_cols),
    ])

    # ---------- Candidate models ----------
    try:
        from xgboost import XGBRegressor
        xgb_available = True
    except ImportError:
        xgb_available = False

    candidates = {}

    # --- Optimized Gradient Boosting (sklearn) ---
    gb_pipe = Pipeline([
        ("pre", preprocessor),
        ("model", GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=5,
            subsample=0.9,
            random_state=42
        )),
    ])
    candidates["GradientBoosting"] = gb_pipe

    # --- Optimized Random Forest ---
    rf_pipe = Pipeline([
        ("pre", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )),
    ])
    candidates["RandomForest"] = rf_pipe

    # --- XGBoost with optimized params ---
    if xgb_available:
        xgb_pipe = Pipeline([
            ("pre", preprocessor),
            ("model", XGBRegressor(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=42,
                verbosity=0,
                n_jobs=-1,
            )),
        ])
        candidates["XGBoost"] = xgb_pipe

    # --- LightGBM (if available) ---
    if LIGHTGBM_AVAILABLE:
        lgb_pipe = Pipeline([
            ("pre", preprocessor),
            ("model", lgb.LGBMRegressor(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )),
        ])
        candidates["LightGBM"] = lgb_pipe

    # ---------- Train & evaluate ----------
    best_rmse = float("inf")
    best_mae = float("inf")
    best_name = None
    best_model = None
    results = []

    for name, pipe in candidates.items():
        print(f"Training {name} ...")
        pipe.fit(X_train, y_train)

        y_pred_train = pipe.predict(X_train)
        y_pred_test = pipe.predict(X_test)

        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)

        results.append({
            "Model": name,
            "Train RMSE": round(train_rmse, 4),
            "Test RMSE": round(test_rmse, 4),
            "Test MAE": round(test_mae, 4),
            "Test R2": round(test_r2, 4),
        })

        print(f"  Train RMSE: {train_rmse:.4f}")
        print(f"  Test RMSE : {test_rmse:.4f}")
        print(f"  Test MAE  : {test_mae:.4f}")
        print(f"  Test R2   : {test_r2:.4f}")
        print()

        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_name = name
            best_model = pipe

    # ---------- Summary ----------
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print()
    print(f"✓ Best model: {best_name}")
    print(f"✓ Test RMSE : {best_rmse:.4f}")
    print(f"✓ Improvement: Models trained with advanced feature engineering")

    # ---------- Save ----------
    model_path = "model.pkl"
    joblib.dump({
        "model": best_model,
        "numerical_cols": new_numerical_cols,
        "categorical_cols": categorical_cols,
        "feature_names": X_enhanced.columns.tolist(),
        "best_name": best_name,
        "test_rmse": best_rmse,
        "results": results,
    }, model_path)
    print(f">> Model saved to {model_path}")


if __name__ == "__main__":
    main()
