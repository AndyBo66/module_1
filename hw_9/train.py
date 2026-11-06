from pathlib import Path
import json
import pickle
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "realty_data.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "real_estate_model.pkl"
INFO_PATH = MODEL_DIR / "model_info.json"

NUMERIC_FEATURES = ["total_square", "rooms", "floor", "lat", "lon"]
CATEGORICAL_FEATURES = ["city", "district", "settlement", "object_type", "source"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "price"


def main():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    required = FEATURES + [TARGET]
    missing_columns = [col for col in required if col not in df.columns]
    if missing_columns:
        raise ValueError(f"В датасете отсутствуют столбцы: {missing_columns}")

    df = df[required].copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Убираем строки без целевой переменной и явно некорректные значения цены/площади.
    df = df.dropna(subset=[TARGET])
    df = df[(df[TARGET] > 0) & (df["total_square"] > 0)]

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].fillna("Не указано").astype(str)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=120,
            max_depth=24,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Объектов после очистки: {len(df):,}".replace(",", " "))
    print("Обучаем RandomForestRegressor...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    info = {
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "mae": float(mae),
        "r2": float(r2),
        "training_rows": int(len(df)),
        "category_options": {
            col: sorted(X[col].dropna().astype(str).unique().tolist())[:300]
            for col in CATEGORICAL_FEATURES
        },
        "numeric_defaults": {
            col: float(X[col].median()) if X[col].notna().any() else 0.0
            for col in NUMERIC_FEATURES
        }
    }
    INFO_PATH.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"MAE: {mae:,.0f} руб.".replace(",", " "))
    print(f"R2:  {r2:.4f}")
    print(f"Модель сохранена: {MODEL_PATH}")


if __name__ == "__main__":
    main()
