from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "realty_data.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "real_estate_model.joblib"

TARGET = "price"
NUMERIC_FEATURES = ["total_square", "rooms", "floor", "lat", "lon"]
CATEGORICAL_FEATURES = ["city", "district", "settlement"]

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Не найден файл {DATA_PATH}. Поместите realty_data.csv в папку data."
        )

    # В файле есть русские тексты; utf-8-sig также корректно читает UTF-8 с BOM.
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    required = [TARGET] + NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing_columns = [col for col in required if col not in df.columns]
    if missing_columns:
        raise ValueError(f"В CSV отсутствуют обязательные столбцы: {missing_columns}")

    # Целевая переменная должна быть числовой и положительной.
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] > 0].copy()

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].astype(float)

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Неизвестно")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    model = Pipeline(steps=[
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

    print(f"Всего объектов: {len(df):,}")
    print("Обучаем RandomForestRegressor...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"MAE: {mae:,.0f} рублей")
    print(f"R2:  {r2:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Модель сохранена: {MODEL_PATH}")

if __name__ == "__main__":
    main()
