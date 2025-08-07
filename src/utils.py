import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def prepare_data(df, features, target_name='price'):
    target = df[target_name]
    X = df[features].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, target, test_size=0.2, random_state=42
    )

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    X_train_processed = numeric_transformer.fit_transform(X_train)
    X_test_processed = numeric_transformer.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test, numeric_transformer

def train_model(X_train, y_train):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, transformer, model_path=None, transformer_path=None):
    if model_path is None:
        model_path = os.path.join(BASE_DIR, 'model.pkl')
    if transformer_path is None:
        transformer_path = os.path.join(BASE_DIR, 'transformer.pkl')
    joblib.dump(model, model_path)
    joblib.dump(transformer, transformer_path)

def load_model(model_path=None, transformer_path=None):
    if model_path is None:
        model_path = os.path.join(BASE_DIR, 'model.pkl')
    if transformer_path is None:
        transformer_path = os.path.join(BASE_DIR, 'transformer.pkl')
    model = joblib.load(model_path)
    transformer = joblib.load(transformer_path)
    return model, transformer