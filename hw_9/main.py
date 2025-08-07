import pandas as pd
from src.utils import prepare_data, train_model, save_model

def main():
    df = pd.read_csv('realty_data.csv')
    features = ['total_square', 'rooms', 'floor']  # признаки
    X_train, X_test, y_train, y_test, transformer = prepare_data(df, features, target_name='price')
    model = train_model(X_train, y_train)

    print("Train R^2:", model.score(X_train, y_train))
    print("Test R^2:", model.score(X_test, y_test))

    save_model(model, transformer)

if __name__ == "__main__":
    main()