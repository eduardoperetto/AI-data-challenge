import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import joblib

def load_data_from_csv(input_csv):
    data_df = pd.read_csv(input_csv)
    X = data_df.iloc[:, :-4].values
    y = data_df.iloc[:, -4:].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dividindo os dados em treinamento (75%) e teste (25%)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

    return X_train, X_test, y_train, y_test, scaler

def train_random_forest(X_train, y_train):
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 6]
    }
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    
    try:
        grid_search.fit(X_train, y_train)
    except Exception as e:
        raise e

    print(f"Melhores parâmetros: {grid_search.best_params_}")
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f"MAPE: {mape * 100:.2f}%")
    return f"{mape * 100:.2f}"

def save_model(model, scaler, model_file):
    joblib.dump({'model': model, 'scaler': scaler}, model_file)
    print(f"Modelo salvo em {model_file}")

def main():
    input_csv = "prepared_data.csv"

    X_train, X_test, y_train, y_test, scaler = load_data_from_csv(input_csv)
    
    print("Treinando o modelo Random Forest...")
    best_rf = train_random_forest(X_train, y_train)
    
    print("Avaliando o modelo com MAPE...")
    mape = evaluate_model(best_rf, X_test, y_test)
    model_file = f"random_forest_model_{mape}.pkl"
    
    save_model(best_rf, scaler, model_file)

if __name__ == "__main__":
    main()
