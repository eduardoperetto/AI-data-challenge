import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import joblib

def load_data_from_csv(input_csv):
    data_df = pd.read_csv(input_csv)
    X = data_df.iloc[:, :-4].values
    y = data_df.iloc[:, -4:].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

def train_random_forest(X, y):
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    
    try:
        grid_search.fit(X, y)
    except Exception as e:
        raise e

    print(f"Melhores parâmetros: {grid_search.best_params_}")
    return grid_search.best_estimator_

def evaluate_model(model, X, y_true):
    y_pred = model.predict(X)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f"MAPE: {mape * 100:.2f}%")
    return mape

def save_model(model, scaler, model_file):
    joblib.dump({'model': model, 'scaler': scaler}, model_file)
    print(f"Modelo salvo em {model_file}")

def main():
    input_csv = "prepared_data.csv"
    model_file = "random_forest_model.pkl"

    X_scaled, y, scaler = load_data_from_csv(input_csv)
    
    print("Treinando o modelo Random Forest...")
    best_rf = train_random_forest(X_scaled, y)
    
    print("Avaliando o modelo com MAPE...")
    evaluate_model(best_rf, X_scaled, y)
    
    save_model(best_rf, scaler, model_file)

if __name__ == "__main__":
    main()
