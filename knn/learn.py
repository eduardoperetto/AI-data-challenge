import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
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

def train_knn(X, y):
    max_n_neighbors = min(10, len(X))

    knn = KNeighborsRegressor()
    param_grid = {'n_neighbors': list(range(3, max_n_neighbors + 1))}
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)

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
    model_file = "knn_model.pkl"

    X_scaled, y, scaler = load_data_from_csv(input_csv)
    
    print("Treinando o modelo KNN...")
    best_knn = train_knn(X_scaled, y)
    
    print("Avaliando o modelo com MAPE...")
    evaluate_model(best_knn, X_scaled, y)
    
    save_model(best_knn, scaler, model_file)

if __name__ == "__main__":
    main()
