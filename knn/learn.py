import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import joblib
import os

USING_RESULT_AS_DIFF=False

def load_data_from_csv(input_csv):
    data_df = pd.read_csv(input_csv)
    X = data_df.iloc[:, :-4].values
    y = data_df.iloc[:, -4:].values

    if USING_RESULT_AS_DIFF:
        global avg_mean, avg_std
        avg_mean = data_df['rates_mean'].values
        avg_std = data_df['rates_stdev'].values

    # Dividindo os dados em treinamento (75%) e teste (25%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    return X_train, X_test, y_train, y_test

def train_knn(X, y):
    max_n_neighbors = min(50, len(X))

    knn = KNeighborsRegressor()
    param_grid = {
        'n_neighbors': list(range(3, max_n_neighbors + 1))
    }
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_absolute_error')
    
    try:
        grid_search.fit(X, y)
    except Exception as e:
        raise e

    print(f"Melhores parâmetros: {grid_search.best_params_}")
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    if USING_RESULT_AS_DIFF:
        for i in range(len(y_pred)):
            y_pred[i, 0] += avg_mean[i]  # mean_1
            y_pred[i, 1] += avg_std[i]   # stdev_1                                                                                                 9
            y_pred[i, 2] += avg_mean[i]  # mean_2
            y_pred[i, 3] += avg_std[i]  # stdev_2

            y_test[i, 0] += avg_mean[i]  # mean_1
            y_test[i, 1] += avg_std[i]   # stdev_1
            y_test[i, 2] += avg_mean[i]  # mean_2
            y_test[i, 3] += avg_std[i]   # stdev_2
    
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f"MAPE: {mape * 100:.2f}%")
    return f"{mape * 100:.2f}"

def save_model(model, model_file):
    joblib.dump({'model': model}, model_file)
    print(f"Modelo salvo em {model_file}")

def main():
    input_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../prepared_data.csv")
    
    X_train, X_test, y_train, y_test = load_data_from_csv(input_csv)
    
    print("Treinando o modelo KNN...")
    best_knn = train_knn(X_train, y_train)
    
    print("Avaliando o modelo com MAPE...")
    mape = evaluate_model(best_knn, X_test, y_test)

    model_file = f"knn_{mape}.pkl"
    save_model(best_knn, model_file)

if __name__ == "__main__":
    main()
