import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import joblib
import os

USING_RESULT_AS_DIFF = True

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

def train_random_forest(X_train, y_train):
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4, 10]
    }
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
    
    try:
        grid_search.fit(X_train, y_train)
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
    
    print("Treinando o modelo Random Forest...")
    best_rf = train_random_forest(X_train, y_train)
    
    print("Avaliando o modelo com MAPE...")
    mape = evaluate_model(best_rf, X_test, y_test)
    
    model_file = f"random_forest_model_{mape}.pkl"
    save_model(best_rf, model_file)

if __name__ == "__main__":
    main()
