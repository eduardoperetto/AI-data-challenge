import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import joblib
import os
import numpy as np

USING_RESULT_AS_DIFF = False
USING_RESULT_AS_DIFF_FROM_LAST = False
COLUMN_ID_LAST_RATE_MEAN=22
COLUMN_ID_LAST_RATE_STD=23

LEARN_ONLY_MEAN = True  # Modelo predirá apenas dois valores: mean_1 e mean_2
LEARN_ONLY_FIRST_MEAN = False  # Modelo predirá apenas um valor: mean_1
LEARN_ONLY_STDEV = False # Modelo predirá apenas dois valores: stdev_1 e stdev_2

def load_data_from_csv(input_csv):
    data_df = pd.read_csv(input_csv)

    columns_to_keep = [
        'rates_mean', 'dash_last_rate', 'rtt0_mean', 'tr0_rtt_sum', 
        'tr0_rtt_stdev', 'dash_last_rate_std', 'dash8_rate_mean', 
        'rates_stdev', 'server_id', 'dash0_rate_mean', 
        'dash9_rate_mean', 'dash3_rate_mean'
    ]

    X = data_df.iloc[:, :-4].values
    y = data_df.iloc[:, -4:].values

    if LEARN_ONLY_MEAN:
        y = data_df.iloc[:, [-4, -2]].values
    elif LEARN_ONLY_FIRST_MEAN:
        y = data_df.iloc[:, -4].values
    elif LEARN_ONLY_STDEV:
        y = data_df.iloc[:, [-3, -1]].values

    # Dividindo os dados em treinamento (75%) e teste (25%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    return X_train, X_test, y_train, y_test, data_df.columns[:-4]

def train_random_forest(X_train, y_train):
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        'max_features': [0.3, 0.5, 0.9],
        'n_estimators': [200],
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [4, 10]
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

    y_pred_adj = []
    y_test_adj = []

    if USING_RESULT_AS_DIFF_FROM_LAST:
        for x_row in X_test:
            last_mean = x_row[COLUMN_ID_LAST_RATE_MEAN]
            last_std = x_row[COLUMN_ID_LAST_RATE_STD]
            if LEARN_ONLY_MEAN:
                y_pred_adj += [y_pred[0] + last_mean, y_pred[1] + last_mean]
                y_test_adj += [y_test[0] + last_mean, y_test[1] + last_mean]
            elif LEARN_ONLY_STDEV:
                y_pred_adj += [y_pred[0] + last_std, y_pred[1] + last_std]
                y_test_adj += [y_test[0] + last_std, y_test[1] + last_std]
        y_pred = y_pred_adj
        y_test = y_test_adj

    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f"MAPE: {mape * 100:.2f}%")
    return f"{mape * 100:.2f}"

def print_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\nImportâncias das Features:")
    for i in indices:
        print(f"{feature_names[i]}: {importances[i]:.4f}")

def save_model(model, model_file):
    joblib.dump({'model': model}, model_file)
    print(f"Modelo salvo em {model_file}")

def main():
    input_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../prepared_data.csv")

    X_train, X_test, y_train, y_test, feature_names = load_data_from_csv(input_csv)
    
    print("Treinando o modelo Random Forest...")
    best_rf = train_random_forest(X_train, y_train)
    
    print("Avaliando o modelo com MAPE...")
    mape = evaluate_model(best_rf, X_test, y_test)

    print_feature_importance(best_rf, feature_names)
    
    suffix = "_stdevs" if LEARN_ONLY_STDEV else ""
    model_file = f"random_forest_model{suffix}_{mape}.pkl"
    save_model(best_rf, model_file)

if __name__ == "__main__":
    main()
