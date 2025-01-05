import pandas as pd
import numpy as np
import os
import joblib
import random
import time

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error

# 1) Import XGBRegressor
from xgboost import XGBRegressor

USING_RESULT_AS_DIFF = False
USING_RESULT_AS_DIFF_FROM_LAST = False
COLUMN_ID_LAST_RATE_MEAN = 22
COLUMN_ID_LAST_RATE_STD = 23

LEARN_ONLY_MEAN = False  # Modelo predirá apenas dois valores: mean_1 e mean_2
LEARN_ONLY_FIRST_MEAN = False  # Modelo predirá apenas um valor: mean_1
LEARN_ONLY_STDEV = True  # Modelo predirá apenas dois valores: stdev_1 e stdev_2

def load_data_from_csv(input_csv, seed):
    data_df = pd.read_csv(input_csv)

    X = data_df.iloc[:, :-4].values
    y = data_df.iloc[:, -4:].values

    if LEARN_ONLY_MEAN:
        y = data_df.iloc[:, [-4, -2]].values
    elif LEARN_ONLY_FIRST_MEAN:
        y = data_df.iloc[:, -4].values
    elif LEARN_ONLY_STDEV:
        y = data_df.iloc[:, [-3, -1]].values

    # Dividindo os dados em treinamento (75%) e teste (25%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed
    )
    print(f"Random state (seed) = {seed}")

    return X_train, X_test, y_train, y_test, data_df.columns[:-4]

# 2) Define a function to train XGBoost using GridSearchCV
def train_xgboost(X_train, y_train, seed):
    xgb_model = XGBRegressor(
        random_state=seed,
        objective="reg:squarederror"  # objective for regression
    )
    
    # Adjust these hyperparameter ranges as needed
    param_grid = {
        "n_estimators": [100],
        "max_depth": [5],
        "learning_rate": [0.1],
        "subsample": [1.0],
        "colsample_bytree": [0.3],
    }

    start_time = time.perf_counter()
    
    grid_search = GridSearchCV(
        xgb_model, 
        param_grid, 
        cv=2, 
        scoring="neg_mean_absolute_error", 
        n_jobs=2, 
        verbose=1
    )
    
    try:
        grid_search.fit(X_train, y_train)
    except Exception as e:
        raise e
    
    elapsed_time = time.perf_counter() - start_time
    print(f"Elapsed time: {elapsed_time:.3f} seconds")

    print(f"Melhores parâmetros: {grid_search.best_params_}")
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    y_pred_adj = []
    y_test_adj = []

    # Caso o resultado esteja configurado para ser a diferença do último valor
    if USING_RESULT_AS_DIFF_FROM_LAST:
        for x_row, target_row, pred_row in zip(X_test, y_test, y_pred):
            last_mean = x_row[COLUMN_ID_LAST_RATE_MEAN]
            last_std = x_row[COLUMN_ID_LAST_RATE_STD]
            if LEARN_ONLY_MEAN:
                # Se estivéssemos prevendo dois meios (ex: mean_1, mean_2)
                # Ajustando cada posição do y_pred para que o resultado final seja cumulativo
                y_pred_adj += [pred_row[0] + last_mean, pred_row[1] + last_mean]
                y_test_adj += [target_row[0] + last_mean, target_row[1] + last_mean]
            elif LEARN_ONLY_STDEV:
                # Se estivéssemos prevendo dois desvios padrão
                y_pred_adj += [pred_row[0] + last_std, pred_row[1] + last_std]
                y_test_adj += [target_row[0] + last_std, target_row[1] + last_std]
        y_pred = y_pred_adj
        y_test = y_test_adj

    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f"MAPE: {mape * 100:.2f}%")
    return mape

def print_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\nImportâncias das Features:")
    for i in indices:
        print(f"{feature_names[i]}: {importances[i]:.4f}")

def save_model(model, model_file):
    joblib.dump({"model": model}, model_file)
    print(f"Modelo salvo em {model_file}")

def rand_seed():
    return random.randint(1,1000)

def main(seed = rand_seed()):
    input_csv = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "../prepared_data.csv"
    )

    X_train, X_test, y_train, y_test, feature_names = load_data_from_csv(input_csv, seed)
    
    print("Treinando o modelo XGBoost...")
    best_xgb = train_xgboost(X_train, y_train, seed)
    
    print("Avaliando o modelo com MAPE...")
    mape = evaluate_model(best_xgb, X_test, y_test)
    mape_str = f"{mape * 100:.2f}"

    print_feature_importance(best_xgb, feature_names)
    
    suffix = "_stdevs" if LEARN_ONLY_STDEV else ""
    if LEARN_ONLY_MEAN:
        suffix = "_mean" 
    model_file = f"xgboost_model{suffix}_{mape_str}.pkl"
    # save_model(best_xgb, model_file)
    print("\n")
    return mape


def loop_exec(n_exec):
    mapes = []
    for i in range(n_exec):
        mapes.append(main(seed=rand_seed()))
    print(f"Average MAPEs: {np.mean(mapes)}")

if __name__ == "__main__":
    loop_exec(30)
