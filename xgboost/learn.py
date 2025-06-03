import os
import time
import random
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, train_test_split

USING_RESULT_AS_DIFF_FROM_LAST = False

CONSIDERING_RTT_TR = False

BEST_PARAMS_XGBOOST_PATH = "best_params_xgboost.joblib" if CONSIDERING_RTT_TR else "best_params_xgboost_wortt.joblib"

def load_best_params_xgboost():
    if os.path.exists(BEST_PARAMS_XGBOOST_PATH):
        return joblib.load(BEST_PARAMS_XGBOOST_PATH)
    return {}

def save_best_params_xgboost(d):
    joblib.dump(d, BEST_PARAMS_XGBOOST_PATH)

def load_data(input_csv):
    df = pd.read_csv(input_csv)
    X = df.iloc[:, :-4].values
    y = df.iloc[:, -4:].values
    return X, y, df.columns[:-4]

def split_data(X, y, seed):
    return train_test_split(X, y, test_size=0.25, random_state=seed)

def create_output_folder():
    folder = os.path.join("output", "XGB_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(folder, exist_ok=True)
    return folder

def get_best_params_for_column_xgboost(X_train, y_train, seed, column_id, best_params_dict):
    if column_id in best_params_dict:
        return best_params_dict[column_id]
    xgb_model = XGBRegressor(random_state=seed, objective="reg:squarederror")
    param_grid = {
        "n_estimators": [100, 500],
        "max_depth": [3, 5],
        "learning_rate": [0.01, 0.1, 0.3],
        "subsample": [0.75,1.0],
        "colsample_bytree": [0.3, 0.5, 1.0]
    }
    start_time = time.perf_counter()
    gs = GridSearchCV(
        xgb_model,
        param_grid,
        cv=2,
        scoring="neg_mean_absolute_error",
        n_jobs=2,
        verbose=1
    )
    gs.fit(X_train, y_train)
    elapsed_time = time.perf_counter() - start_time
    print(f"Hyperparameter search time for column {column_id}: {elapsed_time:.3f}s")
    best_params_dict[column_id] = gs.best_params_
    save_best_params_xgboost(best_params_dict)
    return best_params_dict[column_id]

def build_xgboost_with_params(best_params, seed):
    return XGBRegressor(
        random_state=seed,
        objective="reg:squarederror",
        n_estimators=best_params.get("n_estimators"),
        max_depth=best_params.get("max_depth"),
        learning_rate=best_params.get("learning_rate"),
        subsample=best_params.get("subsample"),
        colsample_bytree=best_params.get("colsample_bytree")
    )

def evaluate_model(model, X_test, y_test, idx_last_rate, idx_last_rate_std, csv_name):
    y_pred = model.predict(X_test)
    if USING_RESULT_AS_DIFF_FROM_LAST:
        y_pred_adj = []
        y_test_adj = []
        for x_row, real_val, pred_val in zip(X_test, y_test, y_pred):
            last_mean = x_row[idx_last_rate]
            last_std = x_row[idx_last_rate_std]
            diff_base = last_std if ("std" in csv_name) else last_mean
            y_pred_adj.append(pred_val + diff_base)
            y_test_adj.append(real_val + diff_base)
        y_pred = y_pred_adj
        y_test = y_test_adj
    return mean_absolute_percentage_error(y_test, y_pred)

def feature_importance_string(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    return "<>".join([f"{feature_names[i]}={importances[i]:.4f}" for i in indices])

def save_csv(csv_path, line):
    exists = os.path.exists(csv_path)
    with open(csv_path, "a" if exists else "w", encoding="utf-8") as f:
        if not exists:
            f.write("Model;MeanOrStd?;ConsiderRTT_TR?;MAPE;Seed;FeatureImportance;MaxDepth;MaxFeatures|ColSampleByTree;MinSampleLeaf|LearningRate;MinSamplesSplit|Subsample;NumEstimators;TrainTime\n")
        f.write(line + "\n")

def execute_xgboost_training_by_column(X, y, feature_names, column, csv_name, seed, folder, best_params_dict, idx_last_rate, idx_last_rate_std):
    X_train, X_test, y_train, y_test = split_data(X, y[:, column], seed)
    col_best_params = get_best_params_for_column_xgboost(X_train, y_train, seed, column, best_params_dict)
    model = build_xgboost_with_params(col_best_params, seed)
    start_fit = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start_fit
    mape = evaluate_model(model, X_test, y_test, idx_last_rate, idx_last_rate_std, csv_name)
    feats = feature_importance_string(model, feature_names)
    line = ";".join([
        "XGB",
        csv_name.replace(".csv", ""),
        str(CONSIDERING_RTT_TR),
        f"{mape:.6f}",
        str(seed),
        feats,
        str(col_best_params.get("max_depth", "")),
        str(col_best_params.get("colsample_bytree", "")),
        str(col_best_params.get("learning_rate", "")),
        str(col_best_params.get("subsample", "")),
        str(col_best_params.get("n_estimators", "")),
        f"{fit_time:.4f}"
    ])
    save_csv(os.path.join(folder, csv_name), line)
    column_name = csv_name.replace('.csv', '')
    save_model(model, column_name, seed, mape, folder)

def save_model(model, column, seed, mape, folder):
    mape_str = f"{mape * 100:.2f}%"
    model_file = f"XGB_{column}_{seed}_{mape_str}.pkl"
    model_file = os.path.join(folder, model_file)
    joblib.dump({'model': model}, model_file)
    print(f"Modelo salvo em {model_file}")

def run_once(folder, seed):
    base_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../prepared_data.csv")
    X, y, feature_names = load_data(base_csv)
    feature_names_list = list(feature_names)
    idx_last_rate = feature_names_list.index('dash_last_rate')
    idx_last_rate_std = feature_names_list.index('dash_last_rate_std')
    best_params_dict = load_best_params_xgboost()
    execute_xgboost_training_by_column(X, y, feature_names, -4, "mean_1.csv", seed, folder, best_params_dict, idx_last_rate, idx_last_rate_std)
    execute_xgboost_training_by_column(X, y, feature_names, -3, "std_1.csv", seed, folder, best_params_dict, idx_last_rate, idx_last_rate_std)
    execute_xgboost_training_by_column(X, y, feature_names, -2, "mean_2.csv", seed, folder, best_params_dict, idx_last_rate, idx_last_rate_std)
    execute_xgboost_training_by_column(X, y, feature_names, -1, "std_2.csv", seed, folder, best_params_dict, idx_last_rate, idx_last_rate_std)

def loop_exec(n):
    folder = create_output_folder()
    seeds = [665, 616, 617, 232, 84, 230, 383, 887, 617, 531, 496]
    for seed in seeds:
        print(f"Seed: {seed}")
        run_once(folder, seed)
    
    # Calculate and print statistics after all seeds
    print("\nFinal Results (MAPE):")
    for csv_name in ["mean_1.csv", "std_1.csv", "mean_2.csv", "std_2.csv"]:
        file_path = os.path.join(folder, csv_name)
        if not os.path.exists(file_path):
            continue
            
        # Read generated CSV
        df = pd.read_csv(file_path, sep=';')
        mape_values = df['MAPE']
        
        # Calculate statistics
        mean_mape = np.mean(mape_values)
        std_mape = np.std(mape_values, ddof=1)
        
        # Print in requested format
        base_name = csv_name.replace('.csv', '')
        print(f"{base_name}: {mean_mape:.6f} ± {std_mape:.6f}")

if __name__ == "__main__":
    loop_exec(1)
