import os
import pandas as pd
import numpy as np
import random
import time
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error
import joblib

USING_RESULT_AS_DIFF_FROM_LAST = True
CONSIDERING_RTT_TR = False
BEST_PARAMS_PATH = "best_params_withrtt.joblib" if CONSIDERING_RTT_TR else "best_params_rf_wo.joblib"

def load_best_params():
    if os.path.exists(BEST_PARAMS_PATH):
        return joblib.load(BEST_PARAMS_PATH)
    return {}

def save_best_params(d):
    joblib.dump(d, BEST_PARAMS_PATH)

def load_data(input_csv):
    df = pd.read_csv(input_csv)
    X = df.iloc[:, :-4].values
    y = df.iloc[:, -4:].values
    return X, y, df.columns[:-4]

def split_data(X, y, seed):
    return train_test_split(X, y, test_size=0.25, random_state=seed)

def create_main_output_folder():
    base_or_enriched = 'enriched' if CONSIDERING_RTT_TR else 'base'
    folder = os.path.join("output", "saving_csv", base_or_enriched, "RF", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(folder, exist_ok=True)
    return folder

def get_best_params_for_column(X_train, y_train, seed, column_id, best_params_dict):
    if column_id in best_params_dict:
        return best_params_dict[column_id]
    rf = RandomForestRegressor(random_state=seed)
    param_grid = {
        'max_features': [0.3, "sqrt"],
        'n_estimators': [500],
        'max_depth': [None, 20],
        'min_samples_split': [2],
        'min_samples_leaf': [1, 10]
    }
    start_time = time.perf_counter()
    gs = GridSearchCV(rf, param_grid, cv=2, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    elapsed_time = time.perf_counter() - start_time
    print(f"Hyperparameter search time for column {column_id}: {elapsed_time:.3f}s")
    best_params_dict[column_id] = gs.best_params_
    save_best_params(best_params_dict)
    return best_params_dict[column_id]

def build_model_with_params(best_params, seed):
    return RandomForestRegressor(
        random_state=seed,
        max_depth=best_params.get('max_depth'),
        max_features=best_params.get('max_features'),
        n_estimators=best_params.get('n_estimators'),
        min_samples_split=best_params.get('min_samples_split'),
        min_samples_leaf=best_params.get('min_samples_leaf')
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

def execute_training_by_column(X_train, X_test, y_train, y_test, feature_names, column, csv_name, seed, folder, best_params_dict, idx_last_rate, idx_last_rate_std):
    col_best_params = get_best_params_for_column(X_train, y_train, seed, column, best_params_dict)
    model = build_model_with_params(col_best_params, seed)
    start_fit = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start_fit
    mape = evaluate_model(model, X_test, y_test, idx_last_rate, idx_last_rate_std, csv_name)
    feats = feature_importance_string(model, feature_names)
    line = ";".join([
        "RF",
        csv_name.replace(".csv", ""),
        str(CONSIDERING_RTT_TR),
        f"{mape:.6f}",
        str(seed),
        feats,
        str(col_best_params.get("max_depth", "")),
        str(col_best_params.get("max_features", "")),
        str(col_best_params.get("min_samples_leaf", "")),
        str(col_best_params.get("min_samples_split", "")),
        str(col_best_params.get("n_estimators", "")),
        f"{fit_time:.4f}"
    ])
    save_csv(os.path.join(folder, csv_name), line)
    column_name = csv_name.replace('.csv', '')
    save_model(model, column_name, seed, mape, folder)

def save_model(model, column, seed, mape, folder):
    mape_str = f"{mape * 100:.2f}%"
    model_file = f"RF_{column}_{seed}_{mape_str}.pkl"
    model_file = os.path.join(folder, model_file)
    joblib.dump({'model': model}, model_file)
    print(f"Modelo salvo em {model_file}")

def run_once(seed_folder, seed):
    base_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../prepared_data.csv")
    X, y, feature_names = load_data(base_csv)
    feature_names_list = list(feature_names)
    idx_last_rate = feature_names_list.index('dash_last_rate')
    idx_last_rate_std = feature_names_list.index('dash_last_rate_std')
    
    X_train, X_test, y_train_all, y_test_all = train_test_split(X, y, test_size=0.25, random_state=seed)
    
    pd.DataFrame(X_train, columns=feature_names).to_csv(os.path.join(seed_folder, "X_train.csv"), index=False)
    pd.DataFrame(X_test, columns=feature_names).to_csv(os.path.join(seed_folder, "X_test.csv"), index=False)
    
    targets = {
        -4: "mean_1",
        -3: "std_1",
        -2: "mean_2",
        -1: "std_2"
    }
    
    for col, name in targets.items():
        y_train_col = y_train_all[:, col]
        y_test_col = y_test_all[:, col]
        pd.DataFrame(y_train_col, columns=[name]).to_csv(os.path.join(seed_folder, f"y_train_{name}.csv"), index=False)
        pd.DataFrame(y_test_col, columns=[name]).to_csv(os.path.join(seed_folder, f"y_test_{name}.csv"), index=False)
    
    best_params_dict = load_best_params()
    for col, name in targets.items():
        y_train_col = y_train_all[:, col]
        y_test_col = y_test_all[:, col]
        execute_training_by_column(
            X_train, X_test, y_train_col, y_test_col,
            feature_names, col, f"{name}.csv", seed, seed_folder,
            best_params_dict, idx_last_rate, idx_last_rate_std
        )

def loop_exec(n):
    main_folder = create_main_output_folder()
    
    prepare_py_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../prepare.py")
    if os.path.exists(prepare_py_path):
        with open(prepare_py_path, 'r') as f:
            lines = f.readlines()
        with open(os.path.join(main_folder, "params.txt"), 'w') as f_out:
            f_out.writelines(lines[7:50])
    
    seeds = [665, 616, 617, 232, 84, 230, 383, 887, 617, 531, 496]
    for seed in seeds:
        print(f"Seed: {seed}")
        seed_folder = os.path.join(main_folder, str(seed))
        os.makedirs(seed_folder, exist_ok=True)
        run_once(seed_folder, seed)
    
    print("\nFinal Results (MAPE):")
    for target in ["mean_1", "std_1", "mean_2", "std_2"]:
        mape_list = []
        for seed in seeds:
            seed_folder = os.path.join(main_folder, str(seed))
            target_file = os.path.join(seed_folder, f"{target}.csv")
            if not os.path.exists(target_file):
                continue
            df = pd.read_csv(target_file, sep=';')
            mape_list.append(df['MAPE'].iloc[0])
        
        if mape_list:
            mean_mape = np.mean(mape_list)
            std_mape = np.std(mape_list, ddof=1)
            print(f"{target}: {mean_mape:.6f} ± {std_mape:.6f}")

if __name__ == "__main__":
    loop_exec(1)
