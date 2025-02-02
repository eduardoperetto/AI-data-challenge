import pandas as pd
import joblib
import os
import time  # <-- ADDED
from sklearn.preprocessing import StandardScaler
import numpy as np

# Definições dos modelos e flags
MODEL_FILE_MEAN1 = "RF_-4_665_11.56%.pkl"  # -> mean_1
MODEL_FILE_MEAN2 = "RF_-3_665_7.79%.pkl"  # -> mean_2
MODEL_FILE_STD1  = "RF_-2_665_14.84%.pkl"  # -> std_1
MODEL_FILE_STD2  = "RF_-1_665_10.60%.pkl"  # -> std_2

# Flag for Features
FEATURES = "Base"

# Example flags/variables (set as needed)
DISCARD_MEAN = False
NORMALIZE = False
USING_DIFF_FROM_AVG = False
USING_RESULT_AS_DIFF_FROM_LAST = False

def load_data_from_csv(input_csv):
    """
    Load the CSV data, optionally apply transformations,
    and return the full data plus a version X with 
    features only (depending on last_x_index).
    """
    data_df = pd.read_csv(input_csv).dropna()

    # Decide how many columns to take as X
    # (For example, if the last 4 are labels [mean_1, mean_2, stdev_1, stdev_2],
    #  we use everything else for X.)
    last_x_index = -6 if DISCARD_MEAN else -4

    if NORMALIZE:
        scaler = StandardScaler()
        data_df.iloc[:, :last_x_index] = scaler.fit_transform(data_df.iloc[:, :last_x_index])

    # X = all columns except the last 4 (or 6) depending on your use case
    X = data_df.iloc[:, :last_x_index].copy()

    return X, data_df

def get_model_and_value_from_filename(filename):
    """
    - Model: substring before first underscore (e.g. "RF" if "RF_-4_665.pkl")
    - Value: 'mean_1', 'mean_2', 'std_1', or 'std_2' 
             derived from whether the file has -4, -3, -2, or -1
    """
    base = os.path.basename(filename)
    # Model = substring before first underscore
    model = base.split("_")[0]  # e.g. "RF" or "XGB"
    
    # Decide value based on the pattern in the filename
    if "-4" in base:
        value = "mean_1"
    elif "-3" in base:
        value = "mean_2"
    elif "-2" in base:
        value = "std_1"
    elif "-1" in base:
        value = "std_2"
    else:
        value = "unknown"

    return model, value

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(script_dir, "prepared_data.csv")

    # Load full data
    X, data_df = load_data_from_csv(input_csv)

    # We only want the first 11 rows for inference
    X_subset = X.iloc[:11].copy()

    # Our 4 model files
    model_files = [
        MODEL_FILE_MEAN1,
        MODEL_FILE_MEAN2,
        MODEL_FILE_STD1,
        MODEL_FILE_STD2
    ]
    model_paths = [os.path.join(script_dir, mf) for mf in model_files]

    # We'll store 44 records: 4 (values) × 11 (rows)
    results = []

    for mp in model_paths:
        # Load the model
        loaded = joblib.load(mp)
        model_obj = loaded['model']  # your trained model

        # Identify model and value from filename
        mod, val = get_model_and_value_from_filename(mp)

        # For each row in the first 11, do a single predict + measure time
        for i in range(len(X_subset)):  # i in 0..10
            row_data = X_subset.iloc[i : i+1]  # single row as DataFrame
            start_time = time.time()
            _ = model_obj.predict(row_data)
            end_time = time.time()

            inference_time = end_time - start_time

            # Append to results [Model, Features, Value, InferenceTime]
            results.append([mod, FEATURES, val, inference_time])

    # Now we have 44 lines in results
    # Save or append them to inference_time.csv
    inference_csv = os.path.join(script_dir, "inference_time.csv")
    columns = ["Model", "Features", "Value", "InferenceTime"]

    # If the file doesn't exist, we create it with header
    if not os.path.isfile(inference_csv):
        df_inference = pd.DataFrame(results, columns=columns)
        df_inference.to_csv(inference_csv, index=False)
    else:
        # If the file exists, we read it, append, then save again
        df_inference = pd.read_csv(inference_csv)
        new_df = pd.DataFrame(results, columns=columns)
        df_inference = pd.concat([df_inference, new_df], ignore_index=True)
        df_inference.to_csv(inference_csv, index=False)

    # ----------------------------------------------------------------
    # Generate the summary CSV: inference_time_summ.csv
    # Summarize by (Model, Features, Value) => mean ± std
    # ----------------------------------------------------------------
    df_all = pd.read_csv(inference_csv)
    group_cols = ["Model", "Features", "Value"]
    
    grouped = df_all.groupby(group_cols)["InferenceTime"].agg(["mean", "std"]).reset_index()
    
    # Format "x +- y" for mean ± std
    grouped["InferenceTime"] = grouped.apply(
        lambda row: f"{row['mean']:.6f} +- {row['std']:.6f}",
        axis=1
    )
    
    # Drop the now-redundant mean/std columns
    grouped.drop(columns=["mean", "std"], inplace=True)

    # Save summary
    summ_csv = os.path.join(script_dir, "inference_time_summ.csv")
    grouped.to_csv(summ_csv, index=False)

    print("Done! Inference times appended to 'inference_time.csv' (now with header + multiple rows).")
    print("Summary written to 'inference_time_summ.csv'.")

if __name__ == "__main__":
    main()
