import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
import os

def load_data_from_csv(input_csv):
    """
    Loads data from a specified CSV file into a pandas DataFrame.
    """
    data_df = pd.read_csv(input_csv)
    return data_df

def calculate_predictions(data_df):
    """
    Calculates prediction columns by first reconstructing the absolute values from an
    initial value and subsequent differences, then calculating the mean.
    """
    rate_means_cols = [col for col in data_df.columns if 'dash' in col and 'rate_mean' in col and 'last' not in col]
    rate_stdevs_cols = [col for col in data_df.columns if 'dash' in col and 'rate_stdev' in col and 'last' not in col]

    def get_dash_number(col_name):
        try:
            return int(col_name.split('_')[0].replace('dash', ''))
        except (ValueError, IndexError):
            return float('inf')

    rate_means_cols.sort(key=get_dash_number)
    rate_stdevs_cols.sort(key=get_dash_number)

    if rate_means_cols:
        reconstructed_means = data_df[rate_means_cols].cumsum(axis=1)
        pred_mean = reconstructed_means.mean(axis=1)
    else:
        pred_mean = pd.Series(0, index=data_df.index)

    if rate_stdevs_cols:
        reconstructed_stdevs = data_df[rate_stdevs_cols].cumsum(axis=1)
        pred_stdev = reconstructed_stdevs.mean(axis=1)
    else:
        pred_stdev = pd.Series(0, index=data_df.index)

    last_rate_values = data_df["dash_last_rate"]
    last_rate_std_values = data_df["dash_last_rate_std"]

    data_df['pred_mean_1'] = pred_mean - last_rate_values
    data_df['pred_mean_2'] = pred_mean - last_rate_values

    data_df['pred_stdev_1'] = pred_stdev - last_rate_std_values
    data_df['pred_stdev_2'] = pred_stdev - last_rate_std_values

    print("Mean of reconstructed values (pred_mean):")
    print(pred_mean.head())
    print("\nLast rate values (for subtraction):")
    print(last_rate_values.head())

    return data_df

def evaluate_predictions(data_df):
    """
    Evaluates the predictions by comparing the absolute predicted values against the
    absolute true values, and calculates MAPE for each prediction type.
    """
    print("\n--- Evaluating on Absolute Values ---")
    
    # Define the pairs of true and predicted columns to evaluate
    prediction_pairs = {
        'mean_1': ('mean_1', 'pred_mean_1', 'dash_last_rate'),
        'stdev_1': ('stdev_1', 'pred_stdev_1', 'dash_last_rate_std'),
        'mean_2': ('mean_2', 'pred_mean_2', 'dash_last_rate'),
        'stdev_2': ('stdev_2', 'pred_stdev_2', 'dash_last_rate_std')
    }

    mape_results = {}

    for name, cols in prediction_pairs.items():
        true_col, pred_col, last_rate_col = cols

        y_true_actual = data_df[true_col] + data_df[last_rate_col]
        
        y_pred_actual = data_df[pred_col] + data_df[last_rate_col]

        non_zero_mask = y_true_actual != 0
        
        if non_zero_mask.sum() > 0:
            mape = mean_absolute_percentage_error(
                y_true_actual[non_zero_mask],
                y_pred_actual[non_zero_mask]
            )
            print(f"MAPE for {name} (comparing absolute values): {mape * 100:.2f}%")
            mape_results[name] = mape
        else:
            print(f"MAPE for {name}: All true values are zero, cannot calculate MAPE.")
            mape_results[name] = float('nan')

    return mape_results

def main():
    """
    Main function to run the data loading, prediction, and evaluation pipeline.
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_csv = os.path.join(script_dir, "../prepared_data.csv")

        if not os.path.exists(input_csv):
            input_csv = "prepared_data.csv"
            print(f"File not found at '../prepared_data.csv', trying current directory: {input_csv}")

        # Load data from the CSV file
        data_df = load_data_from_csv(input_csv)

        # Calculate the predictions
        data_df = calculate_predictions(data_df)

        # Evaluate the predictions with MAPE for each component
        evaluate_predictions(data_df)

    except FileNotFoundError:
        print(f"Error: The file was not found.")
        print("Please ensure 'prepared_data.csv' exists in the script's directory or in a parent directory.")
    except KeyError as e:
        print(f"Error: A required column is missing from the CSV file: {e}")
        print("Please ensure all required 'dash', 'last_rate', and 'mean' columns exist in the input file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
