import pandas as pd
import numpy as np
import time
import os

def get_dash_number(col_name):
    try:
        return int(col_name.split('_')[0].replace('dash', ''))
    except (ValueError, IndexError):
        return float('inf')

def main():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_csv = os.path.join(script_dir, "../prepared_data.csv")

        if not os.path.exists(input_csv):
            input_csv = "prepared_data.csv"
        
        data_df = pd.read_csv(input_csv)
        
        rate_means_cols = [col for col in data_df.columns if 'dash' in col and 'rate_mean' in col and 'last' not in col]
        rate_stdevs_cols = [col for col in data_df.columns if 'dash' in col and 'rate_stdev' in col and 'last' not in col]
        
        rate_means_cols.sort(key=get_dash_number)
        rate_stdevs_cols.sort(key=get_dash_number)
        
        last_rate_mean_col = "dash_last_rate"
        last_rate_std_col = "dash_last_rate_std"
        
        times = {
            'mean_1': [],
            'mean_2': [],
            'stdev_1': [],
            'stdev_2': []
        }
        
        for index, row in data_df.iterrows():
            
            start_time_mean_1 = time.perf_counter()
            history_mean = row[rate_means_cols]
            predicted_average_mean = history_mean.cumsum().mean()
            _ = predicted_average_mean - row[last_rate_mean_col] + row[last_rate_mean_col]
            end_time_mean_1 = time.perf_counter()
            times['mean_1'].append((end_time_mean_1 - start_time_mean_1) * 1000)
            
            start_time_mean_2 = time.perf_counter()
            history_mean = row[rate_means_cols]
            predicted_average_mean = history_mean.cumsum().mean()
            _ = predicted_average_mean - row[last_rate_mean_col] + row[last_rate_mean_col]
            end_time_mean_2 = time.perf_counter()
            times['mean_2'].append((end_time_mean_2 - start_time_mean_2) * 1000)

            start_time_stdev_1 = time.perf_counter()
            history_stdev = row[rate_stdevs_cols]
            predicted_average_stdev = history_stdev.cumsum().mean()
            _ = predicted_average_stdev - row[last_rate_std_col] + row[last_rate_std_col]
            end_time_stdev_1 = time.perf_counter()
            times['stdev_1'].append((end_time_stdev_1 - start_time_stdev_1) * 1000)

            start_time_stdev_2 = time.perf_counter()
            history_stdev = row[rate_stdevs_cols]
            predicted_average_stdev = history_stdev.cumsum().mean()
            _ = predicted_average_stdev - row[last_rate_std_col] + row[last_rate_std_col]
            end_time_stdev_2 = time.perf_counter()
            times['stdev_2'].append((end_time_stdev_2 - start_time_stdev_2) * 1000)
            
        for target_name, time_list in times.items():
            avg_time = np.mean(time_list)
            std_dev_time = np.std(time_list)
            print(f"Inference time for {target_name}: {avg_time:.4f} ms +- {std_dev_time:.4f}")

    except FileNotFoundError:
        print(f"Error: The file was not found.")
        print("Please ensure 'prepared_data.csv' exists in the script's directory or in a parent directory.")
    except KeyError as e:
        print(f"Error: A required column is missing from the CSV file: {e}")
        print("Please ensure all required 'dash', 'rate_mean', 'rate_stdev', 'dash_last_rate', and 'dash_last_rate_std' columns exist in the input file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()