import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from rulefit import RuleFit

# --- NEW CONSTANTS ---
# Define the base path where your 'saving_csv' folder is located.
# This script will recursively search within this path.
TRAINING_PATH = "output/saving_csv/rulefits/base/RuleFit_RF/20250629_183627" 

# Define the specific target value (e.g., "mean_1", "std_1", "mean_2", "std_2")
TARGET_VALUE = "mean_1" 

def load_rulefit_model(model_path):
    """
    Loads a RuleFit model from a .pkl file.

    Args:
        model_path (str): The full path to the .pkl model file.

    Returns:
        object: The loaded RuleFit model object.

    Raises:
        FileNotFoundError: If the specified model file does not exist.
        ValueError: If the .pkl file does not contain a 'model' key.
    """
    if os.path.exists(model_path):
        data = joblib.load(model_path)
        if 'model' in data:
            return data['model']
        else:
            raise ValueError(f"'{model_path}' does not contain a 'model' key. "
                             "Ensure the model was saved with {'model': model_object}.")
    else:
        raise FileNotFoundError(f"Model file not found at: {model_path}")

def get_feature_names_from_csv(csv_path):
    """
    Loads feature names from a CSV file (e.g., X_train.csv).
    Assumes the CSV contains only feature columns.

    Args:
        csv_path (str): Path to the CSV file containing feature data.

    Returns:
        list: A list of feature names (column headers).
    """
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df.columns.tolist()
    else:
        raise FileNotFoundError(f"Feature names CSV file not found at: {csv_path}")

def find_target_run_paths(training_path, target_value):
    """
    Recursively searches for training run directories containing the target CSV,
    and identifies the corresponding model and X_train CSV paths.
    It ignores any directories named 'medians'.

    Args:
        training_path (str): The base directory to start the search.
        target_value (str): The name of the target (e.g., "mean_1").

    Returns:
        list: A list of dictionaries, where each dictionary contains:
              'model_path', 'feature_names_path', 'target_csv_path' for a valid run.
    """
    found_runs = []
    target_csv_name = f"{target_value}.csv"

    for root, dirs, files in os.walk(training_path):
        # Remove 'medians' from the list of directories to visit
        # This prevents os.walk from descending into them
        if 'medians' in dirs:
            dirs.remove('medians')

        if target_csv_name in files:
            # Found a target CSV, now look for the corresponding .pkl model and X_train.csv
            target_csv_path = os.path.join(root, target_csv_name)
            feature_names_path = os.path.join(root, "X_train.csv")

            # Look for the .pkl model in the same directory.
            # The model name pattern is RuleFit_RF_{target_value}_{seed}_{mape_str}.pkl
            model_file = None
            for f in files:
                if f.endswith(".pkl") and target_value in f:
                    model_file = f
                    break
            
            # Ensure both the model .pkl and X_train.csv exist in this directory
            if model_file and os.path.exists(feature_names_path):
                model_path = os.path.join(root, model_file)
                found_runs.append({
                    'model_path': model_path,
                    'feature_names_path': feature_names_path,
                    'target_csv_path': target_csv_path,
                    'folder_path': root # Add the folder path for the output CSV
                })
            else:
                print(f"Skipping directory {root}: Could not find corresponding .pkl model or X_train.csv for {target_value}.")
    return found_runs


if __name__ == "__main__":
    # This dictionary will store feature importances from all runs.
    # Format: {feature_name: [importance_run1, importance_run2, ...]}
    all_feature_importances = {} 
    
    # This list will store detailed importance data for the output CSV
    detailed_importances_for_csv = []

    # Find all relevant training runs based on TRAINING_PATH and TARGET_VALUE
    training_runs = find_target_run_paths(TRAINING_PATH, TARGET_VALUE)

    if not training_runs:
        print(f"No training runs found for target '{TARGET_VALUE}' in '{TRAINING_PATH}' (excluding 'medians' folders).")
    else:
        print(f"Found {len(training_runs)} training runs for target '{TARGET_VALUE}'.")
        
        # Initialize the all_feature_importances dictionary with feature names
        # from the first found X_train.csv. We assume feature names are consistent.
        if training_runs:
            initial_feature_names = get_feature_names_from_csv(training_runs[0]['feature_names_path'])
            for feat_name in initial_feature_names:
                all_feature_importances[feat_name] = []

        # Iterate through each found training run to extract feature importances
        for run_info in training_runs:
            model_path = run_info['model_path']
            feature_names_path = run_info['feature_names_path']
            folder_path = run_info['folder_path'] # Get the folder path

            current_run_details = {'Folder_Path': folder_path}
            
            try:
                rulefit_model = load_rulefit_model(model_path)
                # Get feature names for the current run's X_train.csv
                current_feature_names = get_feature_names_from_csv(feature_names_path) 
                
                # Check if the loaded model has a RandomForestRegressor as its tree_generator
                # and if it has the feature_importances_ attribute.
                if hasattr(rulefit_model, 'tree_generator') and \
                   isinstance(rulefit_model.tree_generator, RandomForestRegressor) and \
                   hasattr(rulefit_model.tree_generator, 'feature_importances_'):
                    
                    rf_importances = rulefit_model.tree_generator.feature_importances_
                    
                    # Basic check for consistency between importances and feature names
                    if len(rf_importances) != len(current_feature_names):
                        print(f"Warning: Mismatch in {model_path}: number of importances ({len(rf_importances)}) "
                              f"does not match number of feature names ({len(current_feature_names)}). Skipping this run.")
                        continue # Skip this run if there's a mismatch

                    # Append the importances to the list for each feature
                    for i, feat_name in enumerate(current_feature_names):
                        if feat_name in all_feature_importances:
                            all_feature_importances[feat_name].append(rf_importances[i])
                        else:
                            # If a feature is found that wasn't in the initial list, add it.
                            all_feature_importances[feat_name] = [rf_importances[i]]
                        
                        # Add to current run details for CSV output
                        current_run_details[feat_name] = rf_importances[i]
                    
                    detailed_importances_for_csv.append(current_run_details)

                else:
                    print(f"Skipping model {model_path}: Not a RandomForestRegressor tree_generator or missing feature_importances_.")

            except (FileNotFoundError, ValueError, Exception) as e:
                print(f"Error processing run {model_path}: {e}")
                continue # Continue to the next run even if one fails

        # --- Output detailed importances to CSV ---
        if detailed_importances_for_csv:
            df_detailed_importances = pd.DataFrame(detailed_importances_for_csv)
            # Fill NaN values (for features not present in all runs, though unlikely with consistent X_train.csv)
            df_detailed_importances = df_detailed_importances.fillna(0) 
            output_detailed_csv_name = f"feature_importances_details_{TARGET_VALUE}.csv"
            df_detailed_importances.to_csv(output_detailed_csv_name, index=False)
            print(f"\nDetailed feature importances saved to: {output_detailed_csv_name}")
        else:
            print("\nNo detailed feature importances could be collected for CSV output.")


        # Aggregate results: Calculate mean and standard deviation for each feature
        aggregated_importances = []
        for feature, importances_list in all_feature_importances.items():
            if importances_list: # Only consider features that had importances recorded
                mean_imp = np.mean(importances_list)
                # Use ddof=1 for sample standard deviation
                std_imp = np.std(importances_list, ddof=1) if len(importances_list) > 1 else 0 
                aggregated_importances.append({'feature': feature, 'mean_importance': mean_imp, 'std_importance': std_imp})

        if not aggregated_importances:
            print(f"No aggregated importances found for target '{TARGET_VALUE}'. This might happen if no valid models were processed.")
        else:
            # Create a DataFrame from the aggregated results
            df_aggregated = pd.DataFrame(aggregated_importances)
            # Sort by mean importance in descending order
            df_aggregated = df_aggregated.sort_values(by='mean_importance', ascending=False)

            # Select the top N features for plotting
            top_n = 5 # Changed to top 5
            df_plot = df_aggregated.head(top_n)

            # Prepare data for matplotlib plotting
            plot_feature_names = df_plot['feature'].tolist()
            plot_importance_values = df_plot['mean_importance'].tolist()
            plot_stds = df_plot['std_importance'].tolist() 

            # No need to reverse for vertical bar chart (most important on left)
            # plot_feature_names.reverse()
            # plot_importance_values.reverse()
            # plot_stds.reverse()

            # --- Plotting the Feature Importances ---
            fig, ax = plt.subplots(figsize=(12, 8)) # Adjust figure size as needed

            # Create vertical bar chart with error bars
            x_pos = np.arange(len(plot_feature_names))
            ax.bar(x_pos, plot_importance_values, yerr=plot_stds, align='center', color='lightcoral', capsize=5)
            
            # Set x-axis ticks and labels
            ax.set_xticks(x_pos)
            ax.set_xticklabels(plot_feature_names, rotation=45, ha="right", fontsize=10) # Rotate labels for readability
            
            # Set y-axis label and title
            ax.set_ylabel("Average Importance Value", fontsize=14)
            ax.set_title(f"Average Top {top_n} Feature Importances from RuleFit's RandomForestGenerator for Target: {TARGET_VALUE}", fontsize=16)
            
            # Adjust y-axis tick label font size
            ax.tick_params(axis='y', labelsize=12)
            
            # No need to invert y-axis for vertical bars
            # ax.invert_yaxis() 

            plt.tight_layout() # Adjust layout to prevent labels from overlapping
            
            # Save the plot as a PDF file
            output_pdf_name = f"{TARGET_VALUE}_RF_avg_feature_importance.pdf"
            plt.savefig(output_pdf_name)
            print(f"Average feature importance graph saved as {output_pdf_name}")
            # plt.show() # Uncomment to display the plot immediately
