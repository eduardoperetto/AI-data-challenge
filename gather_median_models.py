import os
import shutil
import pandas as pd
import statistics

ROOT_FOLDER = './output/saving_csv/base/RuleFit/20250628_203923'

# The names of your target values, which correspond to the .csv filenames.
TARGETS = ['mean_1', 'mean_2', 'std_1', 'std_2']
# The name of the output folder that will be created.
OUTPUT_FOLDER = ROOT_FOLDER + "/medians"

def find_median_mape_folder(root_folder, target):
    """
    Finds the seed folder with the median MAPE for a given target.

    Args:
        root_folder (str): The path to the directory containing seed folders.
        target (str): The target value (e.g., 'mean_1').

    Returns:
        str: The path to the seed folder with the median MAPE, or None if not found.
    """
    mape_data = []
    
    # --- 1. Find all seed directories and extract MAPE values ---
    print(f"🔍 Processing target: {target}")
    if not os.path.isdir(root_folder):
        print(f"❌ Error: Root folder '{root_folder}' not found.")
        return None

    for item in os.listdir(root_folder):
        seed_path = os.path.join(root_folder, item)
        # Ensure the item is a directory and its name is a number (a seed)
        if os.path.isdir(seed_path) and item.isdigit():
            csv_path = os.path.join(seed_path, f"{target}.csv")
            if os.path.exists(csv_path):
                try:
                    # Read the CSV and get the MAPE value.
                    # This assumes a column named 'MAPE' exists and the value is in the first row.
                    df = pd.read_csv(csv_path, sep=";")
                    if 'MAPE' in df.columns and not df['MAPE'].empty:
                        mape_value = float(df['MAPE'].iloc[0])
                        mape_data.append({'seed': item, 'path': seed_path, 'mape': mape_value})
                        print(f"  - Found seed {item} with MAPE: {mape_value}")
                    else:
                        print(f"  - ⚠️  Warning: 'MAPE' column not found or empty in {csv_path}")
                except (pd.errors.ParserError, ValueError, FileNotFoundError) as e:
                    print(f"  - ⚠️  Warning: Could not read or parse {csv_path}. Error: {e}")
            else:
                print(f"  - ⚠️  Warning: CSV file not found at {csv_path}")

    if not mape_data:
        print(f"❌ Error: No valid MAPE data found for target '{target}'.")
        return None

    # --- 2. Calculate the median MAPE ---
    mape_values = [data['mape'] for data in mape_data]
    median_mape = statistics.median(mape_values)
    print(f"\n📊 Calculated median MAPE for '{target}': {median_mape}")

    # --- 3. Find the folder with the MAPE closest to the median ---
    # This handles cases with an even number of seeds correctly.
    closest_entry = min(mape_data, key=lambda x: abs(x['mape'] - median_mape))
    
    print(f"✅ Selected median folder for '{target}': Seed {closest_entry['seed']} (MAPE: {closest_entry['mape']})")
    return closest_entry['path']

def main():
    """
    Main function to orchestrate the folder copying process.
    """
    print("🚀 Starting script to find folders with median MAPE.")
    
    if not os.path.exists(ROOT_FOLDER):
        print(f"❌ The specified ROOT_FOLDER '{ROOT_FOLDER}' does not exist.")
        print("Please create a dummy './seeds' directory with subdirectories like '1', '2', etc., and add some sample CSV files to test the script.")
        return

    for target in TARGETS:
        median_folder_path = find_median_mape_folder(ROOT_FOLDER, target)
        
        if median_folder_path:
            # --- 4. Create destination directory and copy files ---
            seed_name = os.path.basename(median_folder_path)
            destination_path = os.path.join(OUTPUT_FOLDER, target, seed_name)
            
            # Use shutil.copytree, but handle existing directories gracefully
            if os.path.exists(destination_path):
                shutil.rmtree(destination_path)
                print(f"🧹 Removed existing directory: {destination_path}")
                
            try:
                shutil.copytree(median_folder_path, destination_path)
                print(f"📦 Successfully copied '{median_folder_path}' to '{destination_path}'")
            except OSError as e:
                 print(f"❌ Error copying files: {e}")
        
        print("-" * 40)

    print("🎉 Script finished.")

if __name__ == '__main__':
    main()
