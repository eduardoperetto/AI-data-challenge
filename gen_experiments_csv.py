import os
import pandas as pd
from datetime import datetime

INPUT_FOLDER = './output/saving_csv/rulefits'

def find_and_compile_results(root_folder: str):
    """
    Recursively finds specific CSV files, extracts relevant data,
    and compiles it into a single DataFrame.

    Args:
        root_folder (str): The path to the folder to start the search from.

    Returns:
        pd.DataFrame: A DataFrame containing the compiled results,
                      or None if no files were found.
    """
    target_files = {'mean_1.csv', 'mean_2.csv', 'std_1.csv', 'std_2.csv'}
    all_data_frames = []

    print(f"Starting search in folder: '{root_folder}'...")

    if not os.path.isdir(root_folder):
        print(f"Error: The specified input folder '{root_folder}' does not exist.")
        return None

    # Walk through the directory tree
    for dirpath, dirnames, filenames in os.walk(root_folder):
        # Exclude any directories named 'medians' from the search.
        # We modify 'dirnames' in-place, and os.walk will not recurse into them.
        if 'medians' in dirnames:
            print(f"Ignoring directory: {os.path.join(dirpath, 'medians')}")
            dirnames.remove('medians')

        for filename in filenames:
            if filename in target_files:
                file_path = os.path.join(dirpath, filename)
                print(f"Found and processing file: {file_path}")
                try:
                    # Read the semicolon-separated CSV
                    df = pd.read_csv(file_path, sep=';')

                    # Define the columns we want to keep
                    required_columns = [
                        'Model',
                        'MeanOrStd?',
                        'ConsiderRTT_TR?',
                        'MAPE',
                        'Seed',
                        'TrainTime'
                    ]

                    # Check if all required columns exist in the file
                    if all(col in df.columns for col in required_columns):
                        # Extract the required columns
                        filtered_df = df[required_columns]
                        all_data_frames.append(filtered_df)
                    else:
                        # Find which columns are missing for a more informative error message
                        missing_cols = [col for col in required_columns if col not in df.columns]
                        print(f"Warning: Skipping file '{file_path}' because it's missing columns: {missing_cols}")

                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    if not all_data_frames:
        print("No data found. No output file will be created.")
        return None

    # Concatenate all dataframes into a single one
    print("Compiling all found data...")
    compiled_df = pd.concat(all_data_frames, ignore_index=True)
    return compiled_df

def main():
    """
    Main function to execute the script logic.
    """

    # Find and process the files
    final_results = find_and_compile_results(INPUT_FOLDER)

    if final_results is not None and not final_results.empty:
        # Rename columns as per the requirements
        rename_mapping = {
            'Model': 'Modelo',
            'MeanOrStd?': 'Mean or Std?',
            'ConsiderRTT_TR?': 'Consider RTT/TR?',
            'MAPE': 'MAPE',
            'Seed': 'Seed',
            'TrainTime': 'TrainTime'
        }
        final_results.rename(columns=rename_mapping, inplace=True)

        # Sort the DataFrame by the 'Mean or Std?' column
        final_results.sort_values(by='Mean or Std?', inplace=True)

        # Generate the output filename with a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"experiments_{timestamp}.csv"

        # Save the final dataframe to a comma-separated CSV
        try:
            final_results.to_csv(output_filename, sep=',', index=False, encoding='utf-8')
            print(f"\nSuccessfully compiled and sorted results into '{output_filename}'")
            print(f"Total rows compiled: {len(final_results)}")
        except Exception as e:
            print(f"Error saving the final CSV file: {e}")

if __name__ == "__main__":
    main()