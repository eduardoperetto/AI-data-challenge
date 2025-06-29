import os
import pandas as pd
import argparse

INPUT_FOLDER = "./output/saving_csv/base/RuleFit/20250628_203923"

TARGET_FILES = ['mean_1.csv', 'mean_2.csv', 'std_1.csv', 'std_2.csv']

def process_rule_importance(root_folder):
    """
    Recursively finds target CSVs in a directory, extracts and transforms
    the 'RuleImportance' column, and saves it to a text file.

    Args:
        root_folder (str): The path to the top-level directory to start searching from.
    """
    print(f"🚀 Starting script to process rule importance in: {root_folder}")

    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename in TARGET_FILES:
                csv_path = os.path.join(dirpath, filename)
                print(f"🔍 Found target file: {csv_path}")

                try:
                    df = pd.read_csv(csv_path, sep=";")

                    if 'RuleImportance' in df.columns and not df['RuleImportance'].empty:
                        # Get the data from the first row of the 'RuleImportance' column
                        rule_data = str(df['RuleImportance'].iloc[0])
                        
                        # Replace all instances of '<>' with a newline character
                        formatted_rules = rule_data.replace('<>', '\n')
                        
                        # Get the base name of the CSV (e.g., 'mean_1' from 'mean_1.csv')
                        base_name = os.path.splitext(filename)[0]
                        # Construct the new output filename
                        output_filename = f"{base_name}_rule_importance.txt"
                        # Define the output path for the new .txt file
                        output_txt_path = os.path.join(dirpath, output_filename)
                        
                        # Write the formatted string to 'rule_importance.txt'
                        with open(output_txt_path, 'w', encoding='utf-8') as f:
                            f.write(formatted_rules)
                        
                        print(f"✅ Successfully created: {output_txt_path}")
                    else:
                        print(f"  - ⚠️  Warning: 'RuleImportance' column not found or is empty in {csv_path}")

                except (pd.errors.ParserError, FileNotFoundError) as e:
                    print(f"  - ❌ Error: Could not read or parse {csv_path}. Details: {e}")
                except Exception as e:
                    print(f"  - ❌ An unexpected error occurred with {csv_path}. Details: {e}")

    print("\n🎉 Script finished.")


def main():
    """
    Sets up command-line argument parsing and runs the main processing function.
    """

    process_rule_importance(INPUT_FOLDER)

if __name__ == '__main__':
    main()
