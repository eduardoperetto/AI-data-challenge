import pandas as pd
import numpy as np
import os

# --- Configuration ---
CSV_FILE_PATH = 'experiments_20250629_211541.csv' 
OUTPUT_LATEX_FILE = 'mape_training_time_table_rulefit.tex'

# Mapping for model names to appear in the table
MODEL_NAME_MAP = {
    'RF': 'RF',
    'XGB': 'XGBoost',
    'RuleFit': 'RuleFit'
}

def generate_latex_table(df: pd.DataFrame) -> str:
    """
    Generates a LaTeX table string from the aggregated experiment data.

    Args:
        df (pd.DataFrame): DataFrame containing the aggregated results.

    Returns:
        str: A string containing the full LaTeX table code.
    """
    
    # --- 1. LaTeX Header ---
    # This part defines the caption, label, and column headers.
    header = r"""
\begin{table*}[t]
    \centering
    \caption{MAPE and Training Time for RF and XGBoost under both feature configurations (Base vs.\ Enriched), aggregated over all runs.
        \small{RF: Random Forest}
    }
    \label{tab:mape-training-time}
    \begin{tabular}{lcccc}
        \toprule
        \textbf{Model} & \textbf{Target} & \textbf{Feature Set} & \textbf{MAPE} & \textbf{Train Time (s)} \\
        \midrule"""

    # --- 2. Table Body ---
    # This part iterates through the data and formats each row.
    table_rows = []
    for _, row in df.iterrows():
        # Escape underscores in the 'Target' column for LaTeX
        target_escaped = row['Mean or Std?'].replace('_', r'\_')
        
        # Format MAPE and Train Time as "mean ± std"
        mape_str = f"{row['MAPE_mean']:.6f} ± {row['MAPE_std']:.6f}"
        train_time_str = f"{row['TrainTime_mean']:.4f} ± {row['TrainTime_std']:.4f}"
        
        # Construct the LaTeX row
        latex_row = f"        {row['Modelo']:<8} & {target_escaped:<8} & {row['Feature Set']:<9} & {mape_str:<22} & {train_time_str} \\\\"
        table_rows.append(latex_row)

    # --- 3. LaTeX Footer ---
    footer = r"""        \midrule
       % TO DO: Baseline
    \end{tabular}
\end{table*}"""

    # --- 4. Combine all parts ---
    return header + "\n" + "\n".join(table_rows) + "\n" + footer


def main():
    """
    Main function to read CSV, process data, and generate the LaTeX file.
    """
    print(f"Reading data from '{CSV_FILE_PATH}'...")
    
    # Check if the file exists
    if not os.path.exists(CSV_FILE_PATH):
        print(f"\n--- ERROR ---")
        print(f"The file '{CSV_FILE_PATH}' was not found.")
        print("Please make sure the file exists and the CSV_FILE_PATH constant is set correctly.")
        return

    # Read the source CSV file
    df = pd.read_csv(CSV_FILE_PATH)

    # --- Data Processing based on your rules ---

    # Rule 1: 'Feature Set' is 'Enriched' if 'Consider RTT/TR?' is True, else 'Base'.
    df['Feature Set'] = np.where(df['Consider RTT/TR?'] == True, 'Enriched', 'Base')
    
    # Map model names to the desired format for the table
    df['Modelo'] = df['Modelo'].replace(MODEL_NAME_MAP)

    # Rule 2 & 3: Group by configuration and calculate mean and std for MAPE and TrainTime.
    aggregation_rules = {
        'MAPE_mean': ('MAPE', 'mean'),
        'MAPE_std': ('MAPE', 'std'),
        'TrainTime_mean': ('TrainTime', 'mean'),
        'TrainTime_std': ('TrainTime', 'std')
    }
    
    grouping_columns = ['Modelo', 'Mean or Std?', 'Feature Set']
    agg_df = df.groupby(grouping_columns).agg(**aggregation_rules).reset_index()

    # Sort the results to match the desired table order
    # Sort by 'Target', then 'Model', then 'Feature Set'
    agg_df = agg_df.sort_values(by=grouping_columns, ascending=[True, True, True])

    print("Data processed. Generating LaTeX table...")
    
    # Generate the final LaTeX code
    latex_output = generate_latex_table(agg_df)

    # Save the output to a .tex file
    with open(OUTPUT_LATEX_FILE, 'w') as f:
        f.write(latex_output)
    
    print("\n--- SUCCESS ---")
    print(f"LaTeX table has been successfully generated and saved to '{OUTPUT_LATEX_FILE}'")
    print("\n--- LaTeX Output ---")
    print(latex_output)


if __name__ == "__main__":
    main()