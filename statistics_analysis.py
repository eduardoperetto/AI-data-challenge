import pandas as pd
import numpy as np
from scipy.stats import kruskal
import scikit_posthocs as sp
from datetime import datetime
import sys

# Define the significance level
SIGNIFICANCE_LEVEL = 0.05

# File path for the input CSV
INPUT_CSV = 'experiments_20250629_211541.csv'

def main():
    # Read the CSV file with comma as separator
    try:
        df = pd.read_csv(INPUT_CSV, sep=',')
    except FileNotFoundError:
        print(f"Error: The file '{INPUT_CSV}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        sys.exit(1)

    # Check if required columns exist
    required_columns = ['Modelo', 'Mean or Std?', 'Consider RTT/TR?', 'MAPE']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: The CSV file must contain the following columns: {required_columns}")
        sys.exit(1)

    # Clean the 'MAPE' column: remove '%', convert to float
    try:
        df['MAPE'] = df['MAPE'].astype(str).str.replace('%', '').astype(float)
    except Exception as e:
        print(f"Error processing the 'MAPE' column: {e}")
        sys.exit(1)

    # Standardize 'Consider RTT/TR?' column to uppercase
    df['Consider RTT/TR?'] = df['Consider RTT/TR?'].astype(str).str.upper()

    # --- MODIFICATION START ---
    # Standardize the 'Mean or Std?' column to be case-insensitive
    # and to handle variations like 'StdDev', 'Standard Deviation', etc.
    def standardize_metric(metric_name):
        # Convert to string and then to lower case for consistent matching
        metric_name_lower = str(metric_name).lower()
        if 'mean' in metric_name_lower:
            return 'Mean'
        elif 'std' in metric_name_lower:
            return 'Std'
        # Return the original value if it's neither
        return metric_name

    df['Mean or Std?'] = df['Mean or Std?'].apply(standardize_metric)
    # --- MODIFICATION END ---

    # Assign 'Run' numbers within each 'Modelo' and 'Consider RTT/TR?' group
    # This assumes that for each 'Run', there is one 'Mean' and one 'Std' entry
    df['Run'] = df.groupby(['Modelo', 'Consider RTT/TR?']).cumcount() // 2 + 1

    # Pivot the data to have 'Mean' and 'Std' MAPE side by side
    pivot_df = df.pivot_table(index=['Modelo', 'Consider RTT/TR?', 'Run'],
                              columns='Mean or Std?',
                              values='MAPE').reset_index()

    # Check if both 'Mean' and 'Std' exist for all groups after pivoting
    if 'Mean' not in pivot_df.columns or 'Std' not in pivot_df.columns:
        print("Error: The data must contain both 'Mean' and 'Std' entries for each run.")
        print("Please check the 'Mean or Std?' column in your CSV file for consistent naming.")
        sys.exit(1)

    # Calculate the average MAPE for each run by averaging 'Mean' and 'Std'
    pivot_df['Average_MAPE'] = pivot_df[['Mean', 'Std']].mean(axis=1)

    # Create a new column for group labels
    pivot_df['Group'] = pivot_df.apply(
        lambda row: f"{row['Modelo']}_with_RTT_TR" if row['Consider RTT/TR?'] == 'TRUE'
        else f"{row['Modelo']}_without_RTT_TR", axis=1)

    # Collect average MAPEs per group
    groups = pivot_df['Group'].unique()

    # Define expected groups
    expected_groups = ['RF_with_RTT_TR', 'RF_without_RTT_TR',
                       'XGBoost_with_RTT_TR', 'XGBoost_without_RTT_TR']

    # Prepare data for Kruskal-Wallis test
    group_data = {}
    for group in expected_groups:
        mape_values = pivot_df[pivot_df['Group'] == group]['Average_MAPE'].tolist()
        if not mape_values:
            print(f"Warning: No data found for group '{group}'. This group will be excluded from the analysis.")
        group_data[group] = mape_values

    # Remove groups with no data
    valid_groups = [group for group in expected_groups if group_data[group]]

    # Check if there are at least two groups with data
    if len(valid_groups) < 2:
        print("Error: Insufficient groups with data for Kruskal-Wallis test. At least two groups are required.")
        sys.exit(1)

    # Prepare list of groups for Kruskal-Wallis
    data_for_kw = [group_data[group] for group in valid_groups]

    # Perform Kruskal-Wallis H Test
    try:
        stat, p = kruskal(*data_for_kw)
    except ValueError as ve:
        print(f"Error performing Kruskal-Wallis test: {ve}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during Kruskal-Wallis test: {e}")
        sys.exit(1)

    # Initialize output string
    output = ""

    # Introduction
    output += "Statistical Analysis of MAPE Across Different Models and Feature Considerations\n"
    output += "="*80 + "\n\n"

    output += "**Groups Compared:**\n"
    for group in valid_groups:
        modelo, feature = group.split('_with_RTT_TR') if '_with_RTT_TR' in group else group.split('_without_RTT_TR')
        feature_desc = "With RTT-TR Features" if 'with_RTT_TR' in group else "Without RTT-TR Features"
        output += f"- {modelo} ({feature_desc})\n"
    output += "\n"

    # Summary Statistics
    output += "**Summary Statistics:**\n"
    summary = pivot_df.groupby('Group')['Average_MAPE'].agg(['count', 'median']).reset_index()
    summary = summary.rename(columns={'count': 'Number of Runs', 'median': 'Median MAPE (%)'})
    output += summary.to_string(index=False)
    output += "\n\n"

    # Kruskal-Wallis Test Results
    output += "### Kruskal-Wallis H Test\n"
    output += f"- **Test Statistic:** {stat:.4f}\n"
    output += f"- **p-value:** {p:.4f}\n"

    if p < SIGNIFICANCE_LEVEL:
        output += f"- **Result:** p-value ({p:.4f}) < {SIGNIFICANCE_LEVEL}, indicating significant differences among the groups.\n\n"
    else:
        output += f"- **Result:** p-value ({p:.4f}) >= {SIGNIFICANCE_LEVEL}, indicating no significant differences among the groups.\n\n"

    # Dunn's Post Hoc Test
    if p < SIGNIFICANCE_LEVEL:
        output += "### Dunn's Post Hoc Test (Bonferroni Corrected)\n"
        # Perform Dunn's test
        dunn_data = pivot_df[['Group', 'Average_MAPE']].copy()
        try:
            dunn_results = sp.posthoc_dunn(dunn_data, val_col='Average_MAPE', group_col='Group', p_adjust='bonferroni')
        except Exception as e:
            print(f"Error performing Dunn's post hoc test: {e}")
            sys.exit(1)

        # Format Dunn's results
        dunn_formatted = dunn_results.copy()
        for col in dunn_formatted.columns:
            dunn_formatted[col] = dunn_formatted[col].apply(lambda x: f"{x:.4f}")

        # Create a significance matrix
        significance_matrix = dunn_results.copy()
        significance_matrix = significance_matrix.applymap(lambda x: "*" if x < SIGNIFICANCE_LEVEL else "ns")
        significance_matrix = significance_matrix.replace("NaN", "")

        # Combine p-values and significance markers
        combined_matrix = dunn_formatted.astype(str) + " (" + significance_matrix.astype(str) + ")"
        combined_matrix = combined_matrix.replace("nan (ns)", " " * 10) # Adjusted for better alignment

        output += "#### Pairwise Comparisons:\n"
        output += combined_matrix.to_string() + "\n\n"

        # Interpretation of significant comparisons
        significant_pairs = []
        for i in dunn_results.index:
            for j in dunn_results.columns:
                if i < j:  # to avoid duplicate pairs and self-comparisons
                    p_val = dunn_results.loc[i, j]
                    if p_val < SIGNIFICANCE_LEVEL:
                        significant_pairs.append((i, j, p_val))

        if significant_pairs:
            output += "#### Significant Pairwise Comparisons:\n"
            for pair in significant_pairs:
                output += f"- **{pair[0]}** vs **{pair[1]}**: p-value = {pair[2]:.4f}\n"
        else:
            output += "No significant pairwise comparisons found.\n"
    else:
        output += "No post hoc tests were conducted as the Kruskal-Wallis test was not significant.\n"

    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define output file name
    output_file = f"statistic_analysis_{timestamp}.txt"

    # Save output to the file
    try:
        with open(output_file, 'w') as f:
            f.write(output)
        print(f"Statistical analysis completed. Results saved to {output_file}.")
    except Exception as e:
        print(f"Error writing to the output file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()