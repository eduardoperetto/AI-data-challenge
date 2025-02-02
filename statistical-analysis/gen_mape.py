import pandas as pd

# Define the input and output file paths
input_file = 'input.csv'
output_file = 'mape_output.csv'

# Read the input CSV with ';' as the separator
df = pd.read_csv(input_file, sep=';')

# Create the 'Target' column based on 'ConsiderRTT_TR?'
df['Target'] = df['ConsiderRTT_TR?'].apply(lambda x: 'Enriched' if str(x).strip().lower() == 'true' else 'Base')

# Rename 'MeanOrStd?' to 'FeatureSet' for clarity
df = df.rename(columns={'MeanOrStd?': 'FeatureSet'})

# Group the data by 'Model', 'FeatureSet', and 'Target'
grouped = df.groupby(['Model', 'FeatureSet', 'Target'])

# Aggregate the mean and standard deviation for 'MAPE' and 'TrainTime'
agg_df = grouped.agg(
    MAPE_mean=('MAPE', 'mean'),
    MAPE_std=('MAPE', 'std'),
    TrainTime_mean=('TrainTime', 'mean'),
    TrainTime_std=('TrainTime', 'std')
).reset_index()

# Format 'MAPE' and 'Train time (sec)' as 'mean ± std'
agg_df['MAPE'] = agg_df['MAPE_mean'].round(6).astype(str) + ' ± ' + agg_df['MAPE_std'].round(6).astype(str)
agg_df['Train time (sec)'] = agg_df['TrainTime_mean'].round(4).astype(str) + ' ± ' + agg_df['TrainTime_std'].round(4).astype(str)

# Select the required columns for the output
output_df = agg_df[['Model', 'FeatureSet', 'Target', 'MAPE', 'Train time (sec)']]

# Save the processed data to 'mape_output.csv'
output_df.to_csv(output_file, index=False)

print(f"Output has been successfully saved to '{output_file}'.")
