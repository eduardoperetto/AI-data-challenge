import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Read in the CSV
df = pd.read_csv('prepared_data.csv')

# Filter relevant columns
cols_of_interest = [
    'start_ts',
    'client_id',
    'server_id',
    'tr_jumps_std',
    'rates_stdev',
    'mean_1',
    'stdev_1',
    'mean_2',
    'stdev_2'
]
df = df[cols_of_interest]

# Convert Unix timestamp to human-readable datetime
df['start_ts'] = pd.to_datetime(df['start_ts'], unit='s')

# Sort by time to ensure proper ordering
df = df.sort_values(by='start_ts')

# Normalize the feature columns
scaler = MinMaxScaler()
feature_cols = ['tr_jumps_std', 'rates_stdev', 'mean_1', 'stdev_1', 'mean_2', 'stdev_2']
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# Group by client_id and server_id
grouped = df.groupby(['client_id', 'server_id'])

# Iterate through each client-server combination
for (client, server), group_data in grouped:
    # Melt the data for easier plotting
    plot_data = group_data.melt(
        id_vars=['start_ts', 'client_id', 'server_id'],
        value_vars=feature_cols,
        var_name='metric',
        value_name='value'
    )

    # Initialize the figure
    plt.figure(figsize=(12, 6))
    
    # Create a line plot
    sns.lineplot(
        data=plot_data,
        x='start_ts',
        y='value',
        hue='metric'
    )
    
    plt.title(f'Client: {client}, Server: {server}')
    plt.xlabel('Timestamp')
    plt.ylabel('Normalized Value')

    # Customize x-axis ticks for better readability
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # Adjust the number of ticks

    # Ensure a tight layout
    plt.tight_layout()

    # Save the figure if desired
    plt.savefig(f'figures/client_{client}_server_{server}.png', dpi=300)

    # Show the plot
    # plt.show()
