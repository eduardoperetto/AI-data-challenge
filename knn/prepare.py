import os
import json
import pandas as pd
import numpy as np

def reduce_features_to_60(features):
    while len(features) > 60:
        if len(features) - 60 >= 3:
            aggregated_features = [np.mean(features[i:i+3]) for i in range(0, len(features) - 2, 3)]
            remaining_features = features[len(aggregated_features) * 3:]
            features = aggregated_features + remaining_features
        else:
            remaining_count = len(features) - 60
            features[-(remaining_count + 1):] = [np.mean(features[-(remaining_count + 1):])]
    return features

def extract_features(json_data, filename):
    dash_data = json_data['dash']
    rtt_data = json_data['rtt']
    traceroute_data = json_data['traceroute']
    
    features = []
    feature_names = []

    # Extract DASH features
    for i, dash in enumerate(dash_data):
        if dash['rate']:
            features.extend([
                np.mean(dash['rate']) if dash['rate'] else np.nan,
                np.std(dash['rate']) if dash['rate'] else np.nan,
                np.mean(dash['elapsed']) if dash['elapsed'] else np.nan,
                np.std(dash['elapsed']) if dash['elapsed'] else np.nan,
                np.mean(dash['received']) if dash['received'] else np.nan,
                np.std(dash['received']) if dash['received'] else np.nan
            ])
            feature_names.extend([
                f'dash{i}_mean_rate',
                f'dash{i}_std_rate',
                f'dash{i}_mean_elapsed',
                f'dash{i}_std_elapsed',
                f'dash{i}_mean_received',
                f'dash{i}_std_received'
            ])

    # Extract RTT features
    for i, rtt in enumerate(rtt_data):
        rtt_values = np.array(list(map(float, rtt['val'].keys())))
        if len(rtt_values) > 0:
            rtt_counts = np.array(list(rtt['val'].values()))
            rtt_avg = np.dot(rtt_values, rtt_counts) / rtt_counts.sum()
            features.extend([rtt_avg, np.std(rtt_values)])
            feature_names.extend([f'rtt{i}_avg', f'rtt{i}_std'])
        else:
            features.extend([np.nan, np.nan])
            feature_names.extend([f'rtt{i}_avg', f'rtt{i}_std'])

    # Extract Traceroute features
    for i, trace in enumerate(traceroute_data):
        traceroute_rtts = [hop['rtt'] for hop in trace['val'] if 'rtt' in hop]
        features.extend([
            np.mean(traceroute_rtts) if traceroute_rtts else np.nan,
            np.std(traceroute_rtts) if traceroute_rtts else np.nan
        ])
        feature_names.extend([f'traceroute{i}_mean_rtt', f'traceroute{i}_std_rtt'])

    if len(features) >= 60:
        features = reduce_features_to_60(features)
        feature_names = feature_names[:60]  # Ensure the names list matches the reduced features
    else:
        print(f"Skipping {filename}: expected at least 60 features, got {len(features)}")
        return None, None

    return features, feature_names

def load_and_prepare_data(base_dir):
    all_data = []
    all_feature_names = None

    for client_dir in os.listdir(base_dir):
        client_path = os.path.join(base_dir, client_dir)
        for server_dir in os.listdir(client_path):
            server_path = os.path.join(client_path, server_dir)
            for timestamp_dir in os.listdir(server_path):
                path = os.path.join(server_path, timestamp_dir)
                json_file = os.path.join(path, "input.json")
                csv_file = os.path.join(path, "result.csv")
                
                with open(json_file, 'r') as f:
                    json_data = json.load(f)
                
                result_data = pd.read_csv(csv_file)
                features, feature_names = extract_features(json_data, json_file)
                if features is not None:
                    if all_feature_names is None:
                        all_feature_names = feature_names  # Set once from the first valid extraction
                    output = result_data.iloc[0, 1:].values  # mean_1, stdev_1, mean_2, stdev_2
                    all_data.append(features + list(output))

    return all_data, all_feature_names

def save_prepared_data_to_csv(all_data, feature_names, output_csv):
    columns = feature_names + ['mean_1', 'stdev_1', 'mean_2', 'stdev_2']
    data_df = pd.DataFrame(all_data, columns=columns)
    data_df.dropna(inplace=True)  # Remove qualquer linha com NaN
    data_df.to_csv(output_csv, index=False)

def main():
    base_dir = os.path.join(os.getcwd(), "..", "converted_input") 
    output_csv = "prepared_data.csv"

    all_data, feature_names = load_and_prepare_data(base_dir)
    print(f"Total de amostras preparadas: {len(all_data)}")

    save_prepared_data_to_csv(all_data, feature_names, output_csv)
    print(f"Dados de treinamento salvos em {output_csv}")

if __name__ == "__main__":
    main()
