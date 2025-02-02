import csv
import statistics

def aggregate_feature_importances(input_csv='input.csv', output_csv='output.csv'):
    """
    Aggregates feature importances from the input CSV and writes the top 5 features
    plus specified features ("dash0_rate_mean", "rtt0_mean") for each combination of
    Model, MeanOrStd, and ConsiderRTT_TR to the output CSV.

    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to the output CSV file.
    """
    # Features to always include if they exist
    always_include_features = {"dash0_rate_mean", "rtt0_mean", "client_server_id"}

    # Data structure to hold feature values grouped by (Model, MeanOrStd, ConsiderRTT_TR)
    grouped_data = {}

    with open(input_csv, mode='r', newline='', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in, delimiter=';')
        
        for row in reader:
            model = row['Model'].strip()
            mean_or_std = row['MeanOrStd?'].strip()
            consider_rtt = row['ConsiderRTT_TR?'].strip()
            feature_str = row['FeatureImportance'].strip()
            
            # Skip target values that end with '_2'
            # if mean_or_std.endswith('_2'):
            #     continue
            
            key = (model, mean_or_std, consider_rtt)
            if key not in grouped_data:
                grouped_data[key] = {}
                
            # Parse the FeatureImportance column:
            # Example: "rates_mean=0.1397<>dash_last_rate=0.0918<>..."
            features = feature_str.split('<>')
            for feat in features:
                feat = feat.strip()
                if not feat:
                    continue
                # Each feat is like "property=value"
                try:
                    prop, val_str = feat.split('=')
                    prop = prop.strip()
                    val = float(val_str.strip())
                except ValueError:
                    # In case of unexpected format, skip this feature
                    continue
                
                if prop not in grouped_data[key]:
                    grouped_data[key][prop] = []
                grouped_data[key][prop].append(val)

    # Prepare and write the output CSV
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out, delimiter=';')
        # Write header: Model;Value;FeatureSet;Features
        writer.writerow(['Model', 'Value', 'FeatureSet', 'Features'])

        for (model, mean_or_std, consider_rtt), features_dict in grouped_data.items():
            # Compute mean and std for each feature
            results = []
            for feature_name, values in features_dict.items():
                mean_val = statistics.mean(values)
                # If there's only one value, standard deviation is 0.0
                std_val = statistics.pstdev(values) if len(values) > 1 else 0.0
                results.append((feature_name, mean_val, std_val))
            
            # Sort features by descending mean importance
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Select top 5 features
            top_5 = results[:5]
            top_5_feature_names = {feat[0] for feat in top_5}

            # Prepare a dictionary for quick lookup of mean and std
            feature_stats = {feat[0]: (feat[1], feat[2]) for feat in results}

            # Initialize the list of formatted features with top 5
            formatted_features = [
                f"{feat_name}: {m:.4f} ± {s:.4f}"
                for (feat_name, m, s) in top_5
            ]

            # Add always included features if they exist and are not already in top 5
            for special_feat in always_include_features:
                if special_feat in features_dict and special_feat not in top_5_feature_names:
                    m, s = feature_stats[special_feat]
                    formatted_features.append(f"{special_feat}: {m:.4f} ± {s:.4f}")

            # Determine FeatureSet based on ConsiderRTT_TR?
            features_label = 'Enriched' if consider_rtt.lower() == 'true' else 'Base'
            
            # Join formatted features with "<>"
            features_str = "<>".join(formatted_features)

            # Write the row: Model;Value;FeatureSet;Features
            writer.writerow([model, mean_or_std, features_label, features_str])

if __name__ == '__main__':
    aggregate_feature_importances('input.csv', 'output.csv')
