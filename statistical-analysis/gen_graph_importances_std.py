import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Updated dataset processing
# Updated dataset processing
updated_data = [
    ("RF", "Enriched", "std_1", [("rates_stdev", 0.1373, 0.0040), ("rates_mean", 0.0791, 0.0040), ("dash_last_rate_std", 0.0712, 0.0029), 
                                 ("dash_last_rate", 0.0549, 0.0016), ("dash0_rate_mean", 0.0279, 0.0013), ("client_server_id", 0.0187, 0.0012)]),
    ("RF", "Base", "std_1", [("rates_stdev", 0.2065, 0.0069), ("rates_mean", 0.1270, 0.0078), ("dash_last_rate_std", 0.1197, 0.0036), 
                             ("dash_last_rate", 0.0894, 0.0039), ("client_server_id", 0.0504, 0.0026), ("dash0_rate_mean", 0.0482, 0.0034)]),
    ("XGB", "Base", "std_1", [("rates_stdev", 0.2865, 0.0743), ("rates_mean", 0.1574, 0.0768), ("dash_last_rate_std", 0.1048, 0.0248), 
                              ("dash_last_rate", 0.0683, 0.0189), ("client_server_id", 0.0505, 0.0079), ("dash0_rate_mean", 0.0318, 0.0085)]),
    ("XGB", "Enriched", "std_1", [("rates_stdev", 0.2020, 0.0509), ("rates_mean", 0.1473, 0.0605), ("dash_last_rate_std", 0.0709, 0.0175), 
                                  ("client_server_id", 0.0518, 0.0085), ("dash_last_rate", 0.0479, 0.0084), ("dash0_rate_mean", 0.0288, 0.0126)]),
]

# Extract all features
features = set()
for _, _, _, feats in updated_data:
    for feat, _, _ in feats:
        features.add(feat)

features = sorted(features)
models = ["RF Base std_1", "RF Enriched std_1",
          "XGB Base std_1", "XGB Enriched std_1"]

# Create dictionary for feature importance values
feature_dict = {feat: {model: (0, 0) for model in models} for feat in features}

for model, feature_set, value_type, feats in updated_data:
    model_name = f"{model} {feature_set} {value_type}"
    for feat, value, std in feats:
        feature_dict[feat][model_name] = (value, std)

# Convert to DataFrame
df = pd.DataFrame.from_dict(feature_dict, orient="index")
df = df.applymap(lambda x: x if isinstance(x, tuple) else (0, 0))

# Sort features based on overall importance (sum of means across all models)
df["Total Importance"] = df.apply(lambda row: sum([val[0] for val in row]), axis=1)
df = df.sort_values(by="Total Importance", ascending=False).drop(columns=["Total Importance"])

# Plot settings
x = np.arange(len(df.index))
width = 0.16

fig, ax = plt.subplots(figsize=(14, 7))

# Bar plots with error bars
for i, model in enumerate(models):
    means = [df.loc[feat][model][0] for feat in df.index]
    stds = [df.loc[feat][model][1] for feat in df.index]
    ax.bar(x + i * width, means, width, yerr=stds, label=model[:-2], capsize=5)

# Formatting
ax.set_xlabel("Features", fontsize=14)
ax.set_ylabel("Importance Value", fontsize=18)
ax.set_xticks(x + width * 3.5)
ax.set_xticklabels(df.index, rotation=45, ha="right", fontsize=16)
ax.legend(fontsize=16)
ax.tick_params(axis='y', labelsize=14)

plt.tight_layout()
plt.savefig("feat_import_std.pdf")