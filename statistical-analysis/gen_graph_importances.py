import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Updated dataset processing
updated_data = [
    ("RF", "Enriched", "mean_1", [("rates_mean", 0.1345, 0.0031), ("dash_last_rate", 0.0925, 0.0017), ("rates_stdev", 0.0681, 0.0038), 
                                  ("dash_last_rate_std", 0.0405, 0.0018), ("dash0_rate_mean", 0.0391, 0.0017), ("rtt0_mean", 0.0359, 0.0010)]),
    ("RF", "Base", "mean_1", [("rates_mean", 0.2131, 0.0101), ("dash_last_rate", 0.1552, 0.0051), ("rates_stdev", 0.1148, 0.0060), 
                              ("dash0_rate_mean", 0.0779, 0.0053), ("dash_last_rate_std", 0.0699, 0.0027)]),
    ("XGB", "Base", "mean_1", [("rates_mean", 0.2919, 0.0098), ("dash_last_rate", 0.0964, 0.0073), ("rates_stdev", 0.0722, 0.0067), 
                               ("dash0_rate_mean", 0.0374, 0.0040), ("dash_last_rate_std", 0.0363, 0.0051)]),
    ("XGB", "Enriched", "mean_1", [("rates_mean", 0.1947, 0.0081), ("dash_last_rate", 0.0632, 0.0060), ("rates_stdev", 0.0483, 0.0062), 
                                   ("rtt0_mean", 0.0303, 0.0019), ("dash_last_rate_std", 0.0265, 0.0038), ("dash0_rate_mean", 0.0249, 0.0034)])
]

# Extract all features
features = set()
for _, _, _, feats in updated_data:
    for feat, _, _ in feats:
        features.add(feat)

features = sorted(features)
models = ["RF Base mean_1", "RF Enriched mean_1",
          "XGB Base mean_1", "XGB Enriched mean_1"]

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
plt.savefig("feat_import_mean.pdf")