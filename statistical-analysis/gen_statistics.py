import os
import pandas as pd
import numpy as np
import datetime
from scipy.stats import kruskal
import scikit_posthocs as sp

SIGNIF_MIN_P = 0.05

def load_data(filename):
    print(f"Loading data from {filename}")
    return pd.read_csv(filename, sep=";")

def group_by_category(df, category):
    print(f"Grouping data for category: {category}")
    # Agrupando tanto MAPE quanto TrainTime
    grouped = df[df["MeanOrStd?"] == category].groupby(
        ["Model", "MeanOrStd?", "ConsiderRTT_TR?"]
    ).agg({
        "MAPE": list,
        "TrainTime": list
    }).reset_index()
    print(grouped)
    return grouped

def run_kruskal_test(grouped):
    groups = []
    print("Preparing groups for Kruskal-Wallis test")
    for _, row in grouped.iterrows():
        groups.append(row["MAPE"])
    print(f"Groups: {groups}")
    if len(groups) < 2 or any(len(g) < 2 for g in groups):
        print("Not enough data for Kruskal-Wallis test")
        return None, None
    stat, p = kruskal(*groups)
    return stat, p

def run_dunn_test(grouped):
    print("Preparing data for Dunn's post hoc test")
    data = []
    for _, row in grouped.iterrows():
        group_label = f"{row['Model']}_{row['MeanOrStd?']}_{row['ConsiderRTT_TR?']}"
        for val in row["MAPE"]:
            data.append({"score": val, "group": group_label})
    d = pd.DataFrame(data)
    print(d)
    return sp.posthoc_dunn(d, val_col="score", group_col="group", p_adjust='bonferroni')

def analyze_category(df, category, output_file, medians_list):
    print(f"\nProcessing category: {category}")
    output_file.write(f"Category: {category}\n")
    grouped = group_by_category(df, category)
    if grouped.shape[0] < 2:
        print(f"Not enough groups for category {category}")
        output_file.write("Not enough groups\n\n")
        return
    has_multiple = all(len(mape) > 1 and len(train_time) > 1 for mape, train_time in zip(grouped["MAPE"], grouped["TrainTime"]))
    if not has_multiple:
        print(f"Not all groups have multiple observations for category {category}")
        output_file.write("Not all groups have multiple observations\n\n")
        return
    
    # Cálculo das medianas dos MAPEs e dos Tempos de Treinamento para cada grupo
    medians = grouped.copy()
    medians['Median_MAPE'] = grouped['MAPE'].apply(np.median)
    medians['Median_TrainTime'] = grouped['TrainTime'].apply(np.median)
    medians_list.append(medians[['Model', 'MeanOrStd?', 'ConsiderRTT_TR?', 'Median_MAPE', 'Median_TrainTime']])
    
    stat, p = run_kruskal_test(grouped)
    if stat is not None and p is not None:
        print(f"Kruskal-Wallis H-test for {category}: H={stat}, p={p}")
        output_file.write(f"Kruskal-Wallis H-test for {category}: H={stat}, p={p}\n")
        if p < SIGNIF_MIN_P:
            print("Significant differences found. Running Dunn's post hoc test.")
            output_file.write("Significant differences found. Running Dunn's post hoc test.\n")
            p_vals = run_dunn_test(grouped)
            output_file.write("Dunn's test p-values:\n")
            output_file.write(p_vals.to_string())
            output_file.write("\n\n")
        else:
            print("No significant differences found.")
            output_file.write("No significant differences found.\n\n")
    else:
        print("Kruskal-Wallis test could not be performed due to insufficient data.")
        output_file.write("Kruskal-Wallis test could not be performed due to insufficient data.\n\n")

def main():
    filename = "input.csv"
    df = load_data(filename)
    categories = df["MeanOrStd?"].unique()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_filename = f"analysis_{timestamp}.txt"
    medians_output_filename = f"median_mapes_{timestamp}.csv"  # Nome do arquivo para as medianas
    medians_list = []  # Lista para armazenar as medianas
    with open(out_filename, "w") as output_file:
        for category in categories:
            analyze_category(df, category, output_file, medians_list)
    # Concatenar todas as medianas e salvar em CSV
    if medians_list:
        median_df = pd.concat(medians_list, ignore_index=True)
        median_df.to_csv(medians_output_filename, index=False)
        print(f"\nMedian MAPEs and TrainTimes saved to {medians_output_filename}")
    else:
        print("\nNo median MAPEs or TrainTimes to save.")
    print(f"\nAnalysis complete. Results saved to {out_filename}")

if __name__ == "__main__":
    main()
