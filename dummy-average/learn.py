import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
import os

def load_data_from_csv(input_csv):
    data_df = pd.read_csv(input_csv)
    return data_df

def calculate_predictions(data_df):
    # Calcular mean_1 e stdev_1 como a média das colunas dashX_rate_mean e dashX_rate_stdev, respectivamente
    rate_means = [col for col in data_df.columns if 'dash' in col and 'rate_mean' in col]
    rate_stdevs = [col for col in data_df.columns if 'dash' in col and 'rate_stdev' in col]
    
    data_df['pred_mean_1'] = data_df[rate_means].mean(axis=1)
    data_df['pred_stdev_1'] = data_df[rate_stdevs].mean(axis=1)
    
    # mean_2 = mean_1 and stdev_2 = stdev_1
    data_df['pred_mean_2'] = data_df['pred_mean_1']
    data_df['pred_stdev_2'] = data_df['pred_stdev_1']
    
    return data_df

def evaluate_predictions(data_df):
    # Valores reais
    y_true = data_df[['mean_1', 'stdev_1', 'mean_2', 'stdev_2']].values
    
    # Valores previstos
    y_pred = data_df[['pred_mean_1', 'pred_stdev_1', 'pred_mean_2', 'pred_stdev_2']].values
    
    # Calcular MAPE
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f"MAPE: {mape * 100:.2f}%")
    return mape

def main():
    input_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../prepared_data.csv")
    
    # Carregar dados do CSV
    data_df = load_data_from_csv(input_csv)
    
    # Calcular as previsões
    data_df = calculate_predictions(data_df)
    
    # Avaliar as previsões com MAPE
    evaluate_predictions(data_df)

if __name__ == "__main__":
    main()
