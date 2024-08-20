import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
import os

def load_data_from_csv(input_csv):
    data_df = pd.read_csv(input_csv)
    return data_df

def calculate_predictions(data_df):
    predictions = []

    for i, row in data_df.iterrows():
        if row['tr1_jump_count'] == 0:
            # Usar os valores diretamente de 'rates_mean' e 'rates_stdev'
            mean_1 = mean_2 = row['rates_mean']
            stdev_1 = stdev_2 = row['rates_stdev']
            predictions.append([mean_1, stdev_1, mean_2, stdev_2])

    return predictions

def evaluate_predictions(data_df, predictions):
    # Valores reais
    y_true = data_df[['mean_1', 'stdev_1', 'mean_2', 'stdev_2']].values
    
    # Valores previstos
    y_pred = predictions

    # Calcular MAPE
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f"MAPE: {mape * 100:.2f}%")
    return mape

def main():
    input_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../prepared_data.csv")
    
    # Carregar dados do CSV
    data_df = load_data_from_csv(input_csv)
    
    # Calcular as previsões
    predictions = calculate_predictions(data_df)
    
    # Avaliar as previsões com MAPE
    evaluate_predictions(data_df, predictions)

if __name__ == "__main__":
    main()
