import os
import pandas as pd
from datetime import datetime

def load_data_from_csv(input_csv):
    data_df = pd.read_csv(input_csv)
    return data_df

def calculate_predictions(data_df):
    # Calcular mean_1 e stdev_1 como a média das colunas dashX_rate_mean e dashX_rate_stdev, respectivamente
    rate_means = [col for col in data_df.columns if 'dash' in col and 'rate_mean' in col]
    rate_stdevs = [col for col in data_df.columns if 'dash' in col and 'rate_stdev' in col]
    
    data_df['mean_1'] = data_df[rate_means].mean(axis=1)
    data_df['stdev_1'] = data_df[rate_stdevs].mean(axis=1)
    
    # mean_2 = mean_1 and stdev_2 = stdev_1
    data_df['mean_2'] = data_df['mean_1']
    data_df['stdev_2'] = data_df['stdev_1']
    
    return data_df

def save_predictions(data_df, eval_dir):
    # Criar DataFrame com as colunas id, mean_1, stdev_1, mean_2, stdev_2
    eval_df = data_df[['id', 'mean_1', 'stdev_1', 'mean_2', 'stdev_2']]
    
    # Gerar nome do arquivo baseado na data e hora atual
    timestamp = datetime.now().strftime("%d_%m_%H_%M")
    eval_csv_path = os.path.join(eval_dir, f"{timestamp}.csv")
    
    # Criar pasta eval se não existir
    os.makedirs(eval_dir, exist_ok=True)
    
    # Salvar o CSV na pasta eval
    eval_df.to_csv(eval_csv_path, index=False)
    
    print(f"Arquivo salvo em: {eval_csv_path}")

def main():
    input_csv = "prepared_data.csv"
    eval_dir = "eval"
    
    # Carregar dados do CSV
    data_df = load_data_from_csv(input_csv)
    
    # Calcular as previsões
    data_df = calculate_predictions(data_df)
    
    # Salvar as previsões no CSV
    save_predictions(data_df, eval_dir)

if __name__ == "__main__":
    main()
