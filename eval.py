import pandas as pd
import joblib
import os
from datetime import datetime

# Nome do arquivo do modelo
MODEL_FILE = "knn_9.23_tr.pkl"

def load_data_for_prediction(input_csv):
    """
    Carrega os dados de entrada para previsão. 
    Assume que a última coluna é 'id' e que as quatro colunas finais de resultados estão ausentes.
    """
    data_df = pd.read_csv(input_csv)
    ids = data_df['id'].values  # Armazena os IDs
    X = data_df.drop(columns=['id']).values  # Remove a coluna 'id'
    return data_df, ids, X

def make_predictions(model, X):
    """
    Utiliza o modelo carregado para fazer previsões no conjunto de dados X.
    """
    y_pred = model.predict(X)
    return y_pred

def save_predictions_to_csv(ids, predictions, output_dir="eval"):
    """
    Salva as previsões feitas pelo modelo em um arquivo CSV na pasta 'eval'.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Data e hora para o nome do arquivo
    timestamp = datetime.now().strftime("%d_%m_%H_%M")
    output_csv = os.path.join(output_dir, f"{timestamp}.csv")
    
    # Criação do DataFrame com os resultados
    results_df = pd.DataFrame(predictions, columns=['mean_1', 'stdev_1', 'mean_2', 'stdev_2'])
    results_df.insert(0, 'id', ids)  # Insere a coluna 'id' no início

    # Salva o DataFrame em um arquivo CSV
    results_df.to_csv(output_csv, index=False)
    print(f"Previsões salvas em {output_csv}")

def main():
    input_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./prepared_data.csv")
    model_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"./{MODEL_FILE}")

    # Carrega os dados para previsão
    data_df, ids, X = load_data_for_prediction(input_csv)

    # Carrega o modelo pré-treinado
    model_data = joblib.load(model_file)
    model = model_data['model']

    predictions = []

    for i, row in data_df.iterrows():
        if row['tr1_jump_count'] == 0:
            # Usar os valores diretamente de 'rates_mean' e 'rates_stdev'
            mean_1 = mean_2 = row['rates_mean']
            stdev_1 = stdev_2 = row['rates_stdev']
            predictions.append([mean_1, stdev_1, mean_2, stdev_2])
        else:
            # Usar o modelo para prever os valores
            X_row = row.drop(labels=['id']).values.reshape(1, -1)
            y_pred = make_predictions(model, X_row)
            predictions.append(y_pred.flatten().tolist())

    # Salva as previsões no CSV
    save_predictions_to_csv(ids, predictions)

if __name__ == "__main__":
    main()
