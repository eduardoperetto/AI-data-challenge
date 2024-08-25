import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
import joblib
import os
from datetime import datetime

# Configurações de parâmetros
MODEL_FILE = "random_forest_model_11.83.pkl"
MODEL_FILE_STD = "random_forest_model_stdevs_8.57.pkl"

USE_ONLY_AVG = False
USE_AVG_WHEN_NO_TR = True
USE_AVG_FOR_ALL_STDEV = False

MODEL_PREDICTS_ONLY_MEAN = False
MODEL_PREDICTS_ONLY_FIRST_MEAN = False
DIFF_MODELS_FOR_MEAN_STD = True

USING_DIFF_FROM_AVG = False
DISCARD_AVG = False  # Use for KNN

USE_MOVING_AVG = False

def load_data_for_prediction(input_csv):
    """
    Carrega os dados de entrada para previsão.
    Assume que a última coluna é 'id' e que as quatro colunas finais de resultados estão ausentes.
    """
    data_df = pd.read_csv(input_csv)
    ids = data_df['id'].values  # Armazena os IDs
    X = data_df.drop(columns=['id'])  # Remove a coluna 'id'
    

    return data_df, ids, X

def calculate_moving_avgs(data_df, window=3):
    rate_means = [col for col in data_df.columns if 'dash' in col and 'rate_mean' in col]
    rate_stdevs = [col for col in data_df.columns if 'dash' in col and 'rate_stdev' in col]

    means = data_df[rate_means].rolling(window=window, axis=1).mean().iloc[:, -1]
    stdevs = data_df[rate_stdevs].rolling(window=window, axis=1).mean().iloc[:, -1]
    
    return means, stdevs

def calc_predictions(data_df, X, model, model_std):
    predictions = []

    for i, row in data_df.iterrows():
        avg_mean = row['rates_mean']
        avg_std = row['rates_stdev']
        should_use_avg = USE_ONLY_AVG or (USE_AVG_WHEN_NO_TR and (row['tr0_rtt_sum'] == 0 or row['rtt0_mean'] == 0))
        
        if should_use_avg:
            mean_1 = mean_2 = avg_mean
            stdev_1 = stdev_2 = avg_std
            predictions.append([mean_1, stdev_1, mean_2, stdev_2])
        else:
            X_row = X.iloc[i, :].values.reshape(1, -1)
            
            if DIFF_MODELS_FOR_MEAN_STD:
                y_mean = make_predictions(model, X_row).flatten().tolist()
                y_std = make_predictions(model_std, X_row).flatten().tolist()
                y_pred = [y_mean[0], y_std[0], y_mean[1], y_std[1]]
            else:
                y_pred = make_predictions(model, X_row)
                y_pred = y_pred.flatten().tolist()

            if USING_DIFF_FROM_AVG:
                if MODEL_PREDICTS_ONLY_FIRST_MEAN:
                    y_pred = [y_pred[0] + avg_mean, avg_mean, -1, -1]
                elif MODEL_PREDICTS_ONLY_MEAN:
                    y_pred = [y_pred[0] + avg_mean, y_pred[1] + avg_mean, -1, -1]
                else:
                    y_pred = [y_pred[0] + avg_mean, y_pred[1] + avg_std, y_pred[2] + avg_mean, y_pred[3] + avg_std]

            if MODEL_PREDICTS_ONLY_FIRST_MEAN:
                second_pred = avg_mean
            else:
                second_pred = y_pred[1] if MODEL_PREDICTS_ONLY_MEAN else y_pred[2]
           
            if USE_AVG_FOR_ALL_STDEV:
                y_pred = [y_pred[0], avg_std, second_pred, avg_std]
                
            predictions.append(y_pred)
            
    return pd.DataFrame(predictions, columns=['mean_1', 'stdev_1', 'mean_2', 'stdev_2'])

def make_predictions(model, X):
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
    model_file_std = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"./{MODEL_FILE_STD}")

    # Carrega os dados para previsão
    data_df, ids, X = load_data_for_prediction(input_csv)

    # Carrega o modelo pré-treinado
    model_data = joblib.load(model_file)
    model = model_data['model']

    if DIFF_MODELS_FOR_MEAN_STD:
        model_data_std = joblib.load(model_file_std)
        model_std = model_data_std['model']
    else:
        model_std = None

    # Calcula as previsões
    predictions = calc_predictions(data_df, X, model, model_std)

    # Salva as previsões no CSV
    save_predictions_to_csv(ids, predictions)

if __name__ == "__main__":
    main()
