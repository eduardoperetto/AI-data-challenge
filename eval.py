import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
import joblib
import os
from datetime import datetime

# Definições dos modelos e flags
# MODEL_FILE = "old_models/random_forest_3.5F_model_10.79.pkl"
MODEL_FILE = "random_forest_model_12.58_mean_discord_wortt.pkl"

# MODEL_FILE_STD = "old_models/random_forest_model_stdevs_max_f_7.89.pkl"
MODEL_FILE_STD = "random_forest_model_stdevs_9.30_discard_wortt.pkl"

# MODEL_FILE_WO_TR = "old_models/random_forest_model_10.31_wo_alldata.pkl"
MODEL_FILE_WO_TR = "random_forest_model_7.18_mean_wortt2.pkl"

# MODEL_FILE_WO_TR_STD = "old_models/random_forest_model_stdevs_7.95_wo_alldata.pkl"
MODEL_FILE_WO_TR_STD = "random_forest_model_stdevs_6.41_wortt.pkl"

USE_ONLY_AVG = False
USE_AVG_WHEN_NO_TR = False
USE_AVG_FOR_ALL_STDEV = False

MODEL_PREDICTS_ONLY_MEAN = False
MODEL_PREDICTS_ONLY_FIRST_MEAN = False
DIFF_MODELS_FOR_MEAN_STD = True

USE_AVG_FOR_ALL_STDEV_WHEN_NO_TR = False
USE_DIFF_MODELS_WHEN_NO_TR = True

USING_RESULT_AS_DIFF_FROM_LAST = False
COLUMN_ID_LAST_RATE_MEAN=22
COLUMN_ID_LAST_RATE_STD=23

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

def calc_predictions(data_df, X, model, model_std, model_wo, model_wo_std):
    predictions = []

    global num_files
    num_files = len(X)

    for i, row in X.iterrows():
        avg_mean = row['rates_mean']
        avg_std = row['rates_stdev']
        has_tr_rtt = row['tr0_rtt_sum'] != 0 and row['rtt0_mean'] != 0
        
        if USE_ONLY_AVG:
            last_mean = row['dash_last_rate']
            last_std = row['dash_last_rate_std']
            predictions.append([avg_mean, avg_std, avg_mean, avg_std])
        elif not has_tr_rtt:
            global num_files_wo_rtt_tr
            num_files_wo_rtt_tr += 1

            if USE_AVG_WHEN_NO_TR:
                last_mean = row['dash_last_rate']
                last_std = row['dash_last_rate_std']
                predictions.append([avg_mean, avg_std, avg_mean, avg_std])
            else:
                if DIFF_MODELS_FOR_MEAN_STD:
                    columns_to_select = [
                        'client_id', 'server_id', 'dash0_rate_mean', 'dash0_rate_stdev', 'dash1_rate_mean', 'dash1_rate_stdev',
                        'dash2_rate_mean', 'dash2_rate_stdev', 'dash3_rate_mean', 'dash3_rate_stdev', 'dash4_rate_mean', 'dash4_rate_stdev',
                        'dash5_rate_mean', 'dash5_rate_stdev', 'dash6_rate_mean', 'dash6_rate_stdev', 'dash7_rate_mean', 'dash7_rate_stdev',
                        'dash8_rate_mean', 'dash8_rate_stdev', 'dash9_rate_mean', 'dash9_rate_stdev', 'dash_last_rate', 'dash_last_rate_std',
                        'rtt0_mean', 'rtt0_stdev','rtt1_mean', 'rtt1_stdev', 'rtt2_mean', 'rtt2_stdev', 'rtt3_mean', 'rtt3_stdev', 'rtt4_mean', 
                        'rtt4_stdev', 'tr0_rtt_sum', 'tr0_rtt_stdev', 'tr1_rtt_sum', 'tr1_rtt_stdev', 'tr2_rtt_sum', 'tr2_rtt_stdev', 'tr3_rtt_sum', 
                        'tr3_rtt_stdev', 'tr4_rtt_sum', 'tr4_rtt_stdev', 'tr_jumps_std', 'rates_mean', 'rates_stdev'
                    ]
                    X_row = row[columns_to_select].values.reshape(1, -1)
                    y_mean = make_predictions(model_wo, X_row).flatten().tolist()
                    y_std = make_predictions(model_wo_std, X_row).flatten().tolist()
                    if USING_RESULT_AS_DIFF_FROM_LAST:
                        last_mean = X_row[0][COLUMN_ID_LAST_RATE_MEAN]
                        last_std = X_row[0][COLUMN_ID_LAST_RATE_STD]
                        y_pred = [y_mean[0] + last_mean, y_std[0] + last_std, y_mean[1] + last_mean, y_std[1] + last_std]
                    else:
                        y_pred = [y_mean[0], y_std[0], y_mean[1], y_std[1]]
                else:
                    X_row = row.values.reshape(1, -1)
                    y_pred = make_predictions(model_wo, X_row)
                    y_pred = y_pred.flatten().tolist()

                if MODEL_PREDICTS_ONLY_FIRST_MEAN:
                    second_pred = avg_mean
                else:
                    second_pred = y_pred[1] if MODEL_PREDICTS_ONLY_MEAN else y_pred[2]

                if USE_AVG_FOR_ALL_STDEV_WHEN_NO_TR:
                    y_pred = [y_pred[0], avg_std, second_pred, avg_std]
                
                predictions.append(y_pred)
        else:
            X_row = X.iloc[i, :].values.reshape(1, -1)
            
            if DIFF_MODELS_FOR_MEAN_STD:
                y_mean = make_predictions(model, X_row).flatten().tolist()
                y_std = make_predictions(model_std, X_row).flatten().tolist()
                if USING_RESULT_AS_DIFF_FROM_LAST:
                    last_mean = X_row[0][COLUMN_ID_LAST_RATE_MEAN]
                    last_std = X_row[0][COLUMN_ID_LAST_RATE_STD]
                    y_pred = [y_mean[0] + last_mean, y_std[0] + last_std, y_mean[1] + last_mean, y_std[1] + last_std]
                else:
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
    global num_files, num_files_wo_rtt_tr
    num_files = 0
    num_files_wo_rtt_tr = 0

    input_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./prepared_data.csv")

    model_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"./{MODEL_FILE}")
    model_file_std = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"./{MODEL_FILE_STD}")
    model_file_wo = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"./{MODEL_FILE_WO_TR}")
    model_file_wo_std = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"./{MODEL_FILE_WO_TR_STD}")

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

    if USE_DIFF_MODELS_WHEN_NO_TR:
        model_data_wo = joblib.load(model_file_wo)
        model_wo = model_data_wo['model']
        model_data_wo_std = joblib.load(model_file_wo_std)
        model_wo_std = model_data_wo_std['model']
    else:
        model_wo = model_wo_std = None

    # Calcula as previsões
    predictions = calc_predictions(data_df, X, model, model_std, model_wo, model_wo_std)

    # Salva as previsões no CSV
    save_predictions_to_csv(ids, predictions)

    print(f"Total number of files: {num_files}")
    print(f"Number of files without RTT or TR: {num_files_wo_rtt_tr}")

if __name__ == "__main__":
    main()
