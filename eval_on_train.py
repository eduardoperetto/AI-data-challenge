import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import joblib
import os
from sklearn.preprocessing import StandardScaler
import numpy as np

# Definições dos modelos e flags
MODEL_FILE = "random_forest_3.5F_model_10.79.pkl"
MODEL_FILE_STD = "random_forest_model_stdevs_max_f_7.89.pkl"
MODEL_FILE_WO_TR = "random_forest_model_10.31_wo_alldata.pkl"
MODEL_FILE_WO_TR_STD = "random_forest_model_stdevs_7.95_wo_alldata.pkl"

USE_ONLY_AVG = False
USE_AVG_WHEN_NO_TR = True
USE_AVG_FOR_ALL_STDEV = False

MODEL_PREDICTS_ONLY_MEAN = False
MODEL_PREDICTS_ONLY_FIRST_MEAN = False
DIFF_MODELS_FOR_MEAN_STD = True

USE_AVG_FOR_ALL_STDEV_WHEN_NO_TR = False
USE_DIFF_MODELS_WHEN_NO_TR = True

USING_RESULT_AS_DIFF_FROM_LAST = False
COLUMN_ID_LAST_RATE_MEAN = 22
COLUMN_ID_LAST_RATE_STD = 23

USING_DIFF_FROM_AVG = False
DISCARD_MEAN = False
USE_MOVING_AVG = False
USE_WEIGHTED_AVG = False

NORMALIZE = False

def load_data_from_csv(input_csv, random_state):
    data_df = pd.read_csv(input_csv).dropna()

    last_x_index = -6 if DISCARD_MEAN else -4
    if NORMALIZE:
        scaler = StandardScaler()
        data_df.iloc[:, :last_x_index] = scaler.fit_transform(data_df.iloc[:, :last_x_index])

    X = data_df.iloc[:, :last_x_index]
    y = data_df.iloc[:, -4:]

    if USING_DIFF_FROM_AVG: 
        y['mean_1'] = y['mean_1'] + data_df['rates_mean']
        y['mean_2'] = y['mean_2'] + data_df['rates_mean']
        y['stdev_1'] = y['stdev_1'] + data_df['rates_stdev']
        y['stdev_2'] = y['stdev_2'] + data_df['rates_stdev']
    elif USING_RESULT_AS_DIFF_FROM_LAST:
        y['mean_1'] = y['mean_1'] + data_df['dash_last_rate']
        y['mean_2'] = y['mean_2'] + data_df['dash_last_rate']
        y['stdev_1'] = y['stdev_1'] + data_df['dash_last_rate_std']
        y['stdev_2'] = y['stdev_2'] + data_df['dash_last_rate_std']

    y = y.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state)

    return data_df, X_test, y_test

def calculate_moving_avgs(data_df, window=3):
    rate_means = [col for col in data_df.columns if 'dash' in col and 'rate_mean' in col]
    rate_stdevs = [col for col in data_df.columns if 'dash' in col and 'rate_stdev' in col]

    means = data_df[rate_means].rolling(window=window, axis=1).mean().iloc[:, -1]
    stdevs = data_df[rate_stdevs].rolling(window=window, axis=1).mean().iloc[:, -1]
    
    return means, stdevs

def calc_weigthed_avg(row):
    measures = []
    
    last_real_val = 0
    for id in range(10):
        last_real_val += row[f'dash{id}_rate_mean']
        measures.append(last_real_val)
    gamma = 0.75
    curr_value = 1
    weights = [1]

    for i in range(9):
        curr_value = curr_value*gamma
        weights.append(curr_value)
    weights.reverse()
    means = np.average(measures, weights=weights)
    return means

def calc_predictions(X, data_df, model, model_std, model_wo, model_wo_std):
    predictions = []

    for i, row in X.iterrows():
        avg_mean = row['rates_mean']
        avg_std = row['rates_stdev']
        has_tr_rtt = row['tr0_rtt_sum'] != 0 and row['rtt0_mean'] != 0
        
        if USE_ONLY_AVG:
            last_mean = row['dash_last_rate']
            last_std = row['dash_last_rate_std']
            if USE_WEIGHTED_AVG:
                weighted_mean = calc_weigthed_avg(row)
                predictions.append([weighted_mean, avg_std, weighted_mean, avg_std])
            else:
                predictions.append([avg_mean, avg_std, avg_mean, avg_std])
        elif not has_tr_rtt:
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
                        'rates_mean', 'rates_stdev'
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
            X_row = row.values.reshape(1, -1)

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

def evaluate_predictions(y_true, y_pred, indices):
    y_true_subset = y_true[:, indices]
    y_pred_subset = y_pred[:, indices]

    mape = mean_absolute_percentage_error(y_true_subset, y_pred_subset)
    print(f"MAPE ({indices}): {mape * 100:.2f}%")
    return mape

def main():
    input_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./prepared_data.csv")

    model_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"./{MODEL_FILE}")
    model_file_std = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"./{MODEL_FILE_STD}")
    model_file_wo = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"./{MODEL_FILE_WO_TR}")
    model_file_wo_std = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"./{MODEL_FILE_WO_TR_STD}")

    # Carregar o modelo salvo
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
        model_std = None

    mape_list = []
    random_states = [42, 7, 19, 23, 11]  # Lista de 5 valores aleatórios para random_state

    for random_state in random_states:
        print(f"Executando com random_state: {random_state}")
        
        # Carregar dados do CSV
        data_df, X, y_true = load_data_from_csv(input_csv, random_state)
        
        # Calcular as previsões
        y_pred_df = calc_predictions(X, data_df, model, model_std, model_wo, model_wo_std)
        y_pred = y_pred_df.values
        
        # Avaliar as previsões com MAPE
        mape = evaluate_predictions(y_true, y_pred, [0])
        mape = evaluate_predictions(y_true, y_pred, [1])
        mape = evaluate_predictions(y_true, y_pred, [2])
        mape = evaluate_predictions(y_true, y_pred, [3])
        mape_total = evaluate_predictions(y_true, y_pred, [0, 1, 2, 3])
        mape_list.append(mape_total)

    # Calcular a média dos valores de MAPE
    mape_mean = np.mean(mape_list)
    print(f"MAPE médio das 5 execuções: {mape_mean * 100:.2f}%")

if __name__ == "__main__":
    main()
