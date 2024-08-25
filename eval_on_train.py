import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import joblib
import os
from sklearn.preprocessing import StandardScaler

MODEL_FILE = "random_forest_model_11.83.pkl"
MODEL_FILE_STD = "random_forest_model_stdevs_8.57.pkl"

USE_ONLY_AVG = True
USE_AVG_WHEN_NO_TR = True
USE_AVG_FOR_ALL_STDEV=False

MODEL_PREDICTS_ONLY_MEAN=False
MODEL_PREDICTS_ONLY_FIRST_MEAN=False
DIFF_MODELS_FOR_MEAN_STD = True

USING_DIFF_FROM_AVG=False
DISCARD_MEAN = False # Use for KNN when using result as diff
USE_MOVING_AVG = False

NORMALIZE = False # Use for KNN

def load_data_from_csv(input_csv):
    data_df = pd.read_csv(input_csv).dropna()

    if USING_DIFF_FROM_AVG: 
        y['mean_1'] = y['mean_1'] + data_df['rates_mean']
        y['mean_2'] = y['mean_2'] + data_df['rates_mean']
        y['stdev_1'] = y['stdev_1'] + data_df['rates_stdev']
        y['stdev_2'] = y['stdev_2'] + data_df['rates_stdev']

    last_x_index = -6 if DISCARD_MEAN else -4
    if NORMALIZE:
        scaler = StandardScaler()
        data_df.iloc[:, :last_x_index] = scaler.fit_transform(data_df.iloc[:, :last_x_index])

    X = data_df.iloc[:, :last_x_index]  # Colunas de entrada para a previsão
    y = data_df.iloc[:, -4:].values  # Últimas 4 colunas como os valores reais

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=55)

    return data_df, X_test, y_test

def calculate_moving_avgs(data_df, window=3):
    rate_means = [col for col in data_df.columns if 'dash' in col and 'rate_mean' in col]
    rate_stdevs = [col for col in data_df.columns if 'dash' in col and 'rate_stdev' in col]

    means = data_df[rate_means].rolling(window=window, axis=1).mean().iloc[:, -1]
    stdevs = data_df[rate_stdevs].rolling(window=window, axis=1).mean().iloc[:, -1]
    
    return means, stdevs

def calc_predictions(X, data_df, model, model_std):
    predictions = []

    for i, row in X.iterrows():
        avg_mean = row['rates_mean']
        avg_std = row['rates_stdev']
        should_use_avg = USE_ONLY_AVG or (USE_AVG_WHEN_NO_TR and (row['tr0_rtt_sum'] == 0 or row['rtt0_mean'] == 0))
        
        if should_use_avg:
            predictions.append([avg_mean, avg_std, avg_mean, avg_std])
        else:
            X_row = row.values.reshape(1, -1)

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
    
    # Carregar dados do CSV
    data_df, X, y_true = load_data_from_csv(input_csv)
    
    # Carregar o modelo salvo
    model_data = joblib.load(model_file)
    model = model_data['model']

    if DIFF_MODELS_FOR_MEAN_STD:
        model_data_std = joblib.load(model_file_std)
        model_std = model_data_std['model']
    else:
        model_std = None

    # Calcular as previsões
    y_pred_df = calc_predictions(X, data_df, model, model_std)
    y_pred = y_pred_df.values
    
    # Avaliar as previsões com MAPE
    mape_mean = evaluate_predictions(y_true, y_pred, [0])
    mape_stdev = evaluate_predictions(y_true, y_pred, [1])
    mape_mean = evaluate_predictions(y_true, y_pred, [2])
    mape_stdev = evaluate_predictions(y_true, y_pred, [3])
    mape_total = evaluate_predictions(y_true, y_pred, [0, 1, 2, 3])
    
    print(f"Total MAPE: {mape_total * 100:.2f}%")

if __name__ == "__main__":
    main()
