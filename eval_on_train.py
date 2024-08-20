import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
import os

MODEL_FILE = "knn_2.31.pkl"
USE_AVG_WHEN_NO_TR = True

def load_data_from_csv(input_csv):
    data_df = pd.read_csv(input_csv)
    X = data_df.iloc[:, :-4]  # Colunas de entrada para a previsão
    y = data_df.iloc[:, -4:].values  # Últimas 4 colunas como os valores reais
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    return data_df, X_test, y_test

def calculate_avgs(data_df):
    # Calcular mean_1 e stdev_1 como a média das colunas dashX_rate_mean e dashX_rate_stdev, respectivamente
    rate_means = [col for col in data_df.columns if 'dash' in col and 'rate_mean' in col]
    rate_stdevs = [col for col in data_df.columns if 'dash' in col and 'rate_stdev' in col]

    means = data_df[rate_means].mean(axis=1)
    stdevs = data_df[rate_stdevs].mean(axis=1)
    
    return means, stdevs

def calc_predictions(X, data_df, model):
    avg_means, avg_stdevs = calculate_avgs(data_df)

    predictions = []
    for i, row in X.iterrows():
        if USE_AVG_WHEN_NO_TR and row['tr1_jump_count'] == 0:
            # Usar os valores diretamente de 'rates_mean' e 'rates_stdev'
            mean_1 = mean_2 = avg_means[i]
            stdev_1 = stdev_2 = avg_stdevs[i]
            predictions.append([mean_1, stdev_1, mean_2, stdev_2])
        else:
            # Usar o modelo para prever os valores
            X_row = row.values.reshape(1, -1)
            y_pred = make_predictions(model, X_row)
            predictions.append(y_pred.flatten().tolist())
            
    return pd.DataFrame(predictions, columns=['mean_1', 'stdev_1', 'mean_2', 'stdev_2'])

def make_predictions(model, X):
    """
    Utiliza o modelo carregado para fazer previsões no conjunto de dados X.
    """
    y_pred = model.predict(X)
    return y_pred

def evaluate_predictions(y_true, y_pred):
    # Calcular MAPE
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f"MAPE: {mape * 100:.2f}%")
    return mape

def main():
    input_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./prepared_data.csv")
    model_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"./{MODEL_FILE}")
    
    # Carregar dados do CSV
    data_df, X, y_true = load_data_from_csv(input_csv)
    
    # Carregar o modelo salvo
    model_data = joblib.load(model_file)
    model = model_data['model']

    # Calcular as previsões
    y_pred_df = calc_predictions(X, data_df, model)
    
    # Avaliar as previsões com MAPE
    evaluate_predictions(y_true, y_pred_df)

if __name__ == "__main__":
    main()
