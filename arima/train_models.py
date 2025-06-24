import pandas as pd
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
import joblib
import os

USING_RESULT_AS_DIFF = False

def load_data_from_csv(input_csv):
    data_df = pd.read_csv(input_csv, parse_dates=True, index_col=0)
    
    mean_1 = data_df['mean_1']
    stdev_1 = data_df['stdev_1']
    mean_2 = data_df['mean_2']
    stdev_2 = data_df['stdev_2']

    if USING_RESULT_AS_DIFF:
        global avg_mean, avg_std
        avg_mean = data_df['rates_mean'].values
        avg_std = data_df['rates_stdev'].values

    mean_1 = np.log1p(mean_1)
    stdev_1 = np.log1p(stdev_1)
    mean_2 = np.log1p(mean_2)
    stdev_2 = np.log1p(stdev_2)

    train_size = int(len(mean_1) * 0.75)

    train = {
        'mean_1': mean_1.iloc[:train_size],
        'stdev_1': stdev_1.iloc[:train_size],
        'mean_2': mean_2.iloc[:train_size],
        'stdev_2': stdev_2.iloc[:train_size]
    }

    test = {
        'mean_1': mean_1.iloc[train_size:],
        'stdev_1': stdev_1.iloc[train_size:],
        'mean_2': mean_2.iloc[train_size:],
        'stdev_2': stdev_2.iloc[train_size:]
    }

    return train, test

def train_arima_model(train_series):
    arima_model = auto_arima(train_series, 
                             start_p=1, start_q=1,
                             max_p=5, max_q=5,
                             seasonal=False, 
                             trace=True, 
                             error_action='ignore', 
                             suppress_warnings=True,
                             stepwise=True)
    print(arima_model.summary())
    return arima_model

def evaluate_model(model, train_series, test_series, target_name):
    model.fit(train_series)
    
    predictions = model.predict(n_periods=len(test_series))
    
    if USING_RESULT_AS_DIFF:
        predictions += avg_mean[:len(predictions)]
        test_series += avg_mean[:len(test_series)]

    predictions = np.expm1(predictions)
    test_series = np.expm1(test_series)
    
    mape = mean_absolute_percentage_error(test_series, predictions)
    print(f"{target_name} MAPE: {mape * 100:.2f}%")
    return mape * 100

def save_model(model, model_file):
    joblib.dump({'model': model}, model_file)
    print(f"Model saved to {model_file}")

def main():
    input_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../prepared_data.csv")
    
    train_series, test_series = load_data_from_csv(input_csv)
    
    models = {}
    for target in train_series:
        print(f"Training ARIMA model for {target}...")
        arima_model = train_arima_model(train_series[target])
        
        print(f"Evaluating the model for {target} with MAPE...")
        mape = evaluate_model(arima_model, train_series[target], test_series[target], target)
        
        model_file = f"arima_{target}_{mape:.2f}.pkl"
        save_model(arima_model, model_file)
        models[target] = arima_model

    return models

if __name__ == "__main__":
    main()
