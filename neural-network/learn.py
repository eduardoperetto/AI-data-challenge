import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from keras_tuner import RandomSearch
import os

USING_RESULT_AS_DIFF = False
LEARN_ONLY_MEAN = False  # Modelo predirá apenas dois valores: mean_1 e mean_2
LEARN_ONLY_FIRST_MEAN = False  # Modelo predirá apenas um valor: mean_1
LEARN_ONLY_STDEV = True  # Modelo predirá apenas dois valores: stdev_1 e stdev_2
NORMALIZE = False  # Normalizar os dados

def load_data_from_csv(input_csv):
    data_df = pd.read_csv(input_csv)
    feature_names = data_df.columns[:-4]  # Captura os nomes das features

    X = data_df.iloc[:, :-4].values

    if LEARN_ONLY_MEAN:
        y = data_df.iloc[:, [-4, -2]].values  # mean_1, mean_2
    elif LEARN_ONLY_FIRST_MEAN:
        y = data_df.iloc[:, -4].values  # mean_1
    elif LEARN_ONLY_STDEV:
        y = data_df.iloc[:, [-3, -1]].values  # stdev_1, stdev_2
    else:
        y = data_df.iloc[:, -4:].values  # mean_1, stdev_1, mean_2, stdev_2

    # Normalizando os dados se a flag estiver ativada
    if NORMALIZE:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Dividindo os dados em treinamento (75%) e teste (25%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    return X_train, X_test, y_train, y_test, feature_names

def build_model(hp, input_dim, output_dim):
    model = Sequential()
    # Camadas de entrada com uma quantidade de neurônios variável
    model.add(Dense(units=hp.Int('units_input', min_value=64, max_value=512, step=32),
                    input_dim=input_dim, activation='relu'))
    model.add(Dropout(rate=hp.Float('dropout_input', min_value=0.1, max_value=0.5, step=0.1)))
    
    # Camadas ocultas
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(Dense(units=hp.Int(f'units_{i}', min_value=64, max_value=256, step=32),
                        activation='relu'))
        model.add(Dropout(rate=hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1)))
    
    # Camada de saída
    model.add(Dense(output_dim, activation='linear'))
    
    # Compilando o modelo
    model.compile(optimizer=Adam(learning_rate=hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                  loss='mean_absolute_error')
    
    return model

def train_neural_network(X_train, y_train):
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1

    tuner = RandomSearch(
        lambda hp: build_model(hp, input_dim, output_dim),
        objective='val_loss',
        max_trials=10,  # Número de diferentes modelos a testar
        executions_per_trial=2,  # Número de vezes para executar o treinamento em cada teste
        directory='tuner_results',
        project_name='neural_network_tuning')

    tuner.search(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2)

    # Obtém o melhor modelo encontrado
    best_model = tuner.get_best_models(num_models=1)[0]

    return best_model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f"MAPE: {mape * 100:.2f}%")
    return f"{mape * 100:.2f}"

def save_model(model, model_file):
    model.save(model_file)
    print(f"Modelo salvo em {model_file}")

def main():
    input_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../prepared_data.csv")

    X_train, X_test, y_train, y_test, feature_names = load_data_from_csv(input_csv)
    
    print("Tuning e treinando a Rede Neural...")
    best_nn = train_neural_network(X_train, y_train)
    
    print("Avaliando o modelo com MAPE...")
    mape = evaluate_model(best_nn, X_test, y_test)
    
    suffix = "_stdevs" if LEARN_ONLY_STDEV else ""
    model_file = f"neural_network{suffix}_{mape}.h5"
    save_model(best_nn, model_file)

if __name__ == "__main__":
    main()
