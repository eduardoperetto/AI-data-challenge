import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error
import joblib
import os

USING_RESULT_AS_DIFF = False
LEARN_ONLY_MEAN = False  # Modelo predirá apenas dois valores: mean_1 e mean_2
LEARN_ONLY_FIRST_MEAN = False  # Modelo predirá apenas um valor: mean_1
LEARN_ONLY_STDEV = True # Modelo predirá apenas dois valores: stdev_1 e stdev_2

def load_data_from_csv(input_csv):
    data_df = pd.read_csv(input_csv)
    feature_names = data_df.columns[:-4]  # Captura os nomes das features

    X = data_df.iloc[:, :-4].values
    y = data_df.iloc[:, -4:].values

    if LEARN_ONLY_MEAN:
        y = data_df.iloc[:, [-4, -2]].values
    elif LEARN_ONLY_FIRST_MEAN:
        y = data_df.iloc[:, -4].values
    elif LEARN_ONLY_STDEV:
        y = data_df.iloc[:, [-3, -1]].values

    # Dividindo os dados em treinamento (75%) e teste (25%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    return X_train, X_test, y_train, y_test, feature_names

def train_lasso(X_train, y_train):
    lasso = Lasso(random_state=42, positive=True)
    param_grid = {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 50.0, 99.0],  # Different values of alpha for regularization
        'max_iter': [20000, 30000, 100000]  # Number of iterations to converge
    }
    grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
    
    try:
        grid_search.fit(X_train, y_train)
    except Exception as e:
        raise e

    print(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f"MAPE: {mape * 100:.2f}%")
    return f"{mape * 100:.2f}"

def print_penalized_features(model, feature_names):
    # Obtém os coeficientes do modelo
    coef = model.coef_

    if coef.ndim == 1:
        # Se for 1D, lidamos diretamente
        features_with_coef = list(zip(feature_names, coef))
        features_with_coef.sort(key=lambda x: abs(x[1]), reverse=True)

        print("\nFeatures e coeficientes (ordenados por importância):")
        for feature, coef_value in features_with_coef:
            print(f" - {feature}: {coef_value:.4f}")

    else:
        # Se for 2D, iteramos sobre cada alvo
        for i, target_coef in enumerate(coef):
            print(f"\nTarget {i + 1}:")
            features_with_coef = list(zip(feature_names, target_coef))
            features_with_coef.sort(key=lambda x: abs(x[1]), reverse=True)

            print("Features e coeficientes (ordenados por importância):")
            for feature, coef_value in features_with_coef:
                print(f" - {feature}: {coef_value:.4f}")

def save_model(model, model_file):
    joblib.dump({'model': model}, model_file)
    print(f"Modelo salvo em {model_file}")

def main():
    input_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../prepared_data.csv")

    X_train, X_test, y_train, y_test, feature_names = load_data_from_csv(input_csv)
    
    print("Treinando o modelo LASSO...")
    best_lasso = train_lasso(X_train, y_train)
    
    print("Avaliando o modelo com MAPE...")
    mape = evaluate_model(best_lasso, X_test, y_test)
    
    print("Imprimindo as features penalizadas...")
    print_penalized_features(best_lasso, feature_names)
    
    suffix = "_stdevs" if LEARN_ONLY_STDEV else ""
    model_file = f"lasso_model{suffix}_{mape}.pkl"
    save_model(best_lasso, model_file)

if __name__ == "__main__":
    main()
