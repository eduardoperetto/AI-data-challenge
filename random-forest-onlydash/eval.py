import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_percentage_error

def evaluate_model(model, X, y):
    """Avalia o modelo utilizando o MAPE."""
    predictions = model.predict(X)
    return mean_absolute_percentage_error(y, predictions)

def main():
    """Função principal que coordena o fluxo de execução do script."""
    # Carregar os dados de teste e o modelo treinado
    test_data = pd.read_csv("test_data.csv")
    model = joblib.load("knn_model.pkl")

    if 'mean_rate' in test_data.columns:
        X_test = test_data.drop(columns=['mean_rate'])  # Features
        y_test = test_data['mean_rate']  # Target

        mape = evaluate_model(model, X_test, y_test)
        print(f"MAPE: {mape:.4f}")
    else:
        raise ValueError("Coluna 'mean_rate' não encontrada no arquivo CSV de teste.")

if __name__ == "__main__":
    main()
