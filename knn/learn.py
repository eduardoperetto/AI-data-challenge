import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import joblib

def train_knn_model(X, y, n_neighbors=5):
    """Treina um modelo K-Nearest Neighbors com os dados fornecidos."""
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X, y)
    return model

def main():
    """Função principal que coordena o fluxo de execução do script."""
    # Carregar os dados preparados
    data = pd.read_csv("train_data.csv")

    if 'mean_rate' in data.columns:
        X = data.drop(columns=['mean_rate'])  # Features
        y = data['mean_rate']  # Target

        model = train_knn_model(X, y)
        joblib.dump(model, "knn_model.pkl")
    else:
        raise ValueError("Coluna 'mean_rate' não encontrada no arquivo CSV.")

if __name__ == "__main__":
    main()
