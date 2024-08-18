import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Mapeamento de clientes e servidores para valores numéricos
CLIENTS = {"ba": 0, "rj": 1}
SERVERS = {"ce": 0, "df": 0.33, "es": 0.66, "pi": 1}

CONSIDER_CLIENT_SERVER = 1
WEIGHT_LAST_DATA = 1
NORMALIZE = 0
ONLY_RATE = 0

def extract_dash_features(dash_data, filename):
    """
    Extrai as features combinadas de elapsed, request_ticks, rate e received
    de cada medição DASH, retornando uma lista de listas onde cada sublista contém
    as 4 features combinadas (média e desvio padrão juntos) para uma medição DASH.
    """
    dash_features = []
    for dash in dash_data:
        elapseds = dash['elapsed']
        ticks = dash['request_ticks']
        rates = dash['rate']
        receiveds = dash['received']

        if not elapseds: 
            print(f"Invalid elapseds for {filename}")
            return None
        if not ticks: 
            print(f"Invalid ticks for {filename}")
            return None
        if not rates: 
            print(f"Invalid rates for {filename}")
            return None
        if not receiveds: 
            print(f"Invalid receiveds for {filename}")
            return None

        if ONLY_RATE:
            dash_features.append([np.mean(rates) + np.std(rates)])
        else:
            dash_features.append([
                np.mean(elapseds) + np.std(elapseds),
                np.mean(ticks) + np.std(ticks),
                np.mean(rates) + np.std(rates),
                np.mean(receiveds) + np.std(receiveds)
            ])
    return dash_features

def process_json_and_csv(json_file, csv_file):
    """
    Processa um único arquivo JSON e o respectivo arquivo result.csv, 
    extrai as 10 medições DASH mais recentes, calcula as features necessárias,
    e retorna uma linha para o CSV.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    client_id = data['cliente']
    server_id = data['servidor']

    # Verifica se os IDs são válidos e mapeia para valores numéricos
    if client_id not in CLIENTS:
        print(f"Erro: Cliente '{client_id}' não reconhecido em {json_file}. Arquivo ignorado.")
        return None
    if server_id not in SERVERS:
        print(f"Erro: Servidor '{server_id}' não reconhecido em {json_file}. Arquivo ignorado.")
        return None

    client_value = CLIENTS[client_id]
    server_value = SERVERS[server_id]

    dash_data = data['dash']
    dash_features = extract_dash_features(dash_data, json_file)

    if not dash_features:
        return None
    
    if len(dash_features) < 10:
        print(f"Erro: Menos de 10 medições DASH em {json_file}. Arquivo ignorado.")
        return None

    # Manter apenas as 10 medições mais recentes
    dash_features = dash_features[-10:]

    # Aplicar peso maior às medições mais recentes
    weights = np.linspace(1, WEIGHT_LAST_DATA, num=10)
    dash_features_weighted = [np.array(features) * weight for features, weight in zip(dash_features, weights)]
    dash_features_flat = np.hstack(dash_features_weighted).tolist()

    # Ler o arquivo result.csv correspondente
    result_data = pd.read_csv(csv_file)
    result_values = result_data.iloc[0, 1:].values.tolist()  # mean_1, stdev_1, mean_2, stdev_2

    # Preparar a linha para o CSV
    row = []
    if CONSIDER_CLIENT_SERVER:
        row = [client_value, server_value]
    row = row + dash_features_flat + result_values
    return row

def process_all_files(base_dir):
    """
    Itera sobre todos os arquivos JSON e result.csv no diretório de entrada e processa cada um,
    retornando uma lista de todas as linhas para o CSV.
    """
    all_data = []
    for client_dir in os.listdir(base_dir):
        client_path = os.path.join(base_dir, client_dir)
        for server_dir in os.listdir(client_path):
            server_path = os.path.join(client_path, server_dir)
            for timestamp_dir in os.listdir(server_path):
                path = os.path.join(server_path, timestamp_dir)
                json_file = os.path.join(path, "input.json")
                csv_file = os.path.join(path, "result.csv")

                if not os.path.isfile(json_file) or not os.path.isfile(csv_file):
                    continue

                row = process_json_and_csv(json_file, csv_file)
                if row is not None:
                    all_data.append(row)

    return all_data

def save_to_csv(all_data, output_csv):
    """
    Salva os dados processados em um arquivo CSV.
    """
    columns = []
    if CONSIDER_CLIENT_SERVER:
        columns = ['client_id', 'server_id']
    for i in range(10):
        if ONLY_RATE:
            columns.extend([f'dash{i}_rate'])
        else:
            columns.extend([
                f'dash{i}_elapsed',
                f'dash{i}_ticks',
                f'dash{i}_rate',
                f'dash{i}_received'
            ])
    # Adicionar as colunas dos resultados esperados
    columns.extend(['mean_1', 'stdev_1', 'mean_2', 'stdev_2'])

    data_df = pd.DataFrame(all_data, columns=columns)
    
    # Normalizar os dados (exceto client_id e server_id)
    if NORMALIZE:
        scaler = StandardScaler()
        data_df.iloc[:, 2:-4] = scaler.fit_transform(data_df.iloc[:, 2:-4])
    
    data_df.to_csv(output_csv, index=False)

def main():
    base_dir = os.path.join(os.getcwd(), "../converted_input")
    output_csv = "prepared_data.csv"

    all_data = process_all_files(base_dir)
    print(f"Total de amostras preparadas: {len(all_data)}")

    save_to_csv(all_data, output_csv)
    print(f"Dados de treinamento salvos em {output_csv}")

if __name__ == "__main__":
    main()
