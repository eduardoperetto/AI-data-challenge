import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def read_jsonl(file_path):
    """Lê um arquivo .jsonl e retorna uma lista de dicionários."""
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def read_json(file_path):
    """Lê um arquivo .json e retorna o conteúdo como um dicionário."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def extract_dash_features(dash_data, file_path):
    """Extrai as features 'mean_rate', 'std_rate' e outras variáveis das medições DASH."""
    # As primeiras 15 linhas contêm os dados das requisições individuais
    requests_data = dash_data[:-1]  # Todas as linhas exceto a última
    summary_data = dash_data[-1]    # Última linha contém o resumo

    rates = []
    connect_times = []
    elapsed_times = []
    received_data = []

    for req in requests_data:
        rates.append(req['rate'])
        connect_times.append(req['connect_time'])
        elapsed_times.append(req['elapsed'])
        received_data.append(req['received'])

    # Extraindo features a partir dos dados das requisições individuais
    features = {
        'mean_rate': np.mean(rates),
        'std_rate': np.std(rates),
        'mean_connect_time': np.mean(connect_times),
        'std_connect_time': np.std(connect_times),
        'mean_elapsed_time': np.mean(elapsed_times),
        'std_elapsed_time': np.std(elapsed_times),
        'total_received': np.sum(received_data),
    }

    # Adicionando features a partir do resumo final
    try:
        ticks = [item['ticks'] for item in summary_data]
        summary_features = {
            'mean_ticks': np.mean(ticks),
            'std_ticks': np.std(ticks),
            'max_ticks': np.max(ticks),
            'min_ticks': np.min(ticks),
        }
        # Combinando ambas as partes de features
        features.update(summary_features)
    except:
        print(f"Arquivo com defeito: {file_path}")
    
    return features

def process_dash_data(root_dir):
    """Processa todos os arquivos .jsonl de DASH, extraindo as features."""
    all_features = []
    # raise Exception(root_dir)
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(subdir, file)
                dash_data = read_jsonl(file_path)
                features = extract_dash_features(dash_data, file_path)
                all_features.append(features)
    return pd.DataFrame(all_features)

# def extract_rtt_features(rtt_data):
#     """Extrai features relevantes dos dados de RTT."""
#     features = []
#     for measurement in rtt_data:
#         values = [float(k) * v for k, v in measurement['val'].items()]
#         rtt_mean = np.mean(values)
#         rtt_std = np.std(values)
#         features.append({
#             'rtt_mean': rtt_mean,
#             'rtt_std': rtt_std,
#         })
#     return features

# def process_rtt_data(root_dir):
#     """Processa todos os arquivos .json de RTT, extraindo as features."""
#     all_features = []
#     for subdir, _, files in os.walk(root_dir):
#         for file in files:
#             if file.endswith(".json"):
#                 file_path = os.path.join(subdir, file)
#                 rtt_data = read_json(file_path)
#                 features = extract_rtt_features(rtt_data)
#                 all_features.extend(features)
#     return pd.DataFrame(all_features)

# def extract_traceroute_features(traceroute_data):
#     """Extrai features relevantes dos dados de Traceroute."""
#     features = []
#     for measurement in traceroute_data:
#         hops = len(measurement['val'])
#         rtt_values = [hop['rtt'] for hop in measurement['val'] if 'rtt' in hop]
#         mean_rtt = np.mean(rtt_values) if rtt_values else 0
#         std_rtt = np.std(rtt_values) if rtt_values else 0
#         features.append({
#             'hops': hops,
#             'mean_rtt': mean_rtt,
#             'std_rtt': std_rtt,
#         })
#     return features

# def process_traceroute_data(root_dir):
#     """Processa todos os arquivos .json de Traceroute, extraindo as features."""
#     all_features = []
#     for subdir, _, files in os.walk(root_dir):
#         for file in files:
#             if file.endswith(".json"):
#                 file_path = os.path.join(subdir, file)
#                 traceroute_data = read_json(file_path)
#                 features = extract_traceroute_features(traceroute_data)
#                 all_features.extend(features)
#     return pd.DataFrame(all_features)

# def combine_features(dash_df, rtt_df, traceroute_df):
#     """Combina as features de DASH, RTT e Traceroute em um único DataFrame."""
#     combined_df = pd.concat([dash_df, rtt_df, traceroute_df], axis=1)
#     combined_df = combined_df.fillna(0)  # Preenche valores faltantes com 0
#     return combined_df

def normalize_data(df):
    """Normaliza os dados usando StandardScaler."""
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

def save_to_csv(df, filename):
    """Salva o DataFrame em um arquivo .csv."""
    df.to_csv(filename, index=False)

def main():
    """Função principal que coordena o fluxo de execução do script."""
    cwd = os.getcwd()

    # Processar dados de treinamento
    dash_df = process_dash_data(f"{cwd}/../dataset/Train/dash")
    # rtt_df = process_rtt_data("../dataset/Train/rtt")
    # traceroute_df = process_traceroute_data("../dataset/Train/traceroute")

    if dash_df.empty: # or rtt_df.empty or traceroute_df.empty:
        raise ValueError("Nenhuma feature foi extraída. Verifique o formato dos dados de entrada.")

    # combined_df = combine_features(dash_df, rtt_df, traceroute_df)
    # combined_df = normalize_data(combined_df)

    normalized_df = normalize_data(dash_df)

    save_to_csv(normalized_df, f"{cwd}/train_data.csv")

if __name__ == "__main__":
    main()
    