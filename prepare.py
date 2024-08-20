import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Mapeamento de clientes e servidores para valores numéricos
CLIENTS = {"ba": 0, "rj": 1}
SERVERS = {"ce": 0, "df": 0.33, "es": 0.66, "pi": 1}

CONSIDER_CLIENT_SERVER = True # Adicionar cliente e servidor como features
WEIGHT_LAST_DATA = 1 # Multiplicar valores mais recentes (deprecated)
NORMALIZE = False # Aplicar normalizaçao
ONLY_RATE = True # Descartar todos os dados de DASH que nao sejamos de Rate
USE_TRACEROUTES = True # Usar dados de Traceroute
DISCARD_FILES_WO_TR = False # Descarta todos os arquivos que NÃO tem TraceRoute
DISCARD_FILES_W_TR = False # Descarta todos os arquivos que TEM TraceRoute
USE_DIFF = False # Em vez de usar o valor bruto nas medicoes DASH, usar a diferença entre o atual e o anterior
EMA_ALPHA = 1  # Fator de suavização para o EMA (1 significa sem EMA)
ADD_MEAN_VALUES = True # Adiciona como features as média dos valores de rate_mean e rate_std
RESULT_TO_DIFF_FROM_AVG = False # Considerar o que o modelo deve encontrar como a diferenca em relacao a media (NECESSARIO USAR ADD_MEAN_VALUES)

EVALUATING = False # Ative isso quando apontar pra dados de teste pra gerar submissao

all_rates_mean = []
all_rates_std = []

def calc_diff(current, previous):
    return current-previous

def calculate_ema_difference(current, previous, alpha=EMA_ALPHA):
    """
    Calcula a diferença EMA entre a medida atual e a anterior.
    """
    return alpha * (current - previous) + previous

def transform_data(current, previous):
    if USE_DIFF:
        return calc_diff(current, previous)
    else:
        return calculate_ema_difference(current, previous)

def extract_dash_features(dash_data, filename):
    """
    Extrai as features combinadas de elapsed, request_ticks, rate e received
    de cada medição DASH, retornando uma lista de listas onde cada sublista contém
    as 4 features combinadas utilizando a diferença EMA entre medições consecutivas.
    """
    dash_features = []

    previous_ema = None  # Inicializar a variável para armazenar a EMA anterior

    current_rates_mean = []
    current_rates_std = []
    
    for dash in dash_data:
        elapseds = dash['elapsed']
        ticks = dash['request_ticks']
        rates = dash['rate']
        receiveds = dash['received']

        if not elapseds or not ticks or not rates or not receiveds:
            print(f"Invalid data for {filename}")
            return None

        # Calcula as médias das medições atuais
        current_features = [np.mean(rates), np.std(rates),]

        rate_mean = np.mean(rates)
        rate_std = np.std(rates)

        if not ONLY_RATE:
            current_features += [
                rate_mean,
                rate_std,
                np.mean(elapseds),
                np.mean(ticks),
                np.mean(receiveds)
            ]

        # Isso aqui nao é a media das medias. Vai se tornar a media de todas as medidas, o que é diferente
        current_rates_mean.append(rate_mean)
        current_rates_std.append(rate_std)

        if previous_ema is None:
            # Se for a primeira medição, inicializa o EMA com os valores atuais
            features = current_features
        else:
            features = [
                transform_data(current_features[i], previous_ema[i])
                for i in range(len(current_features))
            ]
        
        dash_features.append(features)
        previous_ema = features  # Atualiza o EMA anterior para a próxima iteração
    
    current_rates_mean = current_rates_mean[-10:]
    current_rates_std = current_rates_std[-10:]
    all_rates_mean.append(np.mean(current_rates_mean))
    all_rates_std.append(np.mean(current_rates_std))

    dash_features = dash_features[-10:]
    return dash_features

def extract_tr_features(tr_data, filename):
    if not USE_TRACEROUTES:
        return []
    if not tr_data or (len(tr_data) < 5):
        return None if DISCARD_FILES_WO_TR else 15 * [0]
    elif DISCARD_FILES_W_TR:
        return None
    
    tr_data = tr_data[-5:]
    tr_features = []

    for measure in tr_data:
        try:
            jump_count = len(measure['val'])
            rtt_values = []
            for hop in measure['val']:
                try:
                    if hop['rtt']:
                        rtt_values.append(hop['rtt'])
                except:
                    continue
            rtt_mean = np.mean(rtt_values)
            rtt_std_dev = np.std(rtt_values)
            tr_features += [jump_count,rtt_mean,rtt_std_dev]
        except:
            return None if DISCARD_FILES_WO_TR else 15 * [0]
    
    return tr_features

def process_json_and_csv(json_file, csv_file):
    """
    Processa um único arquivo JSON e o respectivo arquivo result.csv, 
    extrai as 10 medições DASH mais recentes, calcula as features necessárias,
    e retorna uma linha para o CSV.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    row = []

    if EVALUATING:
        row.append(data['id'])

    if CONSIDER_CLIENT_SERVER:
        client_id = data['cliente']
        server_id = data['servidor']

        # Verifica se os IDs são válidos e mapeia para valores numéricos
        if client_id not in CLIENTS:
            print(f"Cliente '{client_id}' não reconhecido em {json_file}. Arquivo ignorado.")
            return None
        if server_id not in SERVERS:
            print(f"Servidor '{server_id}' não reconhecido em {json_file}. Arquivo ignorado.")
            return None

        client_value = CLIENTS[client_id]
        server_value = SERVERS[server_id]
        row += [client_value, server_value]

    # Dash Features
    dash_data = data['dash']
    dash_features = extract_dash_features(dash_data, json_file)

    if not dash_features:
        return None
    
    if len(dash_features) < 10:
        print(f"Menos de 10 medições DASH em {json_file}. Arquivo ignorado.")
        return None

    # Aplicar peso maior às medições mais recentes (opcional)
    weights = np.linspace(1, WEIGHT_LAST_DATA, num=10)
    dash_features_weighted = [np.array(features) * weight for features, weight in zip(dash_features, weights)]
    dash_features_flat = np.hstack(dash_features_weighted).tolist()

    # Traceroute features
    traceroute_features = extract_tr_features(data['traceroute'], json_file)
    if traceroute_features is None:
        return None

    result_values = []

    rate_avgs=[]
    if ADD_MEAN_VALUES:
        mean_from_means = sum(all_rates_mean) / len(all_rates_mean)
        mean_from_stds = sum(all_rates_std) / len(all_rates_std)
        rate_avgs = [mean_from_means, mean_from_stds]

    if not EVALUATING:
        # Ler o arquivo result.csv e colocar resultados calculados
        result_data = pd.read_csv(csv_file)
        result_values = result_data.iloc[0, 1:].values.tolist()  # mean_1, stdev_1, mean_2, stdev_2
        if RESULT_TO_DIFF_FROM_AVG:
            result_values = [value - avg for value, avg in zip(result_values, 2 * rate_avgs)]

    # Merge
    # Ordem: ID, Client, Server, Dashes, Avgs, TRs, Results
    row = row + dash_features_flat + rate_avgs + traceroute_features + result_values
    return row

def process_all_files(base_dir):
    """
    Itera sobre todos os arquivos JSON e result.csv no diretório de entrada e processa cada um,
    retornando uma lista de todas as linhas para o CSV.
    """
    all_data = []

    if EVALUATING:
        for json_file in os.listdir(base_dir):
            path = os.path.join(base_dir, json_file)
            row = process_json_and_csv(path, None)
            if row is not None:
                all_data.append(row)
        return all_data

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

def build_columns():
    columns = []
    if EVALUATING:
        columns += ['id']
    if CONSIDER_CLIENT_SERVER:
        columns += ['client_id', 'server_id']
    for i in range(10):
        if ONLY_RATE:
            columns.extend([f'dash{i}_rate_mean', f'dash{i}_rate_stdev',])
        else:
            columns.extend([
                f'dash{i}_rate_mean',
                f'dash{i}_rate_stdev',
                f'dash{i}_elapsed',
                f'dash{i}_ticks',
                f'dash{i}_received'
            ])

    if ADD_MEAN_VALUES:
        columns += ["rates_mean", "rates_stdev"]

    if USE_TRACEROUTES:
        for i in range(5):
            columns.extend([
                f'tr{i}_jump_count',
                f'tr{i}_rtt_mean',
                f'tr{i}_rtt_stdev',
            ])

    if not EVALUATING:
        # Adicionar as colunas dos resultados esperados
        columns.extend(['mean_1', 'stdev_1', 'mean_2', 'stdev_2'])

    return columns

def save_to_csv(all_data, output_csv):
    """
    Salva os dados processados em um arquivo CSV.
    """
    try:
        data_df = pd.DataFrame(all_data, columns=build_columns())
    except Exception as e:
        print(f"Columns: {build_columns()}")
        raise e

    # Normalizar os dados (exceto client_id e server_id)
    if NORMALIZE:
        scaler = StandardScaler()
        if CONSIDER_CLIENT_SERVER:
            data_df.iloc[:, 2:-4] = scaler.fit_transform(data_df.iloc[:, 2:-4])
        else:
            data_df.iloc[:, :-4] = scaler.fit_transform(data_df.iloc[:, :-4])
    
    data_df.to_csv(output_csv, index=False)

def main():
    if EVALUATING:
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./dataset/Test")
    else:
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./converted_input")
    output_csv = "prepared_data.csv"

    all_data = process_all_files(base_dir)
    print(f"Total de amostras preparadas: {len(all_data)}")

    save_to_csv(all_data, output_csv)
    print(f"Dados de treinamento salvos em {output_csv}")

if __name__ == "__main__":
    main()
