import os
import json
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler

''' FLAGS '''

# Mapeamento de clientes e servidores para valores numéricos
CLIENTS = {"ba": 0, "rj": 1}
SERVERS = {"ce": 0, "df": 0.33, "es": 0.66, "pi": 1}

WEIGHT_LAST_DATA = 1 # Multiplicar valores mais recentes (deprecated)
EMA_ALPHA = 1  # Fator de suavização para o EMA (1 significa sem EMA)

USE_DIFF = True # Em vez de usar o valor bruto nas medições DASH, usar a diferença entre o atual e o anterior

CONSIDER_CLIENT_SERVER = True # Adicionar cliente e servidor como features

USE_ONLY_LAST_DASH_MEASURE = False # Usa apenas a ultima medicao DASH em vez das 10 ultimas
ONLY_RATE = True # Descartar todos os dados de DASH que nao sejam de Rate
INCLUDE_LAST_RATE_MEAN = True or RESULT_TO_DIFF_FROM_LAST
INCLUDE_LAST_RATE_STD = True or RESULT_TO_DIFF_FROM_LAST

ADD_TS = False
ADD_TS_SIN_COS = False

USE_ONLY_FIRST_AND_LAST_TR = False # Usa apenas a 1a e ultima medicao TRACEROUTE em vez das 5 ultimas

USE_TRACEROUTES = False # Usar dados de Traceroute
USE_RTT = False # Usar dados de RTT

ADD_MEAN_VALUES = True # Adiciona como features as médias dos valores de rate_mean e rate_std
RESULT_TO_DIFF_FROM_AVG = False # Considerar o que o modelo deve encontrar como a diferença em relação à média (NECESSÁRIO USAR ADD_MEAN_VALUES)
RESULT_TO_DIFF_FROM_LAST = True # Considerar o que o modelo deve encontrar como a diferença em relação à última medição

NORMALIZE = False # Aplicar normalização

EVALUATING = False # Ative isso quando apontar para dados de teste para gerar submissão

DISCARD_FILES_W_TR_OR_RTT = False and not EVALUATING # Descarta todos os arquivos que têm TraceRoute

DISCARD_FILES_WO_RTT = True and not EVALUATING # Descarta todos os arquivos que NÃO têm RTT
DISCARD_FILES_WO_TR = True and not EVALUATING # Descarta todos os arquivos que NÃO têm TraceRoute

PREDICT_ONLY_ONE_MEASURE = False

''' END OF FLAGS '''

def calc_diff(current, previous):
    return current - previous

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

    dash_data = dash_data[-10:]
    
    for dash in dash_data:
        rates = dash['rate']

        if not rates:
            print(f"Invalid data for {filename}")
            return None

        # Calcula as médias das medições atuais
        last_rate = np.mean(rates)
        last_rate_std = np.std(rates)

        current_features = [last_rate, last_rate_std]
        
        if not ONLY_RATE:
            elapseds = dash['elapsed']
            ticks = dash['request_ticks']
            receiveds = dash['received']
            current_features += [
                np.mean(elapseds),
                np.mean(ticks),
                np.mean(receiveds)
            ]

        if previous_ema is None:
            # Se for a primeira medição, inicializa o EMA com os valores atuais
            features = current_features
        else:
            features = [
                transform_data(current_features[i], previous_ema[i])
                for i in range(len(current_features))
            ]

        previous_ema = current_features
        dash_features.append(features)

    dash_features = dash_features[-10:]

    if USE_ONLY_LAST_DASH_MEASURE:
        dash_features = dash_features[-1:]

    if INCLUDE_LAST_RATE_MEAN:
        dash_features += [last_rate]

    if INCLUDE_LAST_RATE_STD:
        dash_features += [last_rate_std]

    return dash_features

def extract_rtt_features(rtt_data):
    """
    Extrai as features de RTT (Round-Trip Time) de um objeto RTT, retornando um array de arrays.
    Cada subarray contém dois elementos: rtt_mean e rtt_stdev.
    Se o número de medições for menor que 5, retorna None.
    """
    # Verifica se há pelo menos 5 medições
    if len(rtt_data) < 5:
        return None
    
    # Considera apenas as últimas 5 medições
    rtt_data = rtt_data[-5:]

    previous_feat = None
    rtt_features = []

    for rtt in rtt_data:
        rtt_values = []
        rtt_counts = []

        for value_str, count in rtt['val'].items():
            value = float(value_str)
            rtt_values.append(value)
            rtt_counts.append(count)

        # Calcula a média ponderada
        rtt_mean = np.average(rtt_values, weights=rtt_counts)
        # Calcula o desvio padrão ponderado
        rtt_stdev = np.sqrt(np.average((rtt_values - rtt_mean) ** 2, weights=rtt_counts))

        current_features = [rtt_mean, rtt_stdev]

        if previous_feat is None:
            # Se for a primeira medição, inicializa o EMA com os valores atuais
            features = current_features
        else:
            features = [
                transform_data(current_features[i], previous_feat[i])
                for i in range(len(current_features))
            ]

        previous_feat = current_features
        rtt_features.extend(features)

    rtt_features = rtt_features[-10:]
    if len(rtt_features) != 10:
        return None
    
    return rtt_features


def extract_tr_features(tr_data, filename):
    num_features = 11
    if not tr_data or (len(tr_data) < 5):
        return None if DISCARD_FILES_WO_TR else num_features * [0]
    elif DISCARD_FILES_W_TR_OR_RTT:
        return None
    
    tr_data = tr_data[-5:]
    if USE_ONLY_FIRST_AND_LAST_TR:
        tr_data = [tr_data[-5], tr_data[-1]]

    previous_feat = None
    tr_features = []

    jumps = []

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

            jumps.append(jump_count)

            rtt_sum = np.sum(rtt_values)
            rtt_std_dev = np.std(rtt_values)

            current_features = [rtt_sum, rtt_std_dev]
            
            if previous_feat is None:
                # Se for a primeira medição, inicializa
                features = current_features
            else:
                features = [
                    transform_data(current_features[i], previous_feat[i])
                    for i in range(len(current_features))
                ]

            previous_feat = current_features
            tr_features += features
        except:
            return None if DISCARD_FILES_WO_TR else num_features * [0]
    
    tr_features.append(np.std(jumps))
    return tr_features

def get_time_of_day_sin_cos(timestamp):
    time_struct = time.localtime(timestamp)

    hours = time_struct.tm_hour
    minutes = time_struct.tm_min

    total_minutes = hours * 60 + minutes

    # Calculate sine and cosine for the minutes of the day
    sin_value = np.sin(2 * np.pi * total_minutes / 1440)
    cos_value = np.cos(2 * np.pi * total_minutes / 1440)

    return sin_value, cos_value

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

    if ADD_TS:
        row.append(data['start_ts'])

    if ADD_TS_SIN_COS:
        time_sin, time_cos = get_time_of_day_sin_cos(data['start_ts'])
        row.append(time_sin)
        row.append(time_cos)

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
        # Combine client and server into a single feature
        # This ensures each (client, server) pair has a unique value
        combined_value = client_value + (server_value * 10)
        row.append(combined_value)  # Add the combined feature

    # Dash Features
    dash_data = data['dash']
    dash_features = extract_dash_features(dash_data, json_file)

    if not dash_features:
        return None
    
    if len(dash_features) < 10 and not USE_ONLY_LAST_DASH_MEASURE:
        print(f"Menos de 10 medições DASH em {json_file}. Arquivo ignorado.")
        return None
    if INCLUDE_LAST_RATE_MEAN:
        last_id = -2 if INCLUDE_LAST_RATE_STD else -1
        last_measure = dash_features[last_id]
        dash_features_flat = np.hstack(np.array(dash_features[:last_id])).tolist()
        dash_features_flat.append(last_measure)
        if INCLUDE_LAST_RATE_STD:
            last_measure_std = dash_features[-1]
            dash_features_flat.append(last_measure_std)
    else:
        dash_features_flat = np.hstack(np.array(dash_features)).tolist()

     # RTT Features
    rtt_features = extract_rtt_features(data['rtt'])
    if rtt_features is None:
        if DISCARD_FILES_WO_RTT:
            return None
        rtt_features = []
    else:
        if DISCARD_FILES_W_TR_OR_RTT:
            return None
        
    # Traceroute features
    traceroute_features = extract_tr_features(data['traceroute'], json_file)
    if traceroute_features is None:
        return None
    
    # Merge
    # Ordem: ID, Client, Server, Dashes, Avgs, TRs
    row = row + dash_features_flat

    if USE_RTT:
        row = row + rtt_features
    if USE_TRACEROUTES:
        row = row + traceroute_features

    if not EVALUATING:
        # Ler o arquivo result.csv e colocar resultados calculados
        result_data = pd.read_csv(csv_file)
        result_values = result_data.iloc[0, 1:].values.tolist()  # mean_1, stdev_1, mean_2, stdev_2

        row = row + result_values

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
    if ADD_TS:
        columns += ['start_ts']
    if ADD_TS_SIN_COS:
        columns += ['start_ts_sin']
        columns += ['start_ts_cos']
    if CONSIDER_CLIENT_SERVER:
        columns += ['client_server_id']
    
    if USE_ONLY_LAST_DASH_MEASURE:
        columns += ['dash_last_rate_mean', 'dash_last_rate_stdev']
    else:
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
    if INCLUDE_LAST_RATE_MEAN:
        columns.extend(['dash_last_rate'])
    if INCLUDE_LAST_RATE_STD:
        columns.extend(['dash_last_rate_std'])

    if USE_RTT:
        for i in range(5):
            columns.extend([
                f'rtt{i}_mean',
                f'rtt{i}_stdev',
            ])

    if USE_TRACEROUTES:
        if USE_ONLY_FIRST_AND_LAST_TR:
            columns.extend(
                [f'tr0_rtt_sum', f'tr0_rtt_stdev', f'tr4_rtt_sum', f'tr4_rtt_stdev']
            )
        else:
            for i in range(5):
                columns.extend([
                    f'tr{i}_rtt_sum',
                    f'tr{i}_rtt_stdev',
                ])
        columns.extend(['tr_jumps_std'])

    if not EVALUATING:
        # Adicionar as colunas dos resultados esperados
        columns.extend(['mean_1', 'stdev_1', 'mean_2', 'stdev_2'])

    return columns

def calculate_df_mean(df):
    if not USE_DIFF:
        return df.mean(axis=1)
    
    # Compute cumulative sum across the row
    adjusted_values = df.apply(lambda row: row.cumsum(), axis=1)
    
    # Calculate the mean of the adjusted values
    mean_values = adjusted_values.mean(axis=1)
    
    return mean_values


def save_to_csv(all_data, output_csv):
    """
    Salva os dados processados em um arquivo CSV.
    """
    try:
        data_df = pd.DataFrame(all_data, columns=build_columns()).dropna()
    except Exception as e:
        print(f"Columns: {build_columns()}")
        raise e

    if ADD_MEAN_VALUES:
        # Calcular a média das colunas de rate_mean e rate_stdev
        rate_means = [col for col in data_df.columns if 'dash' in col and 'rate_mean' in col]
        rate_stdevs = [col for col in data_df.columns if 'dash' in col and 'rate_stdev' in col]

        data_df['rates_mean'] = calculate_df_mean(data_df[rate_means])
        data_df['rates_stdev'] = calculate_df_mean(data_df[rate_stdevs])

        if not EVALUATING:
            # Reorganizar as colunas para mover 'mean_1', 'stdev_1', 'mean_2', 'stdev_2'
            # após 'rates_mean' e 'rates_stdev'
            ordered_columns = [col for col in data_df.columns if col not in ['mean_1', 'stdev_1', 'mean_2', 'stdev_2']]
            if PREDICT_ONLY_ONE_MEASURE:
                ordered_columns += ['mean_1', 'stdev_1']
            else:
                ordered_columns += ['mean_1', 'stdev_1', 'mean_2', 'stdev_2']

            data_df = data_df[ordered_columns]
            if RESULT_TO_DIFF_FROM_AVG:
                data_df['mean_1'] = data_df['mean_1'] - data_df['rates_mean']
                data_df['stdev_1'] = data_df['stdev_1'] - data_df['rates_stdev']
                if not PREDICT_ONLY_ONE_MEASURE:
                    data_df['mean_2'] = data_df['mean_2'] - data_df['rates_mean']
                    data_df['stdev_2'] = data_df['stdev_2'] - data_df['rates_stdev']
            elif RESULT_TO_DIFF_FROM_LAST:
                data_df['mean_1'] = data_df['mean_1'] - data_df['dash_last_rate']
                data_df['stdev_1'] = data_df['stdev_1'] - data_df['dash_last_rate_std']
                if not PREDICT_ONLY_ONE_MEASURE:
                    data_df['mean_2'] = data_df['mean_2'] - data_df['dash_last_rate']
                    data_df['stdev_2'] = data_df['stdev_2'] - data_df['dash_last_rate_std']


    # Normalizar os dados (exceto client_id e server_id)
    if NORMALIZE:
        scaler = StandardScaler()
        if RESULT_TO_DIFF_FROM_AVG:
            # We cannot normalize the avgs
            data_df.iloc[:, :-6] = scaler.fit_transform(data_df.iloc[:, :-6])
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
