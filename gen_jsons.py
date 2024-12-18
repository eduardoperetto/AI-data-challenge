import os
import json
import uuid
from datetime import datetime, timedelta
import csv
import shutil  # Import shutil to handle folder deletion

# Configurações iniciais
CLIENTS = ["ba", "rj"]
SERVERS = ["ce", "df", "es", "pi"]

INTERSECT_DASH_DATA = False
FRACTION_TO_INTERSECT = 3.5 # Quanto maior, mais vai intersectar, gerando mais inputs

def parse_filename_to_datetime(filename):
    """Extrai a data e hora do nome do arquivo."""
    date_str, time_str = filename.removesuffix(".jsonl").split('_')
    return datetime.strptime(date_str + time_str, "%y%m%d%H%M")

def load_jsonl_file(filepath):
    """Carrega e retorna todos os registros de um arquivo JSONL."""
    with open(filepath, 'r') as file:
        return [json.loads(line) for line in file]

def load_json_file(filepath):
    """Carrega e retorna o conteúdo de um arquivo JSON."""
    with open(filepath, 'r') as file:
        return json.load(file)

def extract_measures(filepaths, base_path):
    """Extrai as medidas DASH a partir de uma lista de arquivos JSONL."""
    measures = []
    for file in filepaths:
        filepath = os.path.join(base_path, file)
        dash_data = load_jsonl_file(filepath)
        measures.append({
            "elapsed": [req["elapsed"] for req in dash_data[:-1]],
            "request_ticks": [req["request_ticks"] for req in dash_data[:-1]],
            "rate": [req["rate"] for req in dash_data[:-1]],
            "received": [req["received"] for req in dash_data[:-1]],
            "timestamp": [req["timestamp"] for req in dash_data[:-1]]
        })
    return measures

def filter_rtt_traceroute(data, start_ts, end_ts):
    """Filtra as medições de RTT ou traceroute para estar dentro do intervalo de tempo."""
    return [measure for measure in data if start_ts <= measure["ts"] <= end_ts]

def calculate_statistics(measures):
    """Calcula mean e std dev das duas últimas medições."""
    last_2 = measures[-2:]
    mean_1, stdev_1 = calculate_mean_and_stdev(last_2[0]["rate"])
    mean_2, stdev_2 = calculate_mean_and_stdev(last_2[1]["rate"])

    if not (mean_1 and stdev_1 and mean_2 and stdev_2):
        return None

    return [
        [mean_1, stdev_1],
        [mean_2, stdev_2]
    ]

def calculate_mean_and_stdev(values):
    """Calcula a média e o desvio padrão de uma lista de valores."""
    try:
        mean = sum(values) / len(values)
        stdev = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
        return mean, stdev
    except:
        return None, None

def generate_json_and_csv(client, server, measures_dash, measures_rtt, measures_traceroute, statistics, output_folder):
    """Gera os arquivos JSON e CSV com os resultados."""
    os.makedirs(output_folder, exist_ok=True)
    
    # Gerar arquivo JSON
    json_data = {
        "id": str(uuid.uuid4()),
        "cliente": client,
        "servidor": server,
        "dash": measures_dash,
        "rtt": measures_rtt,
        "traceroute": measures_traceroute
    }
    
    json_file_path = os.path.join(output_folder, 'input.json')
    with open(json_file_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)
    
    # Gerar arquivo CSV
    csv_file_path = os.path.join(output_folder, 'result.csv')
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "mean_1", "stdev_1", "mean_2", "stdev_2"])
        writer.writerow([json_data["id"], *statistics[0], *statistics[1]])
    global num_files_gen
    num_files_gen += 1

def process_files(client, server, base_dash_path, rtt_data, traceroute_data):
    """Processa todos os arquivos de um par client-server."""
    dash_folder = os.path.join(base_dash_path, client, server)
    all_files = sorted(os.listdir(dash_folder))
    
    time_0 = parse_filename_to_datetime(all_files[0])
    
    while all_files:
        current_files = collect_files_for_time_window(all_files, time_0, timedelta(hours=1))
        if len(current_files) < 3:
            break
        
        time_f = parse_filename_to_datetime(current_files[-1])
        measures_dash = extract_measures(current_files, dash_folder)
        
        last_2_measures = measures_dash[-2:]
        measures_dash = measures_dash[:-2]

        start_ts, end_ts = time_0.timestamp(), time_f.timestamp()

        measures_rtt = filter_rtt_traceroute(rtt_data, start_ts, end_ts)
        measures_traceroute = filter_rtt_traceroute(traceroute_data, start_ts, end_ts)

        statistics = calculate_statistics(last_2_measures)

        if statistics is not None:
            if len(measures_rtt) == 0 or len(measures_traceroute) == 0:
                global files_wo_rtt_tr
                files_wo_rtt_tr += 1

            output_folder = os.path.join(os.getcwd(), "converted_input", f"{client}", f"{server}", f"{time_0.strftime('%Y%m%d_%H%M')}")
            generate_json_and_csv(client, server, measures_dash, measures_rtt, measures_traceroute, statistics, output_folder)
        
        if all_files:
            if parse_filename_to_datetime(all_files[0]) != time_0:
                time_0 = parse_filename_to_datetime(all_files[0])
            else:
                all_files.remove(all_files[0])
                time_0 = parse_filename_to_datetime(all_files[0])

def collect_files_for_time_window(all_files, start_time, time_window):
    """Coleta arquivos dentro de uma janela de tempo."""
    print(f"Collecting dash files for {start_time}")
    print(f"First file is {all_files[0]}")
    print(f"{len(all_files)} files remaining")

    if len(all_files) <= 5:
        current_files = all_files
        for file in all_files:
            all_files.remove(file)
        return current_files

    current_files = []
    for filename in all_files:
        file_datetime = parse_filename_to_datetime(filename)
        if file_datetime > start_time + time_window:
            break
        current_files.append(filename)
    
    if INTERSECT_DASH_DATA:
        range_to_del = int(len(current_files)/FRACTION_TO_INTERSECT)
    else:
        range_to_del = int(len(current_files))

    if range_to_del < 1:
        range_to_del = 1

    for i in range(range_to_del):
        if len(current_files) > 5:
            all_files.remove(current_files[i])
    
    return current_files

def main():
    global files_wo_rtt_tr, num_files_gen
    files_wo_rtt_tr = 0
    num_files_gen = 0

    base_dash_path = "./dataset/Train/dash"
    rtt_path = "./dataset/Train/rtt"
    traceroute_path = "./dataset/Train/traceroute"

    # Remove the converted_input folder if it exists
    converted_input_path = os.path.join(os.getcwd(), "converted_input")
    if os.path.exists(converted_input_path):
        print("Removed old converted_inputs folder")
        shutil.rmtree(converted_input_path)

    for client in CLIENTS:
        for server in SERVERS:
            rtt_file = os.path.join(rtt_path, client, f"measure-rtt_ref-{client}_pop-{server}.json")
            traceroute_file = os.path.join(traceroute_path, client, f"measure-traceroute_ref-{client}_pop-{server}.json")
            rtt_data = load_json_file(rtt_file)
            traceroute_data = load_json_file(traceroute_file)

            process_files(client, server, base_dash_path, rtt_data, traceroute_data)

    print(f"Generated successfully on {converted_input_path}")
    print(f"Num files generated: {num_files_gen}")
    print(f"Num files without RTT or TR: {files_wo_rtt_tr}")

if __name__ == "__main__":
    main()
