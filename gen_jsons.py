import os
import json
import uuid
from datetime import datetime, timedelta
import csv
import shutil

CLIENTS = ["ba", "rj"]
SERVERS = ["ce", "df", "es", "pi"]

INTERSECT_DASH_DATA = True
FRACTION_TO_INTERSECT = 2

def parse_filename_to_datetime(filename):
    return datetime.strptime(filename.removesuffix(".jsonl").replace('_',''), "%y%m%d%H%M")

def load_jsonl_file(filepath):
    try:
        with open(filepath, 'r') as file:
            return [json.loads(line) for line in file]
    except Exception as e:
        print(f"Error loading JSONL file {filepath}: {e}")
        return []

def load_json_file(filepath):
    try:
        with open(filepath, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading JSON file {filepath}: {e}")
        return []

def extract_measures(filepaths, base_path):
    print(f"Extracting measures from {len(filepaths)} files under {base_path}")
    measures = []
    for file in filepaths:
        filepath = os.path.join(base_path, file)
        dash_data = load_jsonl_file(filepath)
        if len(dash_data) == 0:
            print(f"No dash data found in {file}.")
        measures.append({
            "elapsed": [req["elapsed"] for req in dash_data[:-1]],
            "request_ticks": [req["request_ticks"] for req in dash_data[:-1]],
            "rate": [req["rate"] for req in dash_data[:-1]],
            "received": [req["received"] for req in dash_data[:-1]],
            "timestamp": [req["timestamp"] for req in dash_data[:-1]]
        })
    return measures

def filter_rtt_traceroute(data, start_ts, end_ts):
    filtered = []
    for measure in data:
        if start_ts <= measure["ts"] <= end_ts:
            filtered.append(measure)
    print(f"Filtered {len(filtered)} measurements between {start_ts} and {end_ts}")
    return filtered

def calculate_statistics(measures):
    print("Calculating statistics.")
    last_2 = measures[-2:]
    mean_1, stdev_1 = calculate_mean_and_stdev(last_2[0]["rate"])
    mean_2, stdev_2 = calculate_mean_and_stdev(last_2[1]["rate"])
    if not (mean_1 and stdev_1 and mean_2 and stdev_2):
        print("Could not calculate statistics due to missing data.")
        return None
    print("Calculated statistics successfully.")
    return [
        [mean_1, stdev_1],
        [mean_2, stdev_2]
    ]

def calculate_mean_and_stdev(values):
    try:
        mean = sum(values) / len(values)
        stdev = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
        return mean, stdev
    except Exception as e:
        print(f"Error calculating mean and stdev: {e}")
        return None, None

def generate_json_and_csv(timestamp, client, server, measures_dash, measures_rtt, measures_traceroute, statistics, output_folder):
    print(f"Generating JSON and CSV for {client}-{server} in {output_folder}")
    os.makedirs(output_folder, exist_ok=True)
    json_data = {
        "id": str(uuid.uuid4()),
        "start_ts": timestamp,
        "cliente": client,
        "servidor": server,
        "dash": measures_dash,
        "rtt": measures_rtt,
        "traceroute": measures_traceroute
    }
    json_file_path = os.path.join(output_folder, 'input.json')
    try:
        with open(json_file_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)
    except Exception as e:
        print(f"Error writing JSON file: {e}")
    csv_file_path = os.path.join(output_folder, 'result.csv')
    try:
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["id", "mean_1", "stdev_1", "mean_2", "stdev_2"])
            writer.writerow([json_data["id"], *statistics[0], *statistics[1]])
    except Exception as e:
        print(f"Error writing CSV file: {e}")
    global num_files_gen
    num_files_gen += 1
    print(f"Files generated successfully for {client}-{server} at {output_folder}")

def process_files(client, server, base_dash_path, rtt_data, traceroute_data):
    dash_folder = os.path.join(base_dash_path, client, server)
    if not os.path.exists(dash_folder):
        print(f"Dash folder does not exist: {dash_folder}")
        return
    all_files = sorted(os.listdir(dash_folder))
    if not all_files:
        print(f"No files found for {client}-{server} in {dash_folder}")
        return
    time_0 = parse_filename_to_datetime(all_files[0])
    
    print(f"Processing files for {client}-{server}, starting at {time_0}, total {len(all_files)} files.")
    while all_files:
        if time_0 < datetime(2024, 6, 7): # Discard files before 2024-06-07
            all_files.remove(all_files[0])
            time_0 = parse_filename_to_datetime(all_files[0])
            print(f"Discarded file from {time_0}")
            continue
        current_files = collect_files_for_time_window(all_files, time_0, timedelta(hours=1))
        print(f"Collected {len(current_files)} files for timeslot starting at {time_0}")
        if len(current_files) < 3:
            print(f"Not enough files in timeslot starting at {time_0}")
            if (len(all_files) > 3):
                time_0 = parse_filename_to_datetime(all_files[0])
                continue
            else:
                break
        try:
            time_f = parse_filename_to_datetime(current_files[-1])
        except Exception as e:
            print(f"Error parsing filename to datetime: {e}")
            exit(-1)
        print(f"Current timeslot: {time_0} to {time_f}")
        measures_dash = extract_measures(current_files, dash_folder)
        if len(measures_dash) < 3:
            print("Not enough measures extracted.")
            continue
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
                print("No RTT or traceroute data found for this timeslot.")
            output_folder = os.path.join(os.getcwd(), "converted_input", f"{client}", f"{server}", f"{time_0.strftime('%Y%m%d_%H%M')}")
            generate_json_and_csv(start_ts, client, server, measures_dash, measures_rtt, measures_traceroute, statistics, output_folder)
        if all_files:
            if parse_filename_to_datetime(all_files[0]) != time_0:
                time_0 = parse_filename_to_datetime(all_files[0])
            else:
                all_files.remove(all_files[0])
                time_0 = parse_filename_to_datetime(all_files[0])

def collect_files_for_time_window(all_files, start_time, time_window):
    print(f"Collecting dash files for timeslot starting at {start_time}")
    print(f"First file in all_files: {all_files[0] if all_files else 'None'}")
    print(f"{len(all_files)} files currently in queue")
    if len(all_files) <= 5:
        current_files = all_files[:]
        for file in all_files[:]:
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
        print(f"Error: Range to del in timeslot starting at {start_time} is {range_to_del}")
        range_to_del = 1
        # exit(-1)
    print(f"Range to del in timeslot starting at {start_time} is {range_to_del}")
    for i in range(range_to_del):
        # if len(current_files) > 5:
        if current_files[i] in all_files:
            all_files.remove(current_files[i])
    return current_files

def main():
    global files_wo_rtt_tr, num_files_gen
    files_wo_rtt_tr = 0
    num_files_gen = 0
    base_dash_path = "./dataset/Train/dash"
    rtt_path = "./dataset/Train/rtt"
    traceroute_path = "./dataset/Train/traceroute"
    converted_input_path = os.path.join(os.getcwd(), "converted_input")
    if os.path.exists(converted_input_path):
        print("Removed old converted_inputs folder")
        shutil.rmtree(converted_input_path)
    for client in CLIENTS:
        for server in SERVERS:
            print(f"Processing client-server pair: {client}-{server}")
            rtt_file = os.path.join(rtt_path, client, f"measure-rtt_ref-{client}_pop-{server}.json")
            traceroute_file = os.path.join(traceroute_path, client, f"measure-traceroute_ref-{client}_pop-{server}.json")
            rtt_data = load_json_file(rtt_file)
            traceroute_data = load_json_file(traceroute_file)
            if rtt_data is None or traceroute_data is None:
                print(f"Skipping {client}-{server} due to missing RTT or traceroute data.")
                continue
            process_files(client, server, base_dash_path, rtt_data, traceroute_data)
    print(f"Generated successfully on {converted_input_path}")
    print(f"Num files generated: {num_files_gen}")
    print(f"Num files without RTT or TR: {files_wo_rtt_tr}")

if __name__ == "__main__":
    main()
