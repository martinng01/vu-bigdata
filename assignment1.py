import threading
import pandas as pd
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
from haversine import haversine
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import psutil

MAX_DISTANCE_KM = 100
# https://sahyogfreight.com/blog/cargo-ship-speed/#:~:text=The%20optimal%20speed%20of%20a,travel%20at%20around%2015%20knots.
# 24 knots maximum speed = 44.4kmh
MAX_SPEED_KMH = 44.4


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def chunk_data(df: pd.DataFrame, chunk_size: int) -> list[pd.DataFrame]:
    num_chunks = len(df) // chunk_size
    return np.array_split(df, num_chunks)


def preprocess_data(df: pd.DataFrame) -> None:
    # Drop rows with missing/invalid lat, lon values
    df = df.dropna(subset=["Latitude", "Longitude"])
    df = df[(df["Latitude"].between(-90, 90)) &
            (df["Longitude"].between(-180, 180))]

    # Convert timestamp to datetime format
    df['# Timestamp'] = pd.to_datetime(df['# Timestamp'])
    return df


def analyse_loc_anomalies(df: pd.DataFrame) -> list[pd.DataFrame]:
    df = df.copy()

    df['Prev_Latitude'] = df.groupby('MMSI')['Latitude'].shift(1)
    df['Prev_Longitude'] = df.groupby('MMSI')['Longitude'].shift(1)
    df['Distance_km'] = df.apply(
        lambda row: haversine(
            (row['Prev_Latitude'], row['Prev_Longitude']),
            (row['Latitude'], row['Longitude'])
        ), axis=1
    )

    df['Location_Anomaly'] = df['Distance_km'] > MAX_DISTANCE_KM

    return df[df['Location_Anomaly']]


def detect_speed_anomalies(df: pd.DataFrame) -> list[pd.DataFrame]:
    df = df.copy()
    df = df.sort_values(by=['MMSI', '# Timestamp'])

    # Compute previous row values using shift
    df['Prev_Latitude'] = df.groupby('MMSI')['Latitude'].shift(1)
    df['Prev_Longitude'] = df.groupby('MMSI')['Longitude'].shift(1)
    df['Prev_Timestamp'] = df.groupby('MMSI')['# Timestamp'].shift(1)

    # Drop first entry for each MMSI
    df = df.dropna(
        subset=['Prev_Latitude', 'Prev_Longitude', 'Prev_Timestamp'])

    df['Distance_km'] = df.apply(
        lambda row: haversine((row['Prev_Latitude'], row['Prev_Longitude']),
                              (row['Latitude'], row['Longitude'])), axis=1
    )

    df['Time_hr'] = (df['# Timestamp'] - df['Prev_Timestamp']
                     ).dt.total_seconds() / 3600.0

    df['Speed_kmh'] = df['Distance_km'] / df['Time_hr']
    df['Speed_kmh'] = df['Speed_kmh'].fillna(
        0)  # Replace NaNs from division by zero

    df['Anomaly'] = df['Speed_kmh'] > 44.4

    return df[df['Anomaly']]


def compare_neighbouring_vessels(df: pd.DataFrame) -> list[pd.DataFrame]:
    df = df.copy()
    df['Anomaly'] = False
    time_window_min = 5
    threshold_distance_km = 2
    anomalies = []

    df = df.sort_values(by=['# Timestamp', 'MMSI'])

    for i, vessel1 in df.iterrows():
        time_window_start = vessel1['# Timestamp'] - \
            pd.Timedelta(minutes=time_window_min)
        time_window_end = vessel1['# Timestamp'] + \
            pd.Timedelta(minutes=time_window_min)

        # Find vessels within the same time window
        nearby_vessels_mask = (df['# Timestamp'] >= time_window_start) & (
            df['# Timestamp'] <= time_window_end)
        nearby_vessels = df[nearby_vessels_mask]

        # Compare vessel1 with other vessels in the same time window
        for j, vessel2 in nearby_vessels.iterrows():
            if vessel1['MMSI'] == vessel2['MMSI']:  # Skip same vessel
                continue

            distance = haversine((vessel1['Latitude'], vessel1['Longitude']),
                                 (vessel2['Latitude'], vessel2['Longitude']))

            if distance < threshold_distance_km:
                anomalies.append(i)
                anomalies.append(j)

    anomalies = list(set(anomalies))
    df.loc[anomalies, 'Anomaly'] = True

    return df[df['Anomaly']]


def detect_gps_spoofing(df: pd.DataFrame) -> list[pd.DataFrame]:
    loc_anomalies = analyse_loc_anomalies(df)
    speed_anomalies = detect_speed_anomalies(df)
    # neighbouring_vessel_anomalies = compare_neighbouring_vessels(df)
    # print(
    #     f"Neighbouring vessel anomalies: {len(neighbouring_vessel_anomalies)}")

    # all_anomalies = pd.concat(
    #     [loc_anomalies, speed_anomalies, neighbouring_vessel_anomalies])
    # all_anomalies = all_anomalies.drop_duplicates(
    #     subset=['MMSI', '# Timestamp'])

    # return len(all_anomalies)


def monitor(func, *args, **kwargs):
    cpu_usages = []
    mem_usages = []
    done = [False]

    def watch_cpu_mem():
        while not done[0]:
            cpu_usages.append(psutil.cpu_percent(
                interval=0.1, percpu=True))
            mem_usages.append(psutil.virtual_memory().percent)

    monitor_thread = threading.Thread(target=watch_cpu_mem)
    monitor_thread.start()

    result = func(*args, **kwargs)

    done[0] = True
    monitor_thread.join()

    avg_cpu = np.mean(cpu_usages)
    avg_mem = np.mean(mem_usages)

    return result, avg_cpu, avg_mem


if __name__ == '__main__':
    df = load_data('data/aisdk-2024-09-11.csv')

    # Downsample for quicker processing
    df = df.sample(n=10_000_000, random_state=42)
    df = preprocess_data(df)

    chunk_sizes = [200_000, 300_000, 400_000]
    worker_counts = [4, 5, 6, 7, 8]

    results = []

    for chunk_size in chunk_sizes:
        chunks = chunk_data(df, chunk_size)

        seq_time_start = time.time()
        for chunk in tqdm(chunks, desc=f"Processing chunk size {chunk_size} (Sequential)"):
            detect_gps_spoofing(chunk)
        seq_time_taken = time.time() - seq_time_start

        for worker_count in worker_counts:
            cpu_usages, mem_usages = [], []

            time_start = time.time()
            with Pool(worker_count) as p:
                chunk_results = list(
                    tqdm(
                        p.imap(
                            lambda chunk: monitor(
                                detect_gps_spoofing, chunk), chunks
                        ),
                        desc=f"Processing chunk size {chunk_size} (Parallel)")
                )
            par_time_taken = time.time() - time_start

            for result, cpu_usage, mem_usage in chunk_results:
                cpu_usages.append(cpu_usage)
                mem_usages.append(mem_usage)

            speedup = seq_time_taken / par_time_taken
            avg_cpu_usage = np.mean(cpu_usages)
            avg_mem_usage = np.mean(mem_usages)

            results.append(
                (chunk_size, worker_count, speedup, avg_cpu_usage, avg_mem_usage))

    df_results = pd.DataFrame(results, columns=[
                              'Chunk Size', 'Workers', 'Speedup', 'Avg CPU Usage', 'Avg Memory Usage'])

    # Speedup Plot
    plt.figure()
    for chunk_size in chunk_sizes:
        chunk_results = df_results[df_results['Chunk Size'] == chunk_size]
        plt.plot(chunk_results['Workers'],
                 chunk_results['Speedup'], label=f'Chunk {chunk_size}')
    plt.xlabel('Number of Workers')
    plt.ylabel('Speedup')
    plt.title('Speedup vs Number of Workers')
    plt.legend()
    plt.show()

    # CPU Usage Plot
    plt.figure()
    for chunk_size in chunk_sizes:
        chunk_results = df_results[df_results['Chunk Size'] == chunk_size]
        plt.plot(chunk_results['Workers'], chunk_results['Avg CPU Usage'],
                 label=f'Chunk {chunk_size}', linestyle='dashed')
    plt.xlabel('Number of Workers')
    plt.ylabel('Average CPU Usage (%)')
    plt.title('CPU Usage vs Number of Workers')
    plt.legend()
    plt.show()

    # Memory Usage Plot
    plt.figure()
    for chunk_size in chunk_sizes:
        chunk_results = df_results[df_results['Chunk Size'] == chunk_size]
        plt.plot(chunk_results['Workers'],
                 chunk_results['Avg Memory Usage'], label=f'Chunk {chunk_size}')
    plt.xlabel('Number of Workers')
    plt.ylabel('Average Memory Usage (%)')
    plt.title('Memory Usage vs Number of Workers')
    plt.legend()
    plt.show()
