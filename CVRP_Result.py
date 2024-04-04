import pandas as pd

def process_cluster_data(cluster_id, speed_kmh=30.0):
    filename = f'solution_cluster_{cluster_id}.csv'
    df = pd.read_csv(filename)
    
    # 使用車両台数 = データの行数
    vehicle_count = len(df)
    
    # 総移動距離 = Distance (m) 列の合計
    total_distance = df['Distance (m)'].sum()
    
    # 移動時間の和を計算 (速度 = 30km/h)
    # 移動時間(時間) = 距離(m) / 速度(m/s)
    speed_m_s = speed_kmh * 1000 / 3600  # km/h を m/s に変換
    total_time_hours = total_distance / speed_m_s / 3600  # 秒を時間に変換
    
    return (cluster_id, vehicle_count, total_distance, total_time_hours)


n_clusters = 38
cluster_ids = range(n_clusters)  # クラスタIDを適宜調整

# 各クラスタのデータを処理
results = [process_cluster_data(cluster_id) for cluster_id in cluster_ids]

# 結果をDataFrameに変換
results_df = pd.DataFrame(results, columns=['Cluster Number', 'Vehicle Count', 'Total Distance', 'Total Time'])

# 結果をCSVに保存
results_df.to_csv('cluster_summary.csv', index=False)
