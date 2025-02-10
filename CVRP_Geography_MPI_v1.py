from CVRP_Geography_v3 import CVRP_Geography
from CVRP_Calculation_3d_v1 import CVRP_Calculation_3d
import time
import pandas as pd
import numpy as np
from mpi4py import MPI
import csv

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 時間計測開始
start_time = time.time()


# 初期化
symmetric_matrix = None
nodes = None
vehicles = None

# ランク0のプロセスのみがデータを準備
if rank == 0:
    # クラスのインスタンス化
    geo = CVRP_Geography("r2ka22223.topojson")

    # データの読み込み
    geo.load_data()

    # 要支援者人数の割り当て
    total_support_needs = 300  # 例: 要支援者人数を 500 人に設定
    geo.assign_support_needs(total_support_needs)

    # CSVの保存
    geo.save_csv("omaezaki_districts.csv")

    # 地図の生成と保存
    geo.generate_map("omaezaki_map_with_markers.html")

    # 市役所の情報を辞書形式で定義
    city_office_data = {
        "名称": "御前崎市役所",
        "所在地_連結表記": "静岡県御前崎市池新田5585",
        "緯度": 34.637984,
        "経度": 138.128125,
        "想定収容人数": 0,  # 市役所に収容人数は設定しない
        "備考": "市役所"
    }
    # 避難所データの読み込み
    geo.load_shelters("【2024.10.16】御前崎市避難所一覧.csv", city_office_data)

    # 一次避難所以外の避難所の地図表示
    geo.plot_shelters("omaezaki_shelters_map.html")

    #国土地理院API(標高取得)を利用(使わないときはコメントアウト)
    #    geo.assign_random_support_needs("omaezaki_nodes.csv", "omaezaki_nodes_map.html")

    # 御前崎市の道路ネットワークを取得(一度だけ実施)
    #G = ox.graph_from_place("Omaezaki, Shizuoka, Japan", network_type="drive")
    # ネットワークデータをGraphML形式で保存
    #ox.save_graphml(G, filepath="omaezaki_drive_network.graphml")
    #print("ネットワークデータを保存しました。")

    # 道路タイプを色分けした地図を保存
    geo.plot_colored_roads("omaezaki_drive_network.graphml", "omaezaki_road_map_colored.png")

    # 各種ファイルのパス
    graphml_file = "omaezaki_drive_network.graphml"  # GraphMLファイル
    nodes_csv = "omaezaki_nodes.csv"  # ノードデータCSV
    output_csv = "omaezaki_travel_time.csv"  # 通常形式の保存先
    output_matrix_csv = "omaezaki_travel_time_matrix.csv"  # 行列形式の保存先
    # 移動時間の計算と保存(時間がかかる)
    #geo.calculate_travel_times(graphml_file, nodes_csv, output_csv, output_matrix_csv)

    try:
        travel_time_matrix = pd.read_csv(output_matrix_csv, index_col=0)
        # 行列の値だけを取得（インデックスや列名を除く）
        matrix_values = travel_time_matrix.values
        # データの最初の数行を表示
        print(matrix_values)
        # 上三角行列を対称行列に変換する関数
        def make_symmetric(matrix):
            # 上三角部分を下三角にコピーして対称行列を作成
            symmetric_matrix = matrix + matrix.T - np.diag(np.diag(matrix))
            return symmetric_matrix

        # 対称行列を作成
        symmetric_matrix = make_symmetric(np.array(matrix_values))

        # 結果を表示
        print("Symmetric Matrix:")
        print(symmetric_matrix)
    except FileNotFoundError:
        geo.calculate_travel_times(graphml_file, nodes_csv, output_csv, output_matrix_csv)

    omaezaki_nodes_csv = "omaezaki_nodes.csv"
    try:
        # CSVファイルを読み込む
        nodes_data = pd.read_csv(omaezaki_nodes_csv)

        # 必要な列だけを取得
        nodes = nodes_data[["id", "type", "x", "y", "z", "demand"]].to_dict(orient="records")

        # 読み込み成功メッセージ
        print("Nodes data successfully loaded on rank 0.")

    except FileNotFoundError:
        print(f"ファイル '{omaezaki_nodes_csv}' が見つかりませんでした。パスを確認してください。")
        nodes = None

    # 車両情報の設定
    num_vehicles = 10
    vehicle_capacity = 4
    vehicles = geo.set_vehicle_info(num_vehicles, vehicle_capacity)
    print("Vehicle data successfully set on rank 0.")


# 対称行列、ノードデータ、車両データを全プロセスに共有
symmetric_matrix = comm.bcast(symmetric_matrix, root=0)
nodes = comm.bcast(nodes, root=0)
vehicles = comm.bcast(vehicles, root=0)

# データが共有されたかどうか確認
if symmetric_matrix is not None:
    print(f"Rank {rank}: Received symmetric matrix data.")
else:
    print(f"Rank {rank}: No symmetric matrix data received.")

if nodes is not None:
    print(f"Rank {rank}: Received nodes data.")
else:
    print(f"Rank {rank}: No nodes data received.")

if vehicles is not None:
    print(f"Rank {rank}: Received vehicles data.")
else:
    print(f"Rank {rank}: No vehicles data received.")


# 遺伝アルゴリズムのパラメータ設定
population_size = 300
crossover_rate = 0.8
mutation_rate = 0.1
generations = 30
penalty = 1000


# CVRP_Calculation のインスタンス生成
calculation = CVRP_Calculation_3d(
    nodes=nodes,
    vehicles=vehicles,
    cost_matrix=symmetric_matrix,
    population_size=population_size,
    crossover_rate=crossover_rate,
    mutation_rate=mutation_rate,
    generations=generations,
    penalty=penalty
)

# ランク0で初期集団を生成して分配準備
if rank == 0:
    # 初期集団を生成
    population = calculation.generate_initial_population(population_size)

    # 初期集団をプロセス数で均等に分割
    chunk_size = population_size // size
    population_chunks = [population[i * chunk_size:(i + 1) * chunk_size] for i in range(size)]

    # 不均等がある場合に対応するため、余り分を調整
    remaining = population_size % size
    for i in range(remaining):
        population_chunks[i].append(population[-(i + 1)])

    print(f"Rank 0: Initial population divided into {len(population_chunks)} chunks.")
else:
    population_chunks = None

# 各プロセスに初期遺伝子情報を分配
local_population = comm.scatter(population_chunks, root=0)

# 各プロセスで割り当てられた遺伝子を確認
print(f"Rank {rank}: Received {len(local_population)} individuals from the initial population.")

# 各プロセスで個体群を評価
for generation in range(generations):

    # デバッグ: 各プロセスが受け取った個体数を確認
    print(f"Rank {rank}: Received {len(local_population)} individuals for generation {generation + 1}.")

    # 適応度の計算
    #local_fitness_values = [calculation.evaluate_individual(ind) for ind in local_population]
    # 適応度の計算
    local_fitness_values = []
    for i, ind in enumerate(local_population):
        fitness = calculation.evaluate_individual(ind)
        local_fitness_values.append(fitness)
        # デバッグ: 個体ごとの適応度を表示
        print(f"Rank {rank}: Evaluated individual {i} with fitness = {fitness}.")

    # 各プロセスから適応度と最良個体をランク0に集約
    all_fitness_values = comm.gather(local_fitness_values, root=0)
    all_populations = comm.gather(local_population, root=0)

    # CSVデータの準備
    results = []
    #best_overall_individual = None
    #best_overall_fitness = float('inf')

    if rank == 0:
        # 集約結果を確認
        print(f"Rank 0: Combined population size = {len(all_populations)}, Combined fitness size = {len(all_fitness_values)}")
        # 集約された適応度と個体群を統合
        combined_fitness = [fitness for process_fitness in all_fitness_values for fitness in process_fitness]
        combined_population = [ind for process_population in all_populations for ind in process_population]

        # 現世代の最良個体を選択
        best_fitness = min(combined_fitness)
        best_individual = combined_population[np.argmin(combined_fitness)]
        mean_fitness = np.mean(combined_fitness)
        std_fitness = np.std(combined_fitness)

        # 結果を表示
        print(f"世代 {generation + 1}: 最良適合度 = {best_fitness}, 平均適合度 = {mean_fitness}, 標準偏差 = {std_fitness}")
        # 結果を記録
        results.append([generation + 1, best_fitness, mean_fitness, std_fitness])

        # 次世代を作成
        new_population = calculation.create_next_generation(combined_population)

        # 次世代をプロセスに分配
        population_chunks = [new_population[i * chunk_size:(i + 1) * chunk_size] for i in range(size)]
        local_population = comm.scatter(population_chunks, root=0)
    else:
        # 次世代の受け取り
        local_population = comm.scatter(None, root=0)

# 最終結果をランク0で表示
if rank == 0:
    print(f"\n全世代を通しての最良適合度: {best_fitness}")
    print("最良個体:")
    for route in best_individual:
        print(f"  ルート: {route}")

    # 結果をCSVに保存
    output_csv = './result/genetic_results.csv'
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Generation', 'Best Fitness', 'Mean Fitness', 'Std Dev'])
        writer.writerows(results)

    # 最良個体をCSVに保存
    best_individual_csv = './result/best_individual.csv'
    with open(best_individual_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Vehicle', 'Route', 'Fitness'])
        for i, route in enumerate(best_individual):
            writer.writerow([f"Vehicle {i + 1}", ' -> '.join(map(str, route)), ""])
        writer.writerow([])
        writer.writerow(['Best Fitness', best_fitness])

    # グラフの作成、搬送時間の計算、ヒストグラムの作成を呼び出し
    calculation.plot_results(results, output_csv.replace('.csv', '.png'))
    calculation.calculate_times(best_individual)
    calculation.plot_histograms()


'''
# 初期集団を生成
population = calculation.generate_initial_population(population_size)
# 遺伝アルゴリズムの実行
best_individual, best_fitness = calculation.run_genetic_algorithm(generations)

calculation.save_vehicle_statistics(best_individual)
calculation.visualize_routes(best_individual, nodes)
calculation.visualize_routes_3d(best_individual, nodes)
#calculation.plot_elevation_changes(best_individual, nodes, cost_matrix)
'''

# 計算時間の表示
if rank == 0:
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n計算時間: {elapsed_time:.2f} 秒")

#mpiexec -n 4 python3 CVRP_Geography_MPI_v1.py