from CVRP_Geography_v4 import CVRP_Geography
import time
import pandas as pd
import numpy as np

# 時間計測開始
start_time = time.time()

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
#geo.assign_random_support_needs("omaezaki_nodes.csv", "omaezaki_nodes_map.html")

#これ以降(1) ~ (4)は関数化するべき

# (1)御前崎市の道路ネットワークを取得(一度だけ実施) 全ての道路種類を使った場合
'''
G = ox.graph_from_place("Omaezaki, Shizuoka, Japan", network_type="drive")
# ネットワークデータをGraphML形式で保存
ox.save_graphml(G, filepath="omaezaki_drive_network.graphml")
print("ネットワークデータを保存しました。")
# 道路タイプを色分けした地図を保存
geo.plot_colored_roads("omaezaki_drive_network.graphml", "omaezaki_road_map_colored.png")
# 各種ファイルのパス
graphml_file = "omaezaki_drive_network.graphml"  # GraphMLファイル
nodes_csv = "omaezaki_nodes.csv"  # ノードデータCSV
output_csv = "omaezaki_travel_time.csv"  # 通常形式の保存先
output_matrix_csv = "omaezaki_travel_time_matrix.csv"  # 行列形式の保存先
# 移動時間の計算と保存(時間がかかる)
geo.calculate_travel_times(graphml_file, nodes_csv, output_csv, output_matrix_csv)

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

    # 対称行列をCSVファイルに保存
    symmetric_matrix_df = pd.DataFrame(symmetric_matrix, index=travel_time_matrix.index, columns=travel_time_matrix.columns)
    symmetric_matrix_df.to_csv("omaezaki_symmetric_travel_time_matrix.csv")

except FileNotFoundError:
    geo.calculate_travel_times(graphml_file, nodes_csv, output_csv, output_matrix_csv)
'''


#2025/03/11 利用道路の制限
# (2)幹線道路（trunk）使用しないパターン
#geo.get_filtered_road_network(exclude_types=["trunk"], output_file="omaezaki_no_trunk.graphml")
#geo.plot_colored_roads("omaezaki_no_trunk.graphml", "omaezaki_road_map_no_trunk.png")

graphml_file = "omaezaki_no_trunk.graphml"  # GraphMLファイル
nodes_csv = "omaezaki_nodes.csv"  # ノードデータCSV
output_csv = "omaezaki_travel_time_no_trunk.csv"  # 通常形式の保存先
output_matrix_csv = "omaezaki_travel_time_matrix_no_trunk.csv"  # 行列形式の保存先
geo.calculate_travel_times(graphml_file, nodes_csv, output_csv, output_matrix_csv) # 移動時間の計算と保存(時間がかかる)
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

    # 対称行列をCSVファイルに保存
    symmetric_matrix_df = pd.DataFrame(symmetric_matrix, index=travel_time_matrix.index, columns=travel_time_matrix.columns)
    symmetric_matrix_df.to_csv("omaezaki_symmetric_travel_time_matrix_no_trunk.csv")

except FileNotFoundError:
    geo.calculate_travel_times(graphml_file, nodes_csv, output_csv, output_matrix_csv)



# (3)trunk, primary使用しないパターン
#geo.get_filtered_road_network(exclude_types=["trunk", "primary"], output_file="omaezaki_no_trunk_primary.graphml")
#geo.plot_colored_roads("omaezaki_no_trunk_primary.graphml", "omaezaki_road_map_no_trunk_primary.png")

#graphml_file = "omaezaki_no_trunk_primary.graphml"  # GraphMLファイル
#nodes_csv = "omaezaki_nodes.csv"  # ノードデータCSV
#output_csv = "omaezaki_travel_time_no_trunk_primary.csv"  # 通常形式の保存先
#output_matrix_csv = "omaezaki_travel_time_matrix_no_trunk_primary.csv"  # 行列形式の保存先
#geo.calculate_travel_times(graphml_file, nodes_csv, output_csv, output_matrix_csv) # 移動時間の計算と保存(時間がかかる)


# (4)trunk, primary, secondary使用しないパターン
#geo.get_filtered_road_network(exclude_types=["trunk", "primary", "secondary"], output_file="omaezaki_no_trunk_primary_secondary.graphml")
#geo.plot_colored_roads("omaezaki_no_trunk_primary_secondary.graphml", "omaezaki_road_map_no_trunk_primary_secondary.png")

#graphml_file = "omaezaki_no_trunk_primary_secondary.graphml"  # GraphMLファイル
#nodes_csv = "omaezaki_nodes.csv"  # ノードデータCSV
#output_csv = "omaezaki_travel_time_no_trunk_primary_secondary.csv"  # 通常形式の保存先
#output_matrix_csv = "omaezaki_travel_time_matrix_no_trunk_primary_secondary.csv"  # 行列形式の保存先
#geo.calculate_travel_times(graphml_file, nodes_csv, output_csv, output_matrix_csv) # 移動時間の計算と保存(時間がかかる)


'''
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

    # 対称行列をCSVファイルに保存
    symmetric_matrix_df = pd.DataFrame(symmetric_matrix, index=travel_time_matrix.index, columns=travel_time_matrix.columns)
    symmetric_matrix_df.to_csv("omaezaki_symmetric_travel_time_matrix.csv")

except FileNotFoundError:
    geo.calculate_travel_times(graphml_file, nodes_csv, output_csv, output_matrix_csv)
'''
    
#車両情報設定
num_vehicles = 10
vehicle_capacity = 4
vehicles = geo.set_vehicle_info(num_vehicles, vehicle_capacity)
print(vehicles)
# 車両情報をCSVファイルに保存
vehicles_df = pd.DataFrame(vehicles)
vehicles_df.to_csv("omaezaki_vehicle_info.csv", index=False)

# 計算時間の表示
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\n計算時間: {elapsed_time:.2f} 秒")
