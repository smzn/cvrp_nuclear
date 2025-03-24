import geopandas as gpd
import folium
import pandas as pd
import random
import requests
import osmnx as ox
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
from shapely.geometry import Point

class CVRP_Geography:
    def __init__(self, file_path, layer_name="town"):
        """
        初期化メソッド
        :param file_path: TopoJSONファイルのパス
        :param layer_name: 読み込むレイヤー名（デフォルトは "town"）
        """
        self.file_path = file_path
        self.layer_name = layer_name
        self.gdf = None
        self.center_lat = None
        self.center_lon = None

    def load_data(self):
        """ TopoJSONファイルを読み込む """
        try:
            self.gdf = gpd.read_file(self.file_path, layer=self.layer_name)

            # CRS（座標系）の設定（WGS84）
            if self.gdf.crs is None:
                self.gdf.set_crs(epsg=4326, inplace=True)

            # 地図の中心座標を取得
            self.center_lat = self.gdf.geometry.centroid.y.mean()
            self.center_lon = self.gdf.geometry.centroid.x.mean()

            print(f"成功: データの読み込みが完了しました。({self.layer_name})")
        except Exception as e:
            print(f"エラー: データの読み込みに失敗しました - {e}")

    def save_csv(self, output_csv_path):
        """
        CSVファイルとして保存
        :param output_csv_path: 保存するCSVファイルのパス
        """
        try:
            columns_to_keep = ["PREF_NAME", "CITY_NAME", "S_NAME", "AREA", "JINKO", "SETAI", "X_CODE", "Y_CODE", "SUPPORT_NEEDS"]
            if all(col in self.gdf.columns for col in columns_to_keep):
                self.gdf[columns_to_keep].to_csv(output_csv_path, index=False, encoding='utf-8-sig')
                print(f"CSVファイルが {output_csv_path} に保存されました。")
            else:
                print("必要なカラムがデータ内に見つかりませんでした。")
        except Exception as e:
            print(f"エラー: CSVファイルの保存に失敗しました - {e}")

    def generate_map(self, output_html_path):
        """
        Foliumを使用して地図を生成し、HTMLファイルとして保存
        :param output_html_path: 保存するHTMLファイルのパス
        """
        if self.gdf is None:
            print("エラー: データをロードしてください。")
            return

        try:
            m = folium.Map(location=[self.center_lat, self.center_lon], zoom_start=13, tiles="cartodbpositron")

            # 各地区にマーカーを設置
            for _, row in self.gdf.iterrows():
                s_name = row["S_NAME"]      # 町名
                area = row["AREA"]          # 面積
                jinko = row["JINKO"]         # 人口
                x_code = row["X_CODE"]       # 経度
                y_code = row["Y_CODE"]       # 緯度
                support_needs = row["SUPPORT_NEEDS"] # 要支援者人数

                # ポップアップに表示する内容
                popup_text = f"""
                <b>町名:</b> {s_name}<br>
                <b>面積:</b> {area:.2f} m²<br>
                <b>人口:</b> {jinko} 人 <br>
                <b>要支援者:</b> {support_needs} 人
                """

                # マーカーの追加
                folium.Marker(
                    location=[y_code, x_code],
                    popup=folium.Popup(popup_text, max_width=300),
                    icon=folium.Icon(color="blue", icon="info-sign")
                ).add_to(m)

            # GeoJSONレイヤーの追加
            folium.GeoJson(
                self.gdf.to_json(),
                style_function=lambda x: {'color': 'red', 'weight': 2, 'fillOpacity': 0.3}
            ).add_to(m)

            # HTMLファイルとして保存
            m.save(output_html_path)
            print(f"地図が {output_html_path} に保存されました。")
        except Exception as e:
            print(f"エラー: 地図の生成に失敗しました - {e}")
    
    def assign_support_needs(self, total_support_needs):
        """
        総要支援者人数を地区の人口比率に基づいて割り振る
        :param total_support_needs: 全体の要支援者の総数
        """
        if "JINKO" not in self.gdf.columns:
            print("エラー: 人口（JINKO）データが存在しません。")
            return

        # 人口の合計を取得
        total_population = self.gdf["JINKO"].sum()

        # 各地区に要支援者を人口比で割り当て
        self.gdf["SUPPORT_NEEDS"] = (self.gdf["JINKO"] / total_population * total_support_needs).round().astype(int)

        print("要支援者人数の割り当てが完了しました。")

    def load_shelters(self, csv_file, city_office_info=None):
        """
        避難所データのCSVファイルを読み込み、データフレームとして保存
        :param csv_file: 避難所情報を含むCSVファイルのパス
        """
        try:
            self.shelters_df = pd.read_csv(csv_file, encoding='shift_jis')

            # "一時避難所" 以外の避難所をフィルタリング
            self.shelters_df = self.shelters_df[self.shelters_df['備考'] != '一次避難所']
            # 市役所の情報が提供された場合に追加
            if city_office_info:
                city_office_df = pd.DataFrame([city_office_info])
                self.shelters_df = pd.concat([self.shelters_df, city_office_df], ignore_index=True)

            print(f"避難所データが正常にロードされました。対象避難所数: {len(self.shelters_df)}")
        except Exception as e:
            print(f"エラー: 避難所データの読み込みに失敗しました - {e}")

    def plot_shelters(self, output_html_path):
        """
        一時避難所以外の避難所を地図上に表示し、名称と想定収容人数を表示する
        :param output_html_path: 保存するHTMLファイルのパス
        """
        if self.shelters_df is None or self.shelters_df.empty:
            print("エラー: 避難所データがロードされていません。")
            return

        try:
            # 欠損値を適切な値に置き換え（収容人数が不明の場合は0にする）
            self.shelters_df["想定収容人数"] = self.shelters_df["想定収容人数"].fillna(0).astype(int)
            m = folium.Map(location=[self.center_lat, self.center_lon], zoom_start=13, tiles="cartodbpositron")

            for _, row in self.shelters_df.iterrows():
                name = row["名称"]
                capacity = row["想定収容人数"]
                lat = row["緯度"]
                lon = row["経度"]
                category = row["備考"]

                # 市役所は赤色アイコン、他の避難所は緑色アイコン
                if category == "市役所":
                    icon_color = "red"
                    icon_type = "info-sign"
                else:
                    icon_color = "green"
                    icon_type = "home"

                popup_text = f"<b>避難所:</b> {name}<br><b>想定収容人数:</b> {int(capacity)} 人<br><b>備考:</b> {category}"

                folium.Marker(
                    location=[lat, lon],
                    popup=folium.Popup(popup_text, max_width=300),
                    icon=folium.Icon(color=icon_color, icon=icon_type)
                ).add_to(m)

            m.save(output_html_path)
            print(f"避難所の地図が {output_html_path} に保存されました。")
        except Exception as e:
            print(f"エラー: 避難所の地図の生成に失敗しました - {e}")

    def get_gsi_elevation(self, lat, lon):
        """ 国土地理院APIを利用して標高を取得し、無効な値を処理 """
        url = f"https://cyberjapandata2.gsi.go.jp/general/dem/scripts/getelevation.php?lon={lon}&lat={lat}&outtype=JSON"
        try:
            response = requests.get(url)
            response.raise_for_status()  # HTTPリクエストのエラーチェック
            data = response.json()

            # 取得したレスポンスを表示（デバッグ用）
            print(f"取得データ（{lat}, {lon}）: {data}")

            if "elevation" in data:
                elevation = data["elevation"]

                # 標高データが '-----' などの無効な値かどうかをチェック
                if elevation == "-----" or elevation is None:
                    print(f"警告: 標高データが無効 ({lat}, {lon})")
                    return None  # または return 0 に変更可

                return round(float(elevation), 2)

            else:
                print(f"警告: 標高データが存在しない ({lat}, {lon})")
                return None

        except requests.exceptions.RequestException as e:
            print(f"エラー: 標高データの取得に失敗しました ({lat}, {lon}) - {e}")
            return None
        except ValueError as ve:
            print(f"エラー: 標高データの変換に失敗しました（{lat}, {lon}）- {ve}")
            return None

    def assign_random_support_needs(self, output_csv_path, map_output_html):
        """ 各地区の要支援者をランダムに割り当て、位置情報と標高データを設定して保存 """
        if self.gdf is None:
            print("エラー: データをロードしてください。")
            return

        assigned_data = []

        # 1. 市役所データを先に追加
        id_counter = 0  # idのカウンタを0から開始
        for _, row in self.shelters_df.iterrows():
            if row['備考'] == '市役所':
                entry_type = 'city_hall'
                elevation = self.get_gsi_elevation(row['緯度'], row['経度'])
                assigned_data.append({
                    'id': id_counter,  # idを設定
                    'type': entry_type,
                    'x': row['経度'],
                    'y': row['緯度'],
                    'z': elevation,  # elevationをzに変更
                    'demand': 0,
                    'priority': '-',
                    'name': row['名称'],
                    'capacity': row.get('想定収容人数', 0),
                    'remarks': row.get('備考', '')
                })
                id_counter += 1  # idをインクリメント

        # 2. 一般の避難所データを追加
        for _, row in self.shelters_df.iterrows():
            if row['備考'] != '市役所':
                entry_type = 'shelter'
                elevation = self.get_gsi_elevation(row['緯度'], row['経度'])
                assigned_data.append({
                    'id': id_counter,  # idを設定
                    'type': entry_type,
                    'x': row['経度'],
                    'y': row['緯度'],
                    'z': elevation,  # elevationをzに変更
                    'demand': 0,
                    'priority': '-',
                    'name': row['名称'],
                    'capacity': row.get('想定収容人数', 0),
                    'remarks': row.get('備考', '')
                })
                id_counter += 1  # idをインクリメント

        # 3. 要配慮者データを追加（修正版）
        for _, row in self.gdf.iterrows():
            support_needs = row['SUPPORT_NEEDS']
            polygon = row['geometry']  # シェープ情報
            for i in range(support_needs):
                while True:
                    # ポリゴン内にランダムポイントを生成
                    minx, miny, maxx, maxy = polygon.bounds
                    lon = random.uniform(minx, maxx)
                    lat = random.uniform(miny, maxy)
                    random_point = Point(lon, lat)

                    # ポイントがポリゴン内にあるかを確認
                    if not polygon.contains(random_point):
                        continue

                    # 標高データを取得して有効性を確認
                    elevation = self.get_gsi_elevation(lat, lon)
                    if elevation is not None:  # 標高が有効であればループ終了
                        break

                # 有効な座標と標高データで追加
                assigned_data.append({
                    'id': id_counter,
                    'type': 'client',
                    'x': lon,
                    'y': lat,
                    'z': elevation,
                    'demand': random.choice([1, 2]),
                    'priority': random.randint(1, 5),
                    'name': f"{row['S_NAME']}_{i+1}",
                    'capacity': 0,
                    'remarks': ''
                })
                id_counter += 1  # idをインクリメント

        # データをDataFrameに変換してCSV保存
        df_assigned = pd.DataFrame(assigned_data)
        df_assigned.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"要支援者データと避難所情報が {output_csv_path} に保存されました。")

        # 地図の作成
        m = folium.Map(location=[self.center_lat, self.center_lon], zoom_start=13, tiles="cartodbpositron")
        for entry in assigned_data:
            if entry['type'] == 'client':
                popup_text = f"<b>名前:</b> {entry['name']}<br><b>人数:</b> {entry['demand']} 人<br><b>優先度:</b> {entry['priority']}<br><b>標高:</b> {entry['z']} m"
                color, icon = "red", "user"
            else:
                popup_text = f"<b>避難所:</b> {entry['name']}<br><b>想定収容人数:</b> {entry['capacity']} 人<br><b>標高:</b> {entry['z']} m<br><b>備考:</b> {entry['remarks']}"
                color, icon = ("blue", "info-sign") if entry['type'] == 'city_hall' else ("green", "home")

            folium.Marker(
                location=[entry['y'], entry['x']],
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(color=color, icon=icon)
            ).add_to(m)

        m.save(map_output_html)
        print(f"地図が {map_output_html} に保存されました。")

    def plot_colored_roads(self, graphml_file, output_filepath):
        """
        道路タイプを色分けして地図を保存
        :param graphml_file: GraphMLファイルのパス
        :param output_filepath: 保存する画像ファイルのパス
        """
        try:
            # ローカルのGraphMLファイルを読み込み
            G = ox.load_graphml(filepath=graphml_file)
            print("ネットワークデータを読み込みました。")

            # 道路タイプごとに色を設定
            road_colors = {
                "trunk": "red",        # 幹線道路を赤
                "primary": "blue",     # 一次道路を青
                "secondary": "green",  # 二次道路を緑
                "tertiary": "orange",  # 三次道路をオレンジ
            }

            # ノードの位置情報を取得
            pos = {node: (data["x"], data["y"]) for node, data in G.nodes(data=True)}

            # 図を作成
            fig, ax = plt.subplots(figsize=(12, 12))

            # 道路タイプごとにエッジを色分けして描画
            for highway_type, color in road_colors.items():
                # 道路タイプに一致するエッジを取得
                edges = [(u, v) for u, v, k, d in G.edges(keys=True, data=True) if d.get("highway") == highway_type]
                
                # 該当エッジを描画
                if edges:
                    nx.draw_networkx_edges(
                        G, pos, edgelist=edges, ax=ax, edge_color=color, width=2, label=highway_type
                    )

            # 手動で凡例を作成
            legend_labels = {
                "trunk": "Trunk",
                "primary": "Primary",
                "secondary": "Secondary",
                "tertiary": "Tertiary"
            }
            handles = [plt.Line2D([0], [0], color=color, lw=2, label=legend_labels[road_type]) for road_type, color in road_colors.items()]
            ax.legend(handles=handles, title="Road Types", loc="upper left")

            plt.title("Color-coded Road Types in Omaezaki City", fontsize=16)
            plt.axis("off")

            # 地図を保存
            plt.savefig(output_filepath, dpi=300, bbox_inches="tight")
            print(f"地図を保存しました: {output_filepath}")
            #plt.show()

        except Exception as e:
            print(f"エラー: 道路地図の生成に失敗しました - {e}")

    def calculate_travel_times(self, graphml_file, nodes_csv, output_csv, output_matrix_csv):
        """
        全てのノード間の移動時間を計算し、結果をCSVと行列形式で保存
        :param graphml_file: GraphMLファイルのパス
        :param nodes_csv: ノード情報を含むCSVファイルのパス
        :param output_csv: 結果を保存するCSVファイルのパス
        :param output_matrix_csv: 行列形式の結果を保存するCSVファイルのパス
        """
        try:
            # グラフを読み込み
            print("ネットワークデータを読み込んでいます...")
            G = ox.load_graphml(filepath=graphml_file)
            print("ネットワークデータを読み込みました。")

            # エッジに移動時間（weight）を追加
            for u, v, k, data in G.edges(data=True, keys=True):
                # エッジの長さと制限速度を取得
                length = data.get("length", 1)  # 距離 (m)
                speed = data.get("maxspeed", 30)  # 制限速度 (km/h)

                # maxspeedがリストの場合、最初の値を使用
                if isinstance(speed, list):
                    speed = speed[0]

                # 制限速度がない場合はデフォルト値を使用
                try:
                    speed = float(speed)
                except (TypeError, ValueError):
                    speed = 30  # デフォルトの制限速度 (km/h)

                # 移動時間（秒）を計算してエッジに追加
                travel_time = length / (speed * 1000 / 3600)  # 秒単位の時間
                data["weight"] = travel_time

            # ノードデータを読み込み
            print("ノードデータを読み込んでいます...")
            nodes_df = pd.read_csv(nodes_csv)
            print("ノードデータを読み込みました。")

            # 結果を格納するリスト
            travel_times = []

            # 全てのノード間の組み合わせを取得
            node_pairs = combinations(nodes_df["id"], 2)

            # ノードIDと座標、名前をマッピング
            id_to_coords = nodes_df.set_index("id")[["x", "y", "name"]].to_dict(orient="index")

            # 全ての組み合わせで移動時間を計算
            for source_id, target_id in node_pairs:
                try:
                    source_coords = id_to_coords[source_id]
                    target_coords = id_to_coords[target_id]

                    # 起点と終点のノードを取得
                    source_node = ox.distance.nearest_nodes(G, X=source_coords["x"], Y=source_coords["y"])
                    target_node = ox.distance.nearest_nodes(G, X=target_coords["x"], Y=target_coords["y"])

                    # 最短経路を計算
                    route = nx.shortest_path(G, source_node, target_node, weight="weight")
                    travel_time = nx.shortest_path_length(G, source_node, target_node, weight="weight")

                    # 結果をリストに保存
                    travel_times.append({
                        "source_id": source_id,
                        "target_id": target_id,
                        "travel_time": travel_time,
                    })

                    # 計算状況をターミナルに出力
                    print(f"計算中: 拠点 {source_id} -> 拠点 {target_id} | 移動時間: {travel_time:.2f} 秒")

                except nx.NetworkXNoPath:
                    print(f"ルートが見つかりませんでした: {source_id} -> {target_id}")
                    travel_times.append({
                        "source_id": source_id,
                        "target_id": target_id,
                        "travel_time": None,
                    })

            # 結果をデータフレームに変換
            travel_times_df = pd.DataFrame(travel_times)
            all_nodes = list(range(len(nodes_df)))
            # 行列形式に変換して保存
            travel_times_df = travel_times_df.set_index(["source_id", "target_id"]).reindex(
                pd.MultiIndex.from_product([all_nodes, all_nodes], names=["source_id", "target_id"]),
                fill_value=0
            ).reset_index()

            # 結果をCSVに保存
            travel_times_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
            print(f"移動時間データを {output_csv} に保存しました。")

            travel_time_matrix = travel_times_df.pivot(index="source_id", columns="target_id", values="travel_time").fillna(0)
            print(f"Travel time matrix created with shape: {travel_time_matrix.shape}")
            travel_time_matrix.to_csv(output_matrix_csv, index=True, encoding="utf-8-sig")
            print(f"行列形式の移動時間データを {output_matrix_csv} に保存しました。")

        except Exception as e:
            print(f"エラー: {e}")

    def set_vehicle_info(self, num_vehicles, vehicle_capacity, vehicle_file="omaezaki_vehicles.csv"):
        """
        車両情報を設定するメソッド
        :param num_vehicles: 車両の台数
        :param vehicle_capacity: 各車両の容量
        :param vehicle_file: 車両情報を保存するCSVファイル名
        """

        # 車両の生成
        vehicles = [{"id": i, "capacity": vehicle_capacity} for i in range(num_vehicles)]
        df_vehicles = pd.DataFrame(vehicles)

        # CSV保存
        df_vehicles.to_csv(vehicle_file, index=False, columns = ["id", "capacity"])

        return vehicles
    
    def get_filtered_road_network(self, include_types=None, exclude_types=None, output_file="filtered_network.graphml"):
        """
        指定した道路種別を含める・除外する形で OSM ネットワークを取得し、GraphML に保存する。
        """
        if include_types:
            custom_filter = '["highway"~"' + "|".join(include_types) + '"]'
        elif exclude_types:
            custom_filter = '["highway"!~"' + "|".join(exclude_types) + '"]'
        else:
            custom_filter = None

        G = ox.graph_from_place("Omaezaki, Shizuoka, Japan", network_type="drive", custom_filter=custom_filter)

        ox.save_graphml(G, filepath=output_file)
        print(f"ネットワークデータを {output_file} に保存しました。")
        return G



if __name__ == "__main__":
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
    geo.assign_random_support_needs("omaezaki_nodes.csv", "omaezaki_nodes_map.html")

    # 御前崎市の道路ネットワークを取得(一度だけ実施)
    '''
    G = ox.graph_from_place("Omaezaki, Shizuoka, Japan", network_type="drive")
    # ネットワークデータをGraphML形式で保存
    ox.save_graphml(G, filepath="omaezaki_drive_network.graphml")
    print("ネットワークデータを保存しました。")
    '''

    # 道路タイプを色分けした地図を保存
    geo.plot_colored_roads("omaezaki_drive_network.graphml", "omaezaki_road_map_colored.png")

    # 各種ファイルのパス
    graphml_file = "omaezaki_drive_network.graphml"  # GraphMLファイル
    nodes_csv = "omaezaki_nodes.csv"  # ノードデータCSV
    output_csv = "omaezaki_travel_time.csv"  # 通常形式の保存先
    output_matrix_csv = "omaezaki_travel_time_matrix.csv"  # 行列形式の保存先
    # 移動時間の計算と保存(時間がかかる)
    geo.calculate_travel_times(graphml_file, nodes_csv, output_csv, output_matrix_csv)

    #車両情報設定
    num_vehicles = 10
    vehicle_capacity = 4
    geo.set_vehicle_info(num_vehicles, vehicle_capacity)




