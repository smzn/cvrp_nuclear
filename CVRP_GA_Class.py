import matplotlib.pyplot as plt
import folium
from deap import base, creator, tools, algorithms
import numpy as np
import random
from geopy.distance import great_circle
import pandas as pd
import time

class CVRP_GA:
    def __init__(self, df, vehicle_count, vehicle_capacity, population_size, ngen, cxpb, mutpb, depot_latitude, depot_longitude):
        self.df = df
        self.vehicle_count = vehicle_count
        self.vehicle_capacity = vehicle_capacity
        self.population_size = population_size
        self.ngen = ngen
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.toolbox = base.Toolbox()
        self.depot_latitude = depot_latitude
        self.depot_longitude = depot_longitude
        self.cost = None  # cost変数の初期化
        self.calculate_cost_matrix()  # cost行列の計算
        self.setup()

    def calculate_cost_matrix(self):
        """顧客間の距離行列（コスト行列）を計算し、self.costに格納する"""
        # DataFrameの長さを取得
        n_customers = len(self.df)
        
        # 距離行列を計算
        dist_matrix = [[great_circle((self.df.loc[i, 'latitude'], self.df.loc[i, 'longitude']),
                                     (self.df.loc[j, 'latitude'], self.df.loc[j, 'longitude'])).meters
                        if i != j else 0 for j in range(n_customers)] for i in range(n_customers)]
        
        # 距離行列をnumpy配列に変換してself.costに格納
        self.cost = np.array(dist_matrix)

    def setup(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        self.toolbox.register("attribute", random.sample, range(1, len(self.df)), len(self.df) - 1)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.attribute)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self.evaluate)

    def evaluate(self, individual):
        total_distance = 0.0  # 総移動距離
        depot_position = (self.depot_latitude, self.depot_longitude)

        # 現在の車両の位置を初期化（デポの位置）
        current_position = depot_position
        current_load = 0  # 現在の車両の積載量

        for customer_idx in individual:
            customer_demand = self.df.iloc[customer_idx]['demand']
            customer_position = (self.df.iloc[customer_idx]['latitude'], self.df.iloc[customer_idx]['longitude'])

            # 次の顧客を訪問すると車両の容量を超える場合は、最後の顧客からデポに戻る
            if current_load + customer_demand > self.vehicle_capacity:
                # 最後の顧客からデポに戻る距離を加算
                total_distance += great_circle(current_position, depot_position).meters
                # 容量をリセットし、デポから次の顧客への距離を加算
                current_load = 0
                # デポから次の顧客までの距離を加算（この顧客訪問をスキップしない）
                total_distance += great_circle(depot_position, customer_position).meters
                current_position = customer_position
                current_load += customer_demand
            else:
                # 容量内であれば次の顧客を訪問
                total_distance += great_circle(current_position, customer_position).meters
                current_position = customer_position
                current_load += customer_demand

        # 最後の顧客からデポに戻る距離を加算
        total_distance += great_circle(current_position, depot_position).meters

        # 適応度の計算（総移動距離）
        fitness = total_distance

        return fitness,

    def run(self):
        # 計算開始前のタイムスタンプを取得
        start_time = time.time()
        pop = self.toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # アルゴリズムを実行
        pop, logbook = algorithms.eaSimple(pop, self.toolbox, self.cxpb, self.mutpb, self.ngen, stats=stats, halloffame=hof, verbose=True)

        # 計算終了後のタイムスタンプを取得
        end_time = time.time()

        # 実行時間（秒）を計算
        elapsed_time = end_time - start_time

        # 実行時間を表示
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

        return pop, logbook, hof
    
    def save_results(self, logbook, hof, map_filename="map.png", stats_filename="stats.png"):
        # グラフの保存
        gen = logbook.select("gen")
        avg = logbook.select("avg")
        min_ = logbook.select("min")
        max_ = logbook.select("max")

        plt.figure(figsize=(10, 6))
        plt.plot(gen, avg, label="Average Fitness")
        plt.plot(gen, min_, label="Minimum Fitness")
        plt.plot(gen, max_, label="Maximum Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend()
        plt.title("Fitness over Generations")
        plt.grid()
        plt.savefig(stats_filename)
        plt.close()

        # df_routesの作成
        df_routes = self.create_routes_dataframe(hof[0])
        df_routes.to_csv('routes_result.csv', index=False)
        
        # 地図の保存
        self.create_routes_map(df_routes)
    
    def create_routes_dataframe(self, best_individual):
        """最良個体に基づいてルートを構築し、ルートごとのデータフレームを作成する"""
        routes = []
        current_route = [0]  # デポから開始
        current_load = 0
        current_distance = 0
        total_distance = 0
        route_distances = []
        route_demands = []
        route_strings = []  # ルートの文字列を保存するリスト

        # 時速30kmを秒速に変換
        speed_m_s = 30 * 1000 / 3600
        
        for customer_idx in best_individual:
            customer_demand = self.df.loc[customer_idx, 'demand']
            if current_load + customer_demand > self.vehicle_capacity:
                # ルートの情報を保存
                current_route.append(0)  # デポに戻る
                routes.append(current_route)
                route_distances.append(current_distance)
                route_demands.append(current_load)
                
                # デポに戻って新しいルートを開始
                current_route = [0, customer_idx]
                current_load = customer_demand
                current_distance = self.cost[0, customer_idx]
            else:
                # 顧客を訪問
                current_route.append(customer_idx)
                current_load += customer_demand
                if len(current_route) > 2:  # 最初のデポを除く
                    last_customer_idx = current_route[-2]
                    current_distance += self.cost[last_customer_idx, customer_idx]

        for route in routes:
            route_str = "0"
            for i in range(1, len(route)):
                prev_idx = route[i-1]
                curr_idx = route[i]
                distance = self.cost[prev_idx, curr_idx]
                travel_time = distance / speed_m_s  # 移動時間を計算
                route_str += f" -({travel_time:.0f})> {curr_idx}"
            route_strings.append(route_str)
        
        # ルートごとのデータフレームを作成
        df_routes = pd.DataFrame({
            "Route": route_strings,
            "Distance": route_distances,
            "Demand": route_demands
        })

        return df_routes
    
    def create_routes_map_allonly(self, df_routes, map_filename='routes_map.html'):
        """df_routesを基にしてルートを地図上に描画し、ファイルに保存する"""
        # デポの位置
        depot_position = (self.depot_latitude, self.depot_longitude)

        # 地図の作成（デポの位置を中心とする）
        m = folium.Map(location=depot_position, zoom_start=13)

        # デポを地図上に追加
        folium.Marker(depot_position, popup='Depot', icon=folium.Icon(color='red', icon='home')).add_to(m)

        # 色のリストを定義
        colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'darkblue', 'cadetblue', 'darkgreen', 'lightblue']

        # df_routesから各ルートを可視化
        for index, row in df_routes.iterrows():
            # ルートを表す顧客インデックスのリストを取得
            route_indices = []
            parts = row['Route'].split('>')

            for part in parts:
                part = part.strip()
                if '-(' in part:
                    part = part.split('-(')[0].strip()
                if part.isdigit():
                    route_indices.append(int(part))

            # デポを先頭と末尾に追加（重複を避ける）
            if route_indices[0] != 0:
                route_indices = [0] + route_indices
            if route_indices[-1] != 0:
                route_indices.append(0)

            # デバッグ用にルートインデックスを表示
            print(f"Route string: {row['Route']}")
            print(f"Route indices: {route_indices}")

            # ルートの線を描画
            route_positions = [(self.depot_latitude, self.depot_longitude)] + \
                            [(self.df.loc[idx, 'latitude'], self.df.loc[idx, 'longitude']) for idx in route_indices[1:-1]] + \
                            [(self.depot_latitude, self.depot_longitude)]
            print(f"Route positions: {route_positions}")  # デバッグ用: ルート位置を表示
            folium.PolyLine(route_positions, color=colors[index % len(colors)], weight=5, opacity=0.8).add_to(m)

            # 各顧客位置にマーカーを追加
            for pos in route_positions[1:-1]:  # デポを除外
                folium.CircleMarker(
                    location=pos,
                    radius=5,
                    color=colors[index % len(colors)],
                    fill=True,
                    fill_color=colors[index % len(colors)]
                ).add_to(m)

        # 地図をHTMLファイルとして保存
        m.save(map_filename)

    def create_routes_map(self, df_routes, map_filename='routes_map.html'):
        """df_routesを基にしてルートを地図上に描画し、ファイルに保存する"""
        # デポの位置
        depot_position = (self.depot_latitude, self.depot_longitude)

        # 地図の作成（デポの位置を中心とする）
        m = folium.Map(location=depot_position, zoom_start=13)

        # デポを地図上に追加
        folium.Marker(depot_position, popup='Depot', icon=folium.Icon(color='red', icon='home')).add_to(m)

        # 色のリストを定義
        colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'darkblue', 'cadetblue', 'darkgreen', 'lightblue']

        # df_routesから各ルートを可視化
        for index, row in df_routes.iterrows():
            # ルートを表す顧客インデックスのリストを取得
            route_indices = []
            parts = row['Route'].split('>')

            for part in parts:
                part = part.strip()
                if '-(' in part:
                    part = part.split('-(')[0].strip()
                if part.isdigit():
                    route_indices.append(int(part))

            # デポを先頭と末尾に追加（重複を避ける）
            if route_indices[0] != 0:
                route_indices = [0] + route_indices
            if route_indices[-1] != 0:
                route_indices.append(0)

            # デバッグ用にルートインデックスを表示
            print(f"Route string: {row['Route']}")
            print(f"Route indices: {route_indices}")

            # ルートの線を描画
            route_positions = [(self.depot_latitude, self.depot_longitude)] + \
                            [(self.df.loc[idx, 'latitude'], self.df.loc[idx, 'longitude']) for idx in route_indices[1:-1]] + \
                            [(self.depot_latitude, self.depot_longitude)]
            print(f"Route positions: {route_positions}")  # デバッグ用: ルート位置を表示
            folium.PolyLine(route_positions, color=colors[index % len(colors)], weight=5, opacity=0.8).add_to(m)

            # 各顧客位置にマーカーを追加
            for pos in route_positions[1:-1]:  # デポを除外
                folium.CircleMarker(
                    location=pos,
                    radius=5,
                    color=colors[index % len(colors)],
                    fill=True,
                    fill_color=colors[index % len(colors)]
                ).add_to(m)

            # 各ルートごとの地図を作成
            route_map = folium.Map(location=depot_position, zoom_start=13)
            folium.Marker(depot_position, popup='Depot', icon=folium.Icon(color='red', icon='home')).add_to(route_map)
            folium.PolyLine(route_positions, color=colors[index % len(colors)], weight=5, opacity=0.8).add_to(route_map)
            for pos in route_positions[1:-1]:  # デポを除外
                folium.CircleMarker(
                    location=pos,
                    radius=5,
                    color=colors[index % len(colors)],
                    fill=True,
                    fill_color=colors[index % len(colors)]
                ).add_to(route_map)
            
            # 各ルートごとの地図を保存
            route_map.save(f'route_map_{index + 1}.html')

        # 全体の地図をHTMLファイルとして保存
        m.save(map_filename)





if __name__ == "__main__":
    # CSVファイルから顧客データを読み込む
    file_name = 'peoplelist2.csv'
    df = pd.read_csv(file_name)

    # デポの情報をDataFrameに追加
    depot_latitude = 34.638221
    depot_longitude = 138.128204
    new_row = pd.DataFrame({
        'latitude': [depot_latitude],
        'longitude': [depot_longitude],
        'demand': [0]
    })
    df = pd.concat([new_row, df]).reset_index(drop=True)

    # GAのパラメータ
    vehicle_count = 10
    vehicle_capacity = 4
    population_size = 50
    ngen = 20
    cxpb = 0.7
    mutpb = 0.2

    # CVRP_GAクラスのインスタンスを生成
    cvrp_ga = CVRP_GA(df, vehicle_count, vehicle_capacity, population_size, ngen, cxpb, mutpb, depot_latitude, depot_longitude)

    # GAを実行して最適化
    pop, logbook, hof = cvrp_ga.run()

    # 結果のデータフレームを作成
    #df_routes = cvrp_ga.create_routes_dataframe(hof)
    #df_routes.to_csv('routes_result.csv', index=False)

    # 結果を保存
    cvrp_ga.save_results(logbook, hof, map_filename="map_result.html", stats_filename="ga_evolution_stats.png")

    print("Optimization completed. Results saved.")