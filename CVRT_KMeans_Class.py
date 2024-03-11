import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from geopy.distance import great_circle
import folium
import pulp
import itertools
import time
import psutil
import os

class ClusterBasedVehicleRouting:
    def __init__(self, vehicle_count, vehicle_capacity, n_clusters, depot_latitude, depot_longitude, file_name, cluster_id):
        self.vehicle_count = vehicle_count
        self.vehicle_capacity = vehicle_capacity
        self.n_clusters = n_clusters
        self.cluster_id = cluster_id
        self.depot_latitude = depot_latitude
        self.depot_longitude = depot_longitude
        self.file_name = file_name
        self.df = None
        self.cluster_df = None
        self.cost = None
        self.routes = {}
        self.distances = {}
        self.result_status = None

    def load_and_prepare_data(self):
        self.df = pd.read_csv(self.file_name)
        self.df['demand'] = 1
        print(self.df)

    def perform_clustering(self):
        X = self.df[['longitude', 'latitude']]
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(X)
        self.df['Cluster'] = kmeans.labels_
        print(self.df)

    def prepare_depot(self):
        new_row = pd.DataFrame({
            'latitude': [self.depot_latitude],
            'longitude': [self.depot_longitude],
            'demand': [0],
            'GroupID': [-1],
            'NearestCenterName': ['Depot'],
            'Cluster': [-1]
        })
        self.df = pd.concat([new_row, self.df]).reset_index(drop=True)
        print(self.df)

    def extract_cluster(self):
        self.cluster_df = self.df[(self.df['Cluster'] == -1) | (self.df['Cluster'] == self.cluster_id)].reset_index(drop=False)
        self.cluster_df.rename(columns={'index': 'id'}, inplace=True)
        self.cluster_df.to_csv(f'cluster_df_{self.cluster_id}.csv', index=False)
        print(self.cluster_df)

    def calculate_distance_matrix(self):
        dist = [[great_circle((self.cluster_df.loc[i, 'latitude'], self.cluster_df.loc[i, 'longitude']),
                              (self.cluster_df.loc[j, 'latitude'], self.cluster_df.loc[j, 'longitude'])).meters
                 if i != j else 0 for j in range(len(self.cluster_df))] for i in range(len(self.cluster_df))]
        self.cost = np.array(dist)
        print(self.cost)

    def solve_vehicle_routing_problem(self, timeout_seconds=60):
        # Initializations
        customer_count = len(self.cluster_df)
        depot_index = 0  # Assuming the first row is the depot
        demand = self.cluster_df['demand'].tolist()

        # Define the problem
        problem = pulp.LpProblem("CVRP", pulp.LpMinimize)

        # Decision variables
        x = pulp.LpVariable.dicts("x", [(i, j, k) for i in range(customer_count)
                                        for j in range(customer_count)
                                        for k in range(self.vehicle_count)],
                                  cat=pulp.LpBinary)
        
        # Objective function
        problem += pulp.lpSum([self.cost[i][j] * x[(i, j, k)] for i in range(customer_count) for j in range(customer_count) for k in range(self.vehicle_count) if i != j])

        # Constraints
        for j in range(1, customer_count):
            problem += pulp.lpSum([x[(i, j, k)] for i in range(customer_count) for k in range(self.vehicle_count) if i != j]) == 1

        for k in range(self.vehicle_count):
            problem += pulp.lpSum([x[(depot_index, j, k)] for j in range(1, customer_count)]) <= 1  # Leave depot
            problem += pulp.lpSum([x[(j, depot_index, k)] for j in range(1, customer_count)]) <= 1  # Return to depot

        for k in range(self.vehicle_count):
            for i in range(1, customer_count):
                problem += pulp.lpSum([x[(i, j, k)] for j in range(customer_count) if i != j]) - \
                           pulp.lpSum([x[(j, i, k)] for j in range(customer_count) if i != j]) == 0

        for k in range(self.vehicle_count):
            problem += pulp.lpSum([demand[j] * x[(i, j, k)] for i in range(customer_count) for j in range(1, customer_count) if i != j]) <= self.vehicle_capacity

        # Subtour elimination
        for i in range(2, customer_count):
            for s in itertools.combinations(range(1, customer_count), i):
                problem += pulp.lpSum([x[(i, j, k)] for i in s for j in s for k in range(self.vehicle_count) if i != j]) <= len(s) - 1

        # Solve the problem
#        solver = pulp.PULP_CBC_CMD(msg=False)
#        problem.solve(solver)

        # Solve the problem
        solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=timeout_seconds)
        self.result_status = problem.solve(solver)

        # Check if a feasible solution was found
        if self.result_status == pulp.LpStatusOptimal or self.result_status == pulp.LpStatusNotSolved:
            print(f"A feasible solution was found for vehicle count: {self.vehicle_count}")
            # Extract and store the solution as before...
        elif self.result_status in [pulp.LpStatusNotSolved, pulp.LpStatusInfeasible, pulp.LpStatusUndefined]:
            # If no feasible solution was found
            print(f"A feasible solution could not be found for vehicle count: {self.vehicle_count}.")
        else:
            # If the solver status is unknown or an unexpected value
            print(f"Solver ended with status {self.result_status}, which may indicate a problem with the model or solver.")

        print('Extract solution')
        # Extract solution
        # 解を抽出する部分
        for k in range(self.vehicle_count):
            visited = set()  # 訪問済みの顧客を記録
            self.routes[k] = [depot_index]
            visited.add(depot_index)  # デポは最初から訪問済み
            next_location = depot_index

            while True:
                next_locations = [j for j in range(customer_count) if pulp.value(x[(next_location, j, k)]) == 1 and j not in visited]
                if not next_locations:  # 次に訪問する顧客がない場合、ループを終了
                    break
                next_location = next_locations[0]
                self.routes[k].append(next_location)
                visited.add(next_location)  # 訪問済みに追加

        print('Calculate distances for each vehicle')
        # Calculate distances for each vehicle
        for k, route in self.routes.items():
            self.distances[k] = sum([self.cost[route[i]][route[i+1]] for i in range(len(route)-1)])

    def create_routes_map(self):
        map_center = [self.depot_latitude, self.depot_longitude]
        m = folium.Map(location=map_center, zoom_start=12)
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue']

        # Add depot marker
        folium.Marker(map_center, popup='Depot', icon=folium.Icon(color='red', icon='home')).add_to(m)

        # Add markers and routes for each vehicle
        for k, route in self.routes.items():
            route_color = colors[k % len(colors)]
            for i in range(len(route)):
                location = [self.cluster_df.loc[route[i], 'latitude'], self.cluster_df.loc[route[i], 'longitude']]
                if i == 0:
                    folium.Marker(location, popup=f'Vehicle {k} Start', icon=folium.Icon(color=route_color)).add_to(m)
                elif i == len(route) - 1:
                    folium.Marker(location, popup=f'Vehicle {k} End', icon=folium.Icon(color=route_color)).add_to(m)
                else:
                    folium.Marker(location, popup=f'Customer {route[i]}', icon=folium.Icon(color=route_color, icon='user')).add_to(m)
                
                if i < len(route) - 1:
                    next_location = [self.cluster_df.loc[route[i+1], 'latitude'], self.cluster_df.loc[route[i+1], 'longitude']]
                    folium.PolyLine(locations=[location, next_location], color=route_color, weight=2.5, opacity=1).add_to(m)

            # デポに戻るルートを追加するためのコード
            if len(route) > 1:  # ルートがデポのみでない場合
                last_customer_location = [self.cluster_df.loc[route[-1], 'latitude'], self.cluster_df.loc[route[-1], 'longitude']]
                folium.PolyLine(locations=[last_customer_location, map_center], color=route_color, weight=2.5, opacity=1).add_to(m)
                folium.Marker(map_center, popup=f'Vehicle {k} End', icon=folium.Icon(color=route_color)).add_to(m)


        # Save the map
        m.save(f'map_route_cluster_{self.cluster_id}.html')

    def print_solution(self, speed_kmh=30.0):
        print("Vehicle Routing Solution:")
        total_distance = 0
        total_time = 0  # 移動時間の合計を保持する変数
        speed_m_s = speed_kmh / 3.6  # 時速をメートル/秒に変換

        for k, route in self.routes.items():
            if len(route) > 2:  # 実際に顧客を訪問するルートのみを考慮
                route_distance = self.distances[k]
                route_time = route_distance / speed_m_s  # 移動時間を計算
                total_distance += route_distance
                total_time += route_time  # 移動時間を合計時間に加算

                print(f"  Vehicle {k+1}: Route: {route}")
                print(f"    Distance: {route_distance:.2f} meters")
                print(f"    Time: {route_time/3600:.2f} hours")  # 秒を時間に変換して表示

        print(f"Total distance covered by all vehicles: {total_distance:.2f} meters")
        print(f"Total time spent: {total_time/3600:.2f} hours")  # 秒を時間に変換して表示

    def save_solution_to_csv(self, filename="solution_cluster_{cluster_id}.csv", speed_kmh=30.0):
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss

        # 解の計算を含む別のメソッドをここで呼び出し
        # self.print_solution(speed_kmh)

        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss

        # 計算結果とパフォーマンス指標をDataFrameにまとめる
        data = {
            "Vehicle ID": [],
            "Route": [],
            "Distance (m)": [],
            "Time (h)": [],
            "Calculation Time (s)": None,
            "Memory Usage (bytes)": None,
            "Speed (km/h)": speed_kmh,
            "Total Distance (m)": None,
            "Total Time (h)": None,
            "Date Time": None,
            "Cluster ID": self.cluster_id
        }

        total_distance = 0
        total_time = 0  # 秒

        speed_m_s = speed_kmh / 3.6

        for k, route in self.routes.items():
            if len(route) > 2:
                route_distance = self.distances[k]
                route_time = route_distance / speed_m_s
                total_distance += route_distance
                total_time += route_time
                data["Vehicle ID"].append(k + 1)
                data["Route"].append(" -> ".join(map(str, route)))
                data["Distance (m)"].append(route_distance)
                data["Time (h)"].append(route_time / 3600)

        df = pd.DataFrame(data)

        # パフォーマンス指標と総計をDataFrameに追加
        calculation_time = end_time - start_time
        memory_usage = end_memory - start_memory
        total_time_hours = total_time / 3600
        datetime_now = pd.Timestamp.now().strftime('%y/%m/%d')

        # PuLPのステータスコードを文字列に変換
        status_dict = {
            pulp.LpStatusOptimal: "Optimal",
            pulp.LpStatusNotSolved: "Not Solved",
            pulp.LpStatusInfeasible: "Infeasible",
            pulp.LpStatusUnbounded: "Unbounded",
            pulp.LpStatusUndefined: "Undefined"
        }
        # ステータスコードに対応する文字列を取得（見つからない場合は"Unknown"を返す）
        solution_status_str = status_dict.get(self.result_status, "Unknown")

        # データを追加
        df.loc[0, "Calculation Time (s)"] = calculation_time
        df.loc[0, "Memory Usage (bytes)"] = memory_usage
        df.loc[0, "Total Distance (m)"] = total_distance
        df.loc[0, "Total Time (h)"] = total_time_hours
        df.loc[0, "Date Time"] = datetime_now
        df.loc[0, "Solver Status"] = solution_status_str

        # フォーマットされたファイル名
        formatted_filename = filename.format(cluster_id=self.cluster_id)
        # CSVに保存
        df.to_csv(formatted_filename, index=False)


    def run(self):
        self.load_and_prepare_data()
        self.perform_clustering()
        self.prepare_depot()
        self.extract_cluster()
        self.calculate_distance_matrix()
        self.solve_vehicle_routing_problem()
        self.print_solution()
        self.save_solution_to_csv()
        self.create_routes_map()

# Usage
if __name__ == "__main__":
    vehicle_count = 10
    vehicle_capacity = 4
    n_clusters = 38
    depot_latitude = 34.638221
    depot_longitude = 138.128204
    file_name = 'peoplelist.csv'
    cluster_id = 6

    vrp = ClusterBasedVehicleRouting(vehicle_count, vehicle_capacity, n_clusters, depot_latitude, depot_longitude, file_name, cluster_id)  # Update the file_name accordingly
    vrp.run()
