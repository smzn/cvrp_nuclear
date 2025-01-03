import random
import numpy as np
import csv
import matplotlib.pyplot as plt

class CVRP_Setup:
    def __init__(self, num_clients, num_shelters, num_vehicles, demand_options, vehicle_capacity, area_size, min_distance, speed):
        """
        CVRPの初期設定を行うクラス。
        """
        self.num_clients = num_clients
        self.num_shelters = num_shelters
        self.num_vehicles = num_vehicles
        self.demand_options = demand_options
        self.vehicle_capacity = vehicle_capacity
        self.area_size = area_size
        self.min_distance = min_distance
        self.speed = speed

        # 初期化時に nodes と vehicles を None に設定
        self.nodes = None
        self.vehicles = None

    def generate_nodes_and_vehicles(self, node_file="./init/nodes.csv", vehicle_file="./init/vehicles.csv"):
        """
        ノードと車両をランダム生成し、CSVに保存。
        """
        def generate_positions(num_points, area_size, min_distance):
            positions = []
            while len(positions) < num_points:
                x, y = random.uniform(0, area_size), random.uniform(0, area_size)
                if all(np.sqrt((x - px)**2 + (y - py)**2) >= min_distance for px, py in positions):
                    positions.append((x, y))
            return positions

        total_nodes = self.num_clients + self.num_shelters + 1
        positions = generate_positions(total_nodes, self.area_size, self.min_distance)

        # ノードの生成
        nodes = []
        for i, (x, y) in enumerate(positions):
            if i == 0:
                nodes.append({"id": i, "type": "city_hall", "x": x, "y": y, "demand": 0})
            elif i <= self.num_shelters:
                nodes.append({"id": i, "type": "shelter", "x": x, "y": y, "demand": 0})
            else:
                demand = random.choice(self.demand_options)
                nodes.append({"id": i, "type": "client", "x": x, "y": y, "demand": demand})

        # 車両の生成
        vehicles = [{"id": i, "capacity": self.vehicle_capacity} for i in range(1, self.num_vehicles + 1)]

        self.nodes = nodes
        self.vehicles = vehicles

        # CSV保存
        self._save_to_csv(nodes, node_file, ["id", "type", "x", "y", "demand"])
        self._save_to_csv(vehicles, vehicle_file, ["id", "capacity"])
        return nodes, vehicles

    def _save_to_csv(self, data, file_name, fieldnames):
        with open(file_name, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"Data saved to {file_name}")

    def calculate_cost_matrix(self, cost_matrix_file='./init/travel_time.csv'):
        """
        拠点の座標から移動時間のコスト行列を計算。
        """
        num_nodes = len(self.nodes)
        cost_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    dx = self.nodes[i]["x"] - self.nodes[j]["x"]
                    dy = self.nodes[i]["y"] - self.nodes[j]["y"]
                    distance = np.sqrt(dx**2 + dy**2)
                    travel_time = distance / self.speed * 60
                    cost_matrix[i, j] = travel_time
        np.savetxt(cost_matrix_file, cost_matrix, delimiter=",", fmt="%.2f")
        print(f"Cost matrix saved to {cost_matrix_file}")
        return cost_matrix
    
    def plot_nodes(self, map_file='./init/node_map.png'):
        """
        ノードを種類別にプロットし、PNG形式で保存。
        """
        plt.figure(figsize=(10, 10))
        cmap = plt.cm.viridis
        norm = plt.Normalize(1, max(node["demand"] for node in self.nodes if node["type"] == "client"))

        for node in self.nodes:
            if node["type"] == "city_hall":
                plt.scatter(node["x"], node["y"], color="blue", marker="s", s=100)
            elif node["type"] == "shelter":
                plt.scatter(node["x"], node["y"], color="green", marker="^", s=100)
            elif node["type"] == "client":
                color = cmap(norm(node["demand"]))
                plt.scatter(node["x"], node["y"], color=color, marker="o", s=100)

        # ノードの種類に対応する凡例
        type_legend_handles = [
            plt.Line2D([0], [0], color="blue", marker="s", linestyle="", markersize=10, label="City Hall"),
            plt.Line2D([0], [0], color="green", marker="^", linestyle="", markersize=10, label="Shelter"),
            plt.Line2D([0], [0], color="black", marker="o", linestyle="", markersize=10, label="Client"),
        ]

        type_legend = plt.legend(handles=type_legend_handles, title="Node Type", loc="upper left", bbox_to_anchor=(1.0, 1))
        plt.gca().add_artist(type_legend)

        # クライアント需要に対応する凡例
        unique_demands = sorted(set(node["demand"] for node in self.nodes if node["type"] == "client"))
        demand_legend_handles = [
            plt.Line2D([0], [0], color=cmap(norm(demand)), marker="o", linestyle="", markersize=10, label=f"{demand}")
            for demand in unique_demands
        ]
        plt.legend(handles=demand_legend_handles, title="Client Demand", loc="upper left", bbox_to_anchor=(1.0, 0.5))

        # 凡例と装飾
        plt.xlabel("X Coordinate (km)")
        plt.ylabel("Y Coordinate (km)")
        plt.title("Node Locations")
        plt.grid(True)
        plt.savefig(map_file, dpi=300)
        print(f"Node map saved to {map_file}")
        plt.close()
