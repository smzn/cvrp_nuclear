import random
import numpy as np

class CVRP_Calculation:
    def __init__(self, nodes, vehicles, cost_matrix, population_size, crossover_rate, mutation_rate, generations, penalty):
        """
        CVRP計算クラスの初期化。
        :param nodes: ノード情報（辞書リスト）
        :param vehicles: 車両情報（辞書リスト）
        :param cost_matrix: 移動コスト行列（numpy.array）
        :param population_size: 集団サイズ
        :param crossover_rate: 交叉率
        :param mutation_rate: 突然変異率
        :param generations: 世代数
        :param penalty: ペナルティ係数
        """
        # ノード情報から属性を設定
        self.V = [node["id"] for node in nodes if node["type"] == "client"]  # クライアントノード
        self.H = [node["id"] for node in nodes if node["type"] == "shelter"]  # 避難所ノード
        self.d = {node["id"]: node["demand"] for node in nodes if node["type"] == "client"}  # 各クライアントの需要

        # 車両情報から属性を設定
        self.M = [vehicle["id"] for vehicle in vehicles]  # 車両のIDリスト
        self.Q = {vehicle["id"]: vehicle["capacity"] for vehicle in vehicles}  # 各車両の最大積載量

        # コスト行列
        self.c = cost_matrix

        # 辺集合を生成
        self.E = [(i, j) for i in range(len(nodes)) for j in range(len(nodes)) if i != j]

        # 遺伝アルゴリズムのパラメータ
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.penalty = penalty

    def evaluate_individual(self, individual):
        """
        目的関数: 制約確認 + 総移動コスト + ペナルティ
        :param individual: 個体 (遺伝子表現: 各車両のルートのリスト)
        :return: 総コスト (目的関数値) または大きなペナルティ値
        """
        # 制約確認
        if not self.check_constraints(individual):
            return float('inf')  # 制約違反の場合、大きなペナルティ値を返す

        # 制約を満たしている場合、コスト計算
        total_cost = 0
        vehicle_penalties = 0

        for vehicle_route in individual:
            if not vehicle_route:
                continue

            load = 0
            route_cost = 0
            previous_node = 0  # デポ (市役所) から出発

            for node in vehicle_route:
                if node == 0:  # デポは積載量に影響を与えない
                    continue

                load += self.d[node]  # 要支援者人数を加算
                if load > max(self.Q.values()):  # 最大積載量を超えた場合
                    vehicle_penalties += self.penalty

                # 移動コストを加算
                route_cost += self.c[previous_node, node]
                previous_node = node

            # デポに戻るコストを加算
            route_cost += self.c[previous_node, 0]

            total_cost += route_cost

        return total_cost + vehicle_penalties

    
    def generate_initial_population(self):
        """
        初期集団を生成し、制約を確認する。
        :return: 制約を満たす初期集団 (リスト)
        """
        population = []
        for _ in range(self.population_size):
            while True:
                # 要支援者をランダムな順序にシャッフル
                shuffled_clients = random.sample(self.V, len(self.V))
                individual = []  # 一つの個体（ルート群）

                current_vehicle_route = []  # 現在の車両のルート
                current_load = 0  # 現在の車両の積載量

                for client in shuffled_clients:
                    client_demand = self.d[client]
                    if current_load + client_demand <= max(self.Q.values()):  # 積載量制約を確認
                        current_vehicle_route.append(client)
                        current_load += client_demand
                    else:
                        # 制約を超えた場合、ルートを確定して次の車両に移る
                        individual.append([0] + current_vehicle_route + [0])  # デポを追加
                        current_vehicle_route = [client]
                        current_load = client_demand

                # 最後の車両のルートを追加
                if current_vehicle_route:
                    individual.append([0] + current_vehicle_route + [0])  # デポを追加

                # 制約確認: 制約を満たす場合のみ追加
                if self.check_constraints(individual):
                    population.append(individual)
                    break  # 有効な個体を生成した場合、次へ

        return population

    
    def check_constraints(self, individual):
        """
        個体が制約を満たしているか確認。
        :param individual: 個体 (遺伝子表現: 各車両のルートのリスト)
        :return: True (制約を満たす) または False (制約違反)
        """
        visited_clients = set()  # 訪問済みの要支援者

        for vehicle_route in individual:
            current_load = 0  # 現在の積載量

            for node in vehicle_route:
                if node == 0:  # デポ (市役所) は無視
                    continue

                # 同じ要支援者を複数回訪問している場合は制約違反
                if node in visited_clients:
                    return False

                visited_clients.add(node)
                current_load += self.d[node]  # 要支援者人数を加算

                # 積載量制約を超えた場合は制約違反
                if current_load > max(self.Q.values()):
                    return False

        # すべての要支援者が訪問されているか確認
        return visited_clients == set(self.V)



