import random
import numpy as np
import csv
import matplotlib.pyplot as plt
import copy

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
        # 個体の構造確認
        for ind in population:
            for route in ind:
                if not isinstance(route, list):
                    raise ValueError(f"不正なルート構造が発見されました: {route}")
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

    def select_parents(self, population, fitness_values, k=3):
        """
        トーナメント選択で親個体を選択。
        """
        selected = random.sample(range(len(population)), k)
        best = min(selected, key=lambda idx: fitness_values[idx])
        return population[best]

    def crossover(self, parent1, parent2):
        """
        部分順序交叉 (PMX) を実装。
        """
        child1 = [-1] * len(parent1)
        child2 = [-1] * len(parent2)
        start, end = sorted(random.sample(range(len(parent1)), 2))
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]

        pos = end
        for i in range(len(parent2)):
            idx = (end + i) % len(parent2)
            if parent2[idx] not in child1:
                child1[pos % len(child1)] = parent2[idx]
                pos += 1

        pos = end
        for i in range(len(parent1)):
            idx = (end + i) % len(parent1)
            if parent1[idx] not in child2:
                child2[pos % len(child2)] = parent1[idx]
                pos += 1

        # 子個体の構造確認
        if not isinstance(child1, list) or not isinstance(child2, list):
            raise ValueError(f"不正な交叉結果が発見されました: {child1}, {child2}")
        return child1, child2

    def mutate(self, individual, mutation_rate=0.1):
        """
        突然変異: ランダムな部分区間を逆順にする。
        :param individual: 個体 (リスト: 各車両のルート)
        :param mutation_rate: 突然変異率
        """
        for route_index, route in enumerate(individual):
            # ルートがリストでない場合、自動修正またはエラーを記録
            if not isinstance(route, list):
                print(f"警告: 予期しないデータ型 {type(route)} が発見されました。修正を試みます。")
                #individual[route_index] = [0, route, 0] if isinstance(route, int) else [0, 0]
                #route = individual[route_index]

            # 短いルートでは突然変異をスキップ
            if len(route) <= 3:
                continue

            if random.random() < mutation_rate:
                # ランダムな部分区間を逆順にする
                i, j = sorted(random.sample(range(1, len(route) - 1), 2))
                route[i:j] = reversed(route[i:j])


    def create_next_generation(self, population):
        """
        次世代を構築する。
        :param population: 現世代の集団 (リスト)
        :return: 次世代の集団 (リスト)
        """
        # 適応度を計算
        fitness_values = [self.evaluate_individual(ind) for ind in population]

        # エリート個体を保存 (最良個体)
        best_individual = copy.deepcopy(population[np.argmin(fitness_values)])
        best_fitness = min(fitness_values)

        # 次世代集団を構築
        next_generation = []

        # エリート個体を次世代に追加
        next_generation.append(best_individual)

        # 親選択、交叉、突然変異を繰り返して新しい個体を生成
        while len(next_generation) < self.population_size:
            # 親選択
            parent1 = self.select_parents(population, fitness_values)
            parent2 = self.select_parents(population, fitness_values)

            # 交叉
            if random.random() < self.crossover_rate:
                child1, child2 = self.crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]

            # 突然変異
            self.mutate(child1, self.mutation_rate)
            self.mutate(child2, self.mutation_rate)

            # 制約確認を追加
            if self.check_constraints(child1) and len(next_generation) < self.population_size:
                next_generation.append(child1)
            if self.check_constraints(child2) and len(next_generation) < self.population_size:
                next_generation.append(child2)

        return next_generation[:self.population_size]


    def run_genetic_algorithm(self, max_generations, output_csv='./result/genetic_results.csv', best_individual_csv='./result/best_individual.csv'):
        """
        遺伝アルゴリズムを実行し、結果を保存。
        :param max_generations: 世代数
        :param output_csv: 結果を保存するCSVファイルの名前
        :return: 全世代を通じての最良個体とその適合度
        """
        # 初期集団生成
        population = self.generate_initial_population()

        # CSVデータの準備
        results = []
        best_overall_individual = None
        best_overall_fitness = float('inf')

        for generation in range(max_generations):
            # 適応度の計算
            fitness_values = [self.evaluate_individual(ind) for ind in population]

            # 現世代のデータ
            best_fitness = min(fitness_values)
            mean_fitness = np.mean(fitness_values)
            std_fitness = np.std(fitness_values)

            # 最良個体の保存
            best_individual = population[np.argmin(fitness_values)]
            if best_fitness < best_overall_fitness:
                best_overall_fitness = best_fitness
                best_overall_individual = best_individual

            # 結果を記録
            results.append([generation + 1, best_fitness, mean_fitness, std_fitness])

            # 次世代を作成
            population = self.create_next_generation(population)

            # 現世代の進捗を表示
            print(f"世代 {generation + 1}: 最良適合度 = {best_fitness}, 平均適合度 = {mean_fitness}, 標準偏差 = {std_fitness}")

        # 結果をCSVに保存
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Generation', 'Best Fitness', 'Mean Fitness', 'Std Dev'])
            writer.writerows(results)

        # 最良個体をCSVに保存
        with open(best_individual_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Vehicle', 'Route', 'Fitness'])
            for i, route in enumerate(best_overall_individual):
                writer.writerow([f"Vehicle {i + 1}", ' -> '.join(map(str, route)), ""])
            writer.writerow([])
            writer.writerow(['Best Fitness', best_overall_fitness])

        print(f"\n全世代を通しての最良適合度: {best_overall_fitness}")
        print("最良個体:")
        for route in best_overall_individual:
            print(f"  ルート: {route}")

        # グラフの作成
        self.plot_results(results, output_csv.replace('.csv', '.png'))

        return best_overall_individual, best_overall_fitness

    def plot_results(self, results, output_image='./result/genetic_results.png'):
        """
        遺伝アルゴリズムの結果をプロット。
        :param results: 世代ごとの結果データ (リスト)
        :param output_image: グラフを保存するファイル名
        """
        generations = [row[0] for row in results]
        best_fitness = [row[1] for row in results]
        mean_fitness = [row[2] for row in results]

        plt.figure(figsize=(10, 6))
        plt.plot(generations, best_fitness, label="Best Fitness", marker='o')
        plt.plot(generations, mean_fitness, label="Mean Fitness", marker='x')
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Genetic Algorithm Progress")
        plt.legend()
        plt.grid(True)
        plt.savefig(output_image)
        #plt.show()
        print(f"結果のグラフを保存しました: {output_image}")



