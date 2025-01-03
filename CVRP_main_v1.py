from CVRP_Setup_v1 import CVRP_Setup
from CVRP_Calculation_v1 import CVRP_Calculation

# 初期設定
num_clients=100
num_shelters=10
num_vehicles=5
demand_options=(1, 2)
vehicle_capacity=4
area_size=20
min_distance=0.5
speed = 40 #車両の移動速度

setup = CVRP_Setup(num_clients, num_shelters, num_vehicles, demand_options, vehicle_capacity, area_size, min_distance, speed)
nodes, vehicles = setup.generate_nodes_and_vehicles()
cost_matrix = setup.calculate_cost_matrix()
setup.plot_nodes()

# 遺伝アルゴリズムのパラメータ設定
population_size = 50
crossover_rate = 0.8
mutation_rate = 0.1
generations = 100
penalty = 1000

# CVRP_Calculation のインスタンス生成
calculation = CVRP_Calculation(
    nodes=nodes,
    vehicles=vehicles,
    cost_matrix=cost_matrix,
    population_size=population_size,
    crossover_rate=crossover_rate,
    mutation_rate=mutation_rate,
    generations=generations,
    penalty=penalty
)

# 初期化確認の出力 (オプション)
print(f"クライアント数: {len(calculation.V)}")
print(f"避難所数: {len(calculation.H)}")
print(f"車両数: {len(calculation.M)}")
print(f"移動コスト行列のサイズ: {calculation.c.shape}")


# 初期集団を生成
population = calculation.generate_initial_population()

# 初期集団の各個体を評価
print("初期集団の評価結果:")
for i, individual in enumerate(population):
    evaluation = calculation.evaluate_individual(individual)
    print(f"個体 {i + 1} の評価値: {evaluation}")
    if evaluation == float('inf'):
        print(f"  ⚠️ 個体 {i + 1} は制約を満たしていません！")
    else:
        print(f"  ✅ 個体 {i + 1} は制約を満たしています。")
    print("-" * 50)

# サンプルとして最初の個体を詳細表示
print("\nサンプル個体のルート:")
for route in population[0]:
    print(f"  ルート: {route}")
