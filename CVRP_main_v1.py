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

# 遺伝アルゴリズムの実行
best_individual, best_fitness = calculation.run_genetic_algorithm(generations)

# 結果の表示
print(f"\n全世代を通じての最良適合度: {best_fitness}")
print("最良個体:")
for route in best_individual:
    print(f"  ルート: {route}")


'''
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

# 適応度リストを生成 (サンプル)
fitness_values = [calculation.evaluate_individual(ind) for ind in population]

# トーナメント選択のテスト
parent1 = calculation.select_parents(population, fitness_values)
parent2 = calculation.select_parents(population, fitness_values)

print("親個体1:")
for route in parent1:
    print(f"  ルート: {route}")

print("\n親個体2:")
for route in parent2:
    print(f"  ルート: {route}")

# サンプル親個体
parent1 = population[0][0]  # 最初の個体のルート1
parent2 = population[1][0]  # 2番目の個体のルート1

# 交叉のテスト
child1, child2 = calculation.crossover(parent1, parent2)
print("\n親個体1:", parent1)
print("親個体2:", parent2)
print("子個体1:", child1)
print("子個体2:", child2)

# サンプル個体
individual = population[0]  # 最初の個体

# 突然変異のテスト
print("\n突然変異前:")
for route in individual:
    print(f"  ルート: {route}")

calculation.mutate(individual)

print("\n突然変異後:")
for route in individual:
    print(f"  ルート: {route}")

for i, fitness in enumerate(fitness_values):
    print(f"個体 {i + 1}: 適応度 {fitness}")

# 次世代を作成
next_population = calculation.create_next_generation(population)

# 次世代の適応度を確認
print("\n次世代集団の適応度:")
fitness_values_next = [calculation.evaluate_individual(ind) for ind in next_population]
for i, fitness in enumerate(fitness_values_next):
    print(f"個体 {i + 1}: 適応度 {fitness}")
'''
    
