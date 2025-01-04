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
population = calculation.generate_initial_population(population_size)

# 遺伝アルゴリズムの実行
best_individual, best_fitness = calculation.run_genetic_algorithm(generations)


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

# 適応度を計算
print("\n適応度を計算中...")
fitness_values = [calculation.evaluate_individual(ind) for ind in population]
for idx, fitness in enumerate(fitness_values):
    print(f"個体 {idx + 1} の適応度: {fitness}")

# 選択関数をテスト
print("\nトーナメント選択テスト:")
k = 3  # トーナメントサイズ
selected_parent = calculation.select_parents(population, fitness_values, k=k)
print(f"選択された親個体（k={k}）: {selected_parent}")

# 親個体を選択
print("\n親個体の選択中...")
parent1 = calculation.select_parents(population, fitness_values, k=3)
parent2 = calculation.select_parents(population, fitness_values, k=3)
print(f"親個体 1: {parent1}")
print(f"親個体 2: {parent2}")

# 交叉を実行
print("\n交叉を実行中...")
child1, child2 = calculation.crossover(parent1, parent2)
print(f"子個体 1: {child1}")
print(f"子個体 2: {child2}")

# 子個体の適応度を確認
print("\n子個体の適応度を計算中...")
fitness_child1 = calculation.evaluate_individual(child1)
fitness_child2 = calculation.evaluate_individual(child2)
print(f"子個体 1 の適応度: {fitness_child1}")
print(f"子個体 2 の適応度: {fitness_child2}")

# 突然変異の適用
print("\n突然変異を適用中...")
mutated_population = []
for idx, individual in enumerate(population):
    print(f"\n個体 {idx + 1} の突然変異前: {individual}")
    calculation.mutate(individual, mutation_rate=0.1)  # 突然変異率10%
    print(f"個体 {idx + 1} の突然変異後: {individual}")
    mutated_population.append(individual)

# 突然変異後の適応度を確認
print("\n突然変異後の適応度を計算中...")
for idx, individual in enumerate(mutated_population):
    fitness = calculation.evaluate_individual(individual)
    print(f"個体 {idx + 1} の適応度: {fitness}")
'''
