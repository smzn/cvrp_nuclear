from CVRP_Setup_3d_v1 import CVRP_Setup_3d
from CVRP_Calculation_3d_v1 import CVRP_Calculation_3d
import time

# 初期設定
num_clients=300
num_shelters=20
num_vehicles=10
demand_options=(1, 2)
vehicle_capacity=4
area_size=20
min_distance=0.5
speed = 40 #車両の移動速度

# 時間計測開始
start_time = time.time()

setup = CVRP_Setup_3d(num_clients, num_shelters, num_vehicles, demand_options, vehicle_capacity, area_size, min_distance, speed)
nodes, vehicles = setup.generate_nodes_and_vehicles()
cost_matrix = setup.calculate_cost_matrix()
cost_matrix = setup.calculate_cost_matrix()
setup.plot_nodes()
setup.plot_nodes_3d()
setup.plot_contour_2d()
setup.plot_contour_3d()

# 遺伝アルゴリズムのパラメータ設定
population_size = 50
crossover_rate = 0.8
mutation_rate = 0.1
generations = 30
penalty = 1000

# CVRP_Calculation のインスタンス生成
calculation = CVRP_Calculation_3d(
    nodes=nodes,
    vehicles=vehicles,
    cost_matrix=cost_matrix,
    population_size=population_size,
    crossover_rate=crossover_rate,
    mutation_rate=mutation_rate,
    generations=generations,
    penalty=penalty
)

print("==== 初期設定パラメータ ====")
print(f"クライアント数: {num_clients}")
print(f"避難所数: {num_shelters}")
print(f"車両数: {num_vehicles}")
print(f"需要の選択肢: {demand_options}")
print(f"車両の容量: {vehicle_capacity}")
print(f"エリアサイズ: {area_size}")
print(f"最小距離: {min_distance}")
print(f"車両速度: {speed} km/h")
print()

print("==== 遺伝アルゴリズムの設定 ====")
print(f"集団サイズ: {population_size}")
print(f"交叉率: {crossover_rate}")
print(f"突然変異率: {mutation_rate}")
print(f"世代数: {generations}")
print(f"ペナルティ係数: {penalty}")

# 初期集団を生成
population = calculation.generate_initial_population(population_size)

# 遺伝アルゴリズムの実行
best_individual, best_fitness = calculation.run_genetic_algorithm(generations)

calculation.save_vehicle_statistics(best_individual)
calculation.visualize_routes(best_individual, nodes)
calculation.visualize_routes_3d(best_individual, nodes)
#calculation.plot_elevation_changes(best_individual, nodes, cost_matrix)

# 計算時間の表示
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\n計算時間: {elapsed_time:.2f} 秒")
