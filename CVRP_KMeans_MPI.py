from mpi4py import MPI
import sys
from CVRP_KMeans_Class import ClusterBasedVehicleRouting

if __name__ == "__main__":
    # MPIの初期化
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # 現在のプロセス番号
    size = comm.Get_size()  # 総プロセス数

    # 共通のパラメータ
    vehicle_count = 10
    vehicle_capacity = 4
    n_clusters = 38
    depot_latitude = 34.638221
    depot_longitude = 138.128204
    file_name = 'peoplelist.csv'  # 適切なファイル名に更新してください

    # クラスタIDをプロセスに割り当てる
    for cluster_id in range(rank, n_clusters, size):
        # ClusterBasedVehicleRouting インスタンスの作成と実行
        vrp = ClusterBasedVehicleRouting(vehicle_count, vehicle_capacity, n_clusters, depot_latitude, depot_longitude, file_name, cluster_id)
        vrp.run()

    # MPIの終了
    MPI.Finalize()

    # mpirun -n 4 python3 CVRP_KMeans_MPI.py

