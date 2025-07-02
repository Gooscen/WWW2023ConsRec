"""
演示超图构建过程的具体计算
"""
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sp

def demonstrate_hypergraph_construction():
    """演示具体的超图构建过程"""
    
    # 给定的群组字典
    group_dict = {
        0: [10, 20, 30],  # 群组0包含用户10,20,30
        1: [20, 40],      # 群组1包含用户20,40
        2: [30, 40, 50]   # 群组2包含用户30,40,50
    }
    
    num_groups = 3
    rows = 51  # 因为最大用户ID是50，所以需要51行（ID从0开始）
    
    print("=== 超图构建演示 ===")
    print(f"输入群组字典: {group_dict}")
    print(f"节点总数: {rows}, 群组总数: {num_groups}")
    print()
    
    # 步骤1: 构建节点和群组列表
    nodes, groups = [], []
    
    print("步骤1: 构建连接列表")
    for group_id in range(num_groups):
        valid_nodes = [node for node in group_dict[group_id] if node < rows]
        groups.extend([group_id] * len(valid_nodes))
        nodes.extend(valid_nodes)
        
        print(f"  群组{group_id}: 节点{valid_nodes} -> 添加到连接列表")
    
    print(f"最终节点列表: {nodes}")
    print(f"最终群组列表: {groups}")
    print()
    
    # 步骤2: 构建超图邻接矩阵
    print("步骤2: 构建超图邻接矩阵")
    hyper_graph = csr_matrix((np.ones(len(nodes)), (nodes, groups)), shape=(rows, num_groups))
    
    print(f"矩阵形状: {hyper_graph.shape}")
    print("非零元素的坐标和值:")
    for i, (row, col) in enumerate(zip(nodes, groups)):
        print(f"  ({row}, {col}) = 1")
    print()
    
    # 转换为密集矩阵以便查看
    dense_matrix = hyper_graph.toarray()
    
    print("完整的超图邻接矩阵 (只显示非零行):")
    print("行索引\t群组0\t群组1\t群组2")
    print("-" * 35)
    
    for i in range(rows):
        if np.any(dense_matrix[i] > 0):  # 只显示有连接的行
            print(f"节点{i:2d}\t{int(dense_matrix[i, 0])}\t{int(dense_matrix[i, 1])}\t{int(dense_matrix[i, 2])}")
    print()
    
    # 步骤3: 计算度归一化 (axis=0的情况)
    print("步骤3: 度归一化计算 (axis=0, 按列求和)")
    hyper_deg_axis0 = np.array(hyper_graph.sum(axis=0)).squeeze()
    print(f"按列求和 (每个群组的大小): {hyper_deg_axis0}")
    
    hyper_deg_axis0[hyper_deg_axis0 == 0.] = 1
    deg_matrix_axis0 = sp.diags(1.0 / hyper_deg_axis0)
    print(f"度的倒数: {1.0 / hyper_deg_axis0}")
    print(f"度归一化矩阵 (对角线): {deg_matrix_axis0.diagonal()}")
    print()
    
    # 步骤4: 计算度归一化 (axis=1的情况)
    print("步骤4: 度归一化计算 (axis=1, 按行求和)")
    hyper_deg_axis1 = np.array(hyper_graph.sum(axis=1)).squeeze()
    print("按行求和 (每个节点的度):")
    
    for i in range(rows):
        if hyper_deg_axis1[i] > 0:  # 只显示有连接的节点
            print(f"  节点{i:2d}: {int(hyper_deg_axis1[i])} 个群组")
    
    hyper_deg_axis1[hyper_deg_axis1 == 0.] = 1
    deg_matrix_axis1 = sp.diags(1.0 / hyper_deg_axis1)
    print(f"\n所有节点的度倒数 (前10个): {(1.0 / hyper_deg_axis1)[:10]}")
    print()
    
    # 步骤5: 展示超图的含义
    print("步骤5: 超图的含义解释")
    print("这个超图表示:")
    for group_id, members in group_dict.items():
        print(f"  超边{group_id}: 连接节点 {members}")
    
    print("\n节点的连接情况:")
    unique_nodes = sorted(set(nodes))
    for node in unique_nodes:
        connected_groups = [g for n, g in zip(nodes, groups) if n == node]
        print(f"  节点{node}: 属于群组 {connected_groups}")
    
    return hyper_graph, deg_matrix_axis0, deg_matrix_axis1

if __name__ == "__main__":
    hyper_graph, deg_axis0, deg_axis1 = demonstrate_hypergraph_construction() 