"""数据处理工具模块 - 提供数据加载和图构建的辅助函数"""
import scipy.sparse as sp
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
import torch


def load_rating_file_to_list(filename):
    """
    将评分文件加载为列表格式
    
    Args:
        filename (str): 评分文件路径，每行格式为 "用户ID 物品ID"
        
    Returns:
        list: 包含[用户ID, 物品ID]对的列表
        
    Example:
        >>> ratings = load_rating_file_to_list("userRatingTrain.txt")
        >>> print(ratings[:3])  # [[0, 5], [0, 10], [1, 3]]
    """
    rating_list = []
    lines = open(filename, 'r').readlines()

    for line in lines:
        contents = line.split()
        # Each line: user item
        rating_list.append([int(contents[0]), int(contents[1])])
    return rating_list


def load_rating_file_to_matrix(filename, num_users=None, num_items=None):
    """
    将评分文件加载为稀疏矩阵格式
    
    Args:
        filename (str): 评分文件路径
        num_users (int, optional): 用户数量，如果为None则自动计算
        num_items (int, optional): 物品数量，如果为None则自动计算
        
    Returns:
        scipy.sparse.dok_matrix: 用户-物品交互稀疏矩阵，1表示有交互，0表示无交互
        
    Note:
        支持两种文件格式:
        1. "用户ID 物品ID" (隐式反馈)
        2. "用户ID 物品ID 评分" (显式反馈，评分>0才记录为交互)
    """
    if num_users is None:
        num_users, num_items = 0, 0

    lines = open(filename, 'r').readlines()
    for line in lines:
        contents = line.split()
        u, i = int(contents[0]), int(contents[1])
        num_users = max(num_users, u)
        num_items = max(num_items, i)

    mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
    for line in lines:
        contents = line.split()
        if len(contents) > 2:
            u, i, rating = int(contents[0]), int(contents[1]), int(contents[2])
            if rating > 0:
                mat[u, i] = rating
        else:
            u, i = int(contents[0]), int(contents[1])
            mat[u, i] = 1.0
    return mat


def load_negative_file(filename):
    """
    加载负样本文件
    
    Args:
        filename (str): 负样本文件路径，每行格式为 "(用户ID,正样本ID) 负样本ID1 负样本ID2 ..."
        
    Returns:
        list: 每个元素是一个负样本ID列表，对应每个测试样本的负样本
        
    Example:
        文件内容: "(0,5) 1 3 7 9"
        返回: [[1, 3, 7, 9], ...]
    """
    negative_list = []

    lines = open(filename, 'r').readlines()

    for line in lines:
        negatives = line.split()[1:]
        negatives = [int(neg_item) for neg_item in negatives]
        negative_list.append(negatives)
    return negative_list


def load_group_member_to_dict(user_in_group_path):
    """
    加载群组成员映射关系
    
    Args:
        user_in_group_path (str): 群组成员文件路径，每行格式为 "群组ID 成员ID1,成员ID2,..."
        
    Returns:
        defaultdict(list): 群组ID到成员ID列表的映射字典
        
    Example:
        文件内容: "0 1,3,5"
        返回: {0: [1, 3, 5], 1: [...], ...}
    """
    group_member_dict = defaultdict(list)
    lines = open(user_in_group_path, 'r').readlines()

    for line in lines:
        contents = line.split()
        group = int(contents[0])
        for member in contents[1].split(','):
            group_member_dict[group].append(int(member))
    return group_member_dict


def build_group_graph(group_data, num_groups):
    """
    构建群组级别的重叠图
    
    Args:
        group_data (list): 每个群组包含的节点(用户+物品)列表
        num_groups (int): 群组总数
        
    Returns:
        numpy.ndarray: 归一化的群组邻接矩阵 D^(-1) * A
        
    Note:
        权重计算公式: weight = |交集| / |并集| (Jaccard相似度)
        对角线元素为1.0(自连接)
        最终返回度归一化后的邻接矩阵
    """
    matrix = np.zeros((num_groups, num_groups))

    for i in range(num_groups):
        group_a = set(group_data[i])
        for j in range(i + 1, num_groups):
            group_b = set(group_data[j])
            overlap = group_a & group_b
            union = group_a | group_b
            # weight computation
            matrix[i][j] = float(len(overlap) / len(union))
            matrix[j][i] = matrix[i][j]

    matrix = matrix + np.diag([1.0] * num_groups)
    degree = np.sum(np.array(matrix), 1)
    # \mathbf{D}^{-1} \dot \mathbf{A}
    return np.dot(np.diag(1.0 / degree), matrix)


def build_hyper_graph(group_member_dict, group_train_path, num_users, num_items, num_groups, group_item_dict=None):
    """
    构建成员级别的超图结构
    
    Args:
        group_member_dict (dict): 群组到成员的映射
        group_train_path (str): 群组训练数据文件路径
        num_users (int): 用户总数
        num_items (int): 物品总数  
        num_groups (int): 群组总数
        group_item_dict (dict, optional): 群组到物品的映射，如果为None则从文件构建
        
    Returns:
        tuple: (用户超图, 物品超图, 完整超图, 群组数据)
            - 用户超图: 用户通过群组超边连接的归一化超图
            - 物品超图: 物品通过群组超边连接的归一化超图  
            - 完整超图: 用户和物品统一的超图结构
            - 群组数据: 每个群组包含的所有节点列表
            
    Note:
        超图中每个群组都是一个超边，连接该群组中的所有用户和物品
        返回的超图已经过度归一化处理，可直接用于超图卷积计算
    """
    # Construct group-to-item-list mapping
    if group_item_dict is None:
        group_item_dict = defaultdict(list)

        for line in open(group_train_path, 'r').readlines():
            contents = line.split()
            if len(contents) > 2:
                group, item, rating = int(contents[0]), int(contents[1]), int(contents[2])
                if rating > 0:
                    group_item_dict[group].append(item)
            else:
                group, item = int(contents[0]), int(contents[1])
                group_item_dict[group].append(item)

    def _prepare(group_dict, rows, axis=0):
        """
        构建超图的内部辅助函数
        
        Args:
            group_dict (dict): 群组到节点的映射
            rows (int): 节点总数
            axis (int): 求和轴，0表示按群组求和，1表示按节点求和
            
        Returns:
            tuple: (超图邻接矩阵, 度归一化矩阵)
        """
        nodes, groups = [], []

        for group_id in range(num_groups):
            # groups.extend([group_id] * len(group_dict[group_id]))
            # nodes.extend(group_dict[group_id])
            # 过滤超出范围的节点ID
            valid_nodes = [node for node in group_dict[group_id] if node < rows]
            groups.extend([group_id] * len(valid_nodes))
            nodes.extend(valid_nodes)

        # csr_matrix构建超图邻接矩阵
        # 示例：
        #        群组0  群组1  群组2
        # 用户10   1     0     0      # 用户10只属于群组0
        # 用户20   1     1     0      # 用户20属于群组0和群组1
        # 用户30   1     0     1      # 用户30属于群组0和群组2  
        # 用户40   0     1     1      # 用户40属于群组1和群组2
        # 用户50   0     0     1      # 用户50只属于群组2
        hyper_graph = csr_matrix((np.ones(len(nodes)), (nodes, groups)), shape=(rows, num_groups))
        hyper_deg = np.array(hyper_graph.sum(axis=axis)).squeeze()
        hyper_deg[hyper_deg == 0.] = 1
        hyper_deg = sp.diags(1.0 / hyper_deg)
        return hyper_graph, hyper_deg

    # Two separate hypergraphs (user_hypergraph, item_hypergraph for hypergraph convolution computation)
    user_hg, user_hg_deg = _prepare(group_member_dict, num_users)
    item_hg, item_hg_deg = _prepare(group_item_dict, num_items)

    for group_id, items in group_item_dict.items():
        group_item_dict[group_id] = [item + num_users for item in items]
    group_data = [group_member_dict[group_id] + group_item_dict[group_id] for group_id in range(num_groups)]
    full_hg, hg_dg = _prepare(group_data, num_users + num_items, axis=1)

    user_hyper_graph = torch.sparse.mm(convert_sp_mat_to_sp_tensor(user_hg_deg),
                                       convert_sp_mat_to_sp_tensor(user_hg).t())
    item_hyper_graph = torch.sparse.mm(convert_sp_mat_to_sp_tensor(item_hg_deg),
                                       convert_sp_mat_to_sp_tensor(item_hg).t())
    full_hyper_graph = torch.sparse.mm(convert_sp_mat_to_sp_tensor(hg_dg), convert_sp_mat_to_sp_tensor(full_hg))
    print(
        f"User hyper-graph {user_hyper_graph.shape}, Item hyper-graph {item_hyper_graph.shape}, Full hyper-graph {full_hyper_graph.shape}")

    return user_hyper_graph, item_hyper_graph, full_hyper_graph, group_data


def convert_sp_mat_to_sp_tensor(x):
    """
    将scipy稀疏矩阵转换为PyTorch稀疏张量
    
    Args:
        x (scipy.sparse matrix): 输入的scipy稀疏矩阵
        
    Returns:
        torch.sparse.FloatTensor: 转换后的PyTorch稀疏张量
        
    Note:
        通过COO格式进行转换，保持稀疏性以节省内存
    """
    coo = x.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))


def build_light_gcn_graph(group_item_net, num_groups, num_items):
    """
    构建物品级别的二分图 (用于LightGCN)
    
    Args:
        group_item_net (scipy.sparse matrix): 群组-物品交互矩阵
        num_groups (int): 群组总数
        num_items (int): 物品总数
        
    Returns:
        torch.sparse.FloatTensor: 归一化的群组-物品二分图邻接矩阵
        
    Note:
        构建形式: [   0    R  ]
                [  R^T   0  ]
        其中R是群组-物品交互矩阵
        使用对称归一化: D^(-1/2) * A * D^(-1/2)
    """
    adj_mat = sp.dok_matrix((num_groups + num_items, num_groups + num_items), dtype=np.float32)
    adj_mat = adj_mat.tolil()

    R = group_item_net.tolil()
    adj_mat[:num_groups, num_groups:] = R
    adj_mat[num_groups:, :num_groups] = R.T
    adj_mat = adj_mat.todok()
    # print(adj_mat, adj_mat.shape)

    row_sum = np.array(adj_mat.sum(axis=1))
    d_inv = np.power(row_sum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    # print(d_mat)

    norm_adj = d_mat.dot(adj_mat)
    norm_adj = norm_adj.dot(d_mat)
    norm_adj = norm_adj.tocsr()
    graph = convert_sp_mat_to_sp_tensor(norm_adj)
    return graph.coalesce()


# Test code
# if __name__ == "__main__":
#     g_m_d = {0: [0, 1, 2], 1: [2, 3], 2: [4, 5, 6]}
#     g_i_d = {0: [0, 1], 1: [1, 2], 2: [3]}
#     user_g, item_g, hg, g_data = build_hyper_graph(g_m_d, "", 7, 4, 3, g_i_d)
#
#     print(user_g)
#     print(item_g)
#     print(hg)
#     print()
#     g = build_group_graph(g_data, 3)
#     print(g)
