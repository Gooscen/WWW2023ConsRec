"""推荐系统评估指标模块 - 提供Hit@K和NDCG@K等评估指标的计算"""
import torch
import numpy as np
import math


def get_hit_k(pred_rank, k):
    """
    计算Hit@K指标
    
    Args:
        pred_rank (numpy.ndarray): 预测排序矩阵，形状为(用户数, 候选物品数)
                                  每行表示一个用户的物品排序，0表示真实正样本的位置
        k (int): 计算Top-K推荐的K值
        
    Returns:
        float: Hit@K值，范围[0, 1]，值越高表示推荐效果越好
        
    Note:
        Hit@K = 命中用户数 / 总用户数
        如果真实物品在Top-K推荐列表中，则认为命中
        
    Example:
        >>> pred_rank = np.array([[0, 1, 2], [2, 0, 1]])  # 2个用户，3个候选物品
        >>> hit_5 = get_hit_k(pred_rank, 5)  # 两个用户的真实物品都在Top-5中
        >>> print(hit_5)  # 1.0
    """
    pred_rank_k = pred_rank[:, :k]
    hit = np.count_nonzero(pred_rank_k == 0)
    hit = hit / pred_rank.shape[0]
    return round(hit, 5)


def get_ndcg_k(pred_rank, k):
    """
    计算NDCG@K指标 (Normalized Discounted Cumulative Gain)
    
    Args:
        pred_rank (numpy.ndarray): 预测排序矩阵，形状为(用户数, 候选物品数)
                                  每行表示一个用户的物品排序，0表示真实正样本的位置
        k (int): 计算Top-K推荐的K值
        
    Returns:
        float: NDCG@K值，范围[0, 1]，值越高表示推荐效果越好
        
    Note:
        NDCG考虑推荐位置的重要性，排序越靠前的正确推荐获得更高的分数
        计算公式: DCG = log(2) / log(位置+2)  (当真实物品在位置i时)
        由于只有一个正样本，IDCG = log(2) / log(2) = 1，所以NDCG = DCG
        
    Example:
        >>> pred_rank = np.array([[0, 1, 2], [1, 0, 2]])  # 用户0的真实物品在位置0，用户1的在位置1
        >>> ndcg_5 = get_ndcg_k(pred_rank, 5)
    """
    ndcgs = np.zeros(pred_rank.shape[0])
    for user in range(pred_rank.shape[0]):
        for j in range(k):
            if pred_rank[user][j] == 0:
                ndcgs[user] = math.log(2) / math.log(j+2)
    return np.round(np.mean(ndcgs), decimals=5)


def evaluate(model, test_ratings, test_negatives, device, k_list, type_m='group'):
    """
    评估推荐模型的性能
    
    Args:
        model (torch.nn.Module): 待评估的推荐模型
        test_ratings (list): 测试评分列表，每个元素为[实体ID, 物品ID]
        test_negatives (list): 测试负样本列表，每个元素为负样本物品ID列表
        device (torch.device): 计算设备 (CPU或GPU)
        k_list (list): 要计算的K值列表，如[1, 5, 10]
        type_m (str): 评估类型，'group'表示群组推荐，'user'表示用户推荐
        
    Returns:
        tuple: (Hit@K列表, NDCG@K列表)
            - Hit@K列表: 对应k_list中每个K值的Hit@K结果
            - NDCG@K列表: 对应k_list中每个K值的NDCG@K结果
            
    Note:
        评估流程:
        1. 将每个测试样本的真实物品放在候选列表首位
        2. 添加对应的负样本作为候选物品
        3. 使用模型预测所有候选物品的分数
        4. 根据预测分数排序，计算Hit@K和NDCG@K
        
    Warning:
        注释中提到的另一种方法是将真实物品放在最后，此时需要修改评估逻辑
        当前实现假设真实物品在首位(索引0)
    """
    model.eval()
    hits, ndcgs = [], []
    user_test, item_test = [], []

    for idx in range(len(test_ratings)):
        rating = test_ratings[idx]
        # Important
        # for testing, we put the ground-truth item as the first one and remaining are negative samples
        # for evaluation, we check whether prediction's idx is the ground-truth (idx with 0)
        items = [rating[1]]
        items.extend(test_negatives[idx])

        # an alternative
        # to avoid the dead relu issue where model predicts all candidate items with score 1.0 and thus lead to invalid predictions
        # we can put the ground-truth item to the last 
        # for evaluation, the checked ground-truth idx should be 100 in Line 17 & Line 8
        # items = test_negatives[idx] + [rating[1]]

        item_test.append(items)
        user_test.append(np.full(len(items), rating[0]))

    users_var = torch.LongTensor(user_test).to(device)
    items_var = torch.LongTensor(item_test).to(device)

    bsz = len(test_ratings)
    item_len = len(test_negatives[0]) + 1

    users_var = users_var.view(-1)
    items_var = items_var.view(-1)

    if type_m == 'group':
        predictions = model(users_var, None, items_var)
    elif type_m == 'user':
        predictions = model(None, users_var, items_var)

    predictions = torch.reshape(predictions, (bsz, item_len))

    pred_score = predictions.data.cpu().numpy()
    # print(pred_score[:10, ])
    pred_rank = np.argsort(pred_score * -1, axis=1)
    for k in k_list:
        hits.append(get_hit_k(pred_rank, k))
        ndcgs.append(get_ndcg_k(pred_rank, k))

    return hits, ndcgs


def calculate_hyperedge_coverage_centrality(hypergraph, alpha=0.5):
    """
    计算超边覆盖度中心性（HCC）指标
    
    Args:
        hypergraph (torch.sparse.FloatTensor or scipy.sparse matrix): 超图邻接矩阵
                                                                     形状为 (节点数, 超边数)
        alpha (float): 调节参数，用于平衡直接连接和间接覆盖的重要性，默认0.5
        
    Returns:
        numpy.ndarray: 每个节点的HCC值，形状为 (节点数,)
        
    Note:
        HCC(v_i) = C(v_i) + α·H(v_i)/log(|V|)
        其中：
        - C(v_i): 节点v_i的超边连接度（直接通过超边连接到的节点数量）
        - H(v_i): 节点v_i的超边覆盖度（通过超边间接覆盖到的节点集合大小）
        - α: 调节参数，平衡直接连接和间接覆盖的重要性
        - |V|: 超图中节点的总数
        
    Example:
        >>> # 假设有4个节点，2个超边
        >>> # 超边0连接节点{0,1,2}，超边1连接节点{1,2,3}
        >>> hypergraph = torch.sparse.FloatTensor(
        ...     indices=torch.LongTensor([[0,1,2,1,2,3], [0,0,0,1,1,1]]),
        ...     values=torch.FloatTensor([1,1,1,1,1,1]),
        ...     size=(4, 2)
        ... )
        >>> hcc_scores = calculate_hyperedge_coverage_centrality(hypergraph, alpha=0.5)
        >>> print(hcc_scores)
    """
    # 转换为numpy矩阵以便计算
    if torch.is_tensor(hypergraph):
        if hypergraph.is_sparse:
            # 转换sparse tensor到numpy
            hypergraph = hypergraph.coalesce()
            indices = hypergraph.indices().cpu().numpy()
            values = hypergraph.values().cpu().numpy()
            shape = hypergraph.shape
            from scipy.sparse import coo_matrix
            H = coo_matrix((values, (indices[0], indices[1])), shape=shape).tocsr()
        else:
            H = hypergraph.cpu().numpy()
    else:
        H = hypergraph.tocsr() if hasattr(hypergraph, 'tocsr') else hypergraph
    
    num_nodes, num_hyperedges = H.shape
    hcc_scores = np.zeros(num_nodes)
    
    # 为了避免log(0)的情况
    log_num_nodes = math.log(max(num_nodes, 2))
    
    for node_i in range(num_nodes):
        # 1. 计算直接连接度 C(v_i)
        # 找到节点i所属的所有超边
        hyperedges_of_node_i = H[node_i, :].nonzero()[1]  # 节点i连接的超边索引
        
        # 计算通过这些超边直接连接的所有其他节点
        directly_connected_nodes = set()
        for hyperedge_idx in hyperedges_of_node_i:
            # 找到这个超边中的所有节点
            nodes_in_hyperedge = H[:, hyperedge_idx].nonzero()[0]
            directly_connected_nodes.update(nodes_in_hyperedge)
        
        # 移除节点i自身，得到直接连接的节点数量
        directly_connected_nodes.discard(node_i)
        C_vi = len(directly_connected_nodes)
        
        # 2. 计算间接覆盖度 H(v_i)
        # 找到与节点i的直接邻居超边相邻的所有超边
        indirectly_covered_nodes = set(directly_connected_nodes)  # 初始化为直接连接的节点
        
        # 对于每个直接连接的节点，找到它们连接的超边
        for connected_node in directly_connected_nodes:
            # 找到connected_node所属的超边
            hyperedges_of_connected_node = H[connected_node, :].nonzero()[1]
            
            # 通过这些超边找到间接覆盖的节点
            for hyperedge_idx in hyperedges_of_connected_node:
                nodes_in_hyperedge = H[:, hyperedge_idx].nonzero()[0]
                indirectly_covered_nodes.update(nodes_in_hyperedge)
        
        # 移除节点i自身
        indirectly_covered_nodes.discard(node_i)
        H_vi = len(indirectly_covered_nodes)
        
        # 3. 计算最终的HCC值
        hcc_scores[node_i] = C_vi + alpha * H_vi / log_num_nodes
    
    return hcc_scores


def evaluate_hyperedge_coverage_centrality(hypergraphs, alpha_list=[0.1, 0.3, 0.5, 0.7, 0.9]):
    """
    评估多个超图结构的超边覆盖度中心性
    
    Args:
        hypergraphs (dict): 超图字典，键为图名称，值为超图矩阵
                           例如: {'user_hg': user_hypergraph, 'item_hg': item_hypergraph, 'full_hg': full_hypergraph}
        alpha_list (list): 不同的α参数值列表，用于对比分析
        
    Returns:
        dict: 嵌套字典，结构为 {graph_name: {alpha: hcc_scores}}
        
    Example:
        >>> hypergraphs = {
        ...     'user_hg': user_hypergraph,
        ...     'item_hg': item_hypergraph, 
        ...     'full_hg': full_hypergraph
        ... }
        >>> results = evaluate_hyperedge_coverage_centrality(hypergraphs)
        >>> print(f"User hypergraph HCC (α=0.5): {results['user_hg'][0.5]}")
    """
    results = {}
    
    for graph_name, hypergraph in hypergraphs.items():
        results[graph_name] = {}
        print(f"\n计算 {graph_name} 的超边覆盖度中心性...")
        
        for alpha in alpha_list:
            hcc_scores = calculate_hyperedge_coverage_centrality(hypergraph, alpha=alpha)
            results[graph_name][alpha] = hcc_scores
            
            # 输出统计信息
            print(f"  α={alpha:.1f}: 平均HCC={np.mean(hcc_scores):.4f}, "
                  f"标准差={np.std(hcc_scores):.4f}, "
                  f"最大值={np.max(hcc_scores):.4f}, "
                  f"最小值={np.min(hcc_scores):.4f}")
    
    return results
