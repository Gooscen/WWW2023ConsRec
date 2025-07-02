"""
ConsRec 模型定义模块
包含基于共识的群体推荐系统的所有网络组件
"""
import torch.nn as nn
import torch


class PredictLayer(nn.Module):
    """
    预测层 - 将嵌入向量转换为预测分数
    
    Args:
        emb_dim (int): 输入嵌入向量的维度
        drop_ratio (float): Dropout比率，用于防止过拟合
        
    Architecture:
        输入 -> Linear(emb_dim, 8) -> ReLU -> Dropout -> Linear(8, 1) -> 输出
    """
    def __init__(self, emb_dim, drop_ratio=0.):
        super(PredictLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(emb_dim, 8),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入特征向量，形状为 (batch_size, emb_dim)
            
        Returns:
            torch.Tensor: 预测分数，形状为 (batch_size, 1)
        """
        return self.linear(x)


class OverlapGraphConvolution(nn.Module):
    """
    重叠图卷积 - 用于群组级别图的卷积操作
    
    功能:
        基于群组间的重叠关系进行信息传播
        使用多层卷积聚合邻居群组的信息
        
    Args:
        layers (int): 卷积层数
    """

    def __init__(self, layers):
        super(OverlapGraphConvolution, self).__init__()
        self.layers = layers

    def forward(self, embedding, adj):
        """
        前向传播
        
        Args:
            embedding (torch.Tensor): 群组嵌入向量，形状为 (num_groups, emb_dim)
            adj (torch.Tensor): 归一化的邻接矩阵，形状为 (num_groups, num_groups)
            
        Returns:
            torch.Tensor: 更新后的群组嵌入向量
            
        Note:
            使用残差连接的思想，将所有层的输出相加
            公式: H^(l+1) = A * H^(l)，最终输出为所有层输出的和
        """
        group_emb = embedding
        final = [group_emb]

        for _ in range(self.layers):
            group_emb = torch.mm(adj, group_emb)
            final.append(group_emb)

        final_emb = torch.sum(torch.stack(final), dim=0)
        return final_emb


class LightGCN(nn.Module):
    """
    LightGCN 图神经网络 - 用于物品级别图的卷积操作
    
    功能:
        在群组-物品二分图上进行信息传播
        简化版的图卷积网络，去除了非线性激活和特征变换
        
    Args:
        num_groups (int): 群组数量
        num_items (int): 物品数量
        layers (int): 卷积层数
        g (torch.sparse.FloatTensor): 归一化的二分图邻接矩阵
    """

    def __init__(self, num_groups, num_items, layers, g):
        super(LightGCN, self).__init__()

        self.num_groups, self.num_items = num_groups, num_items
        self.layers = layers
        self.graph = g

    def compute(self, groups_emb, items_emb):
        """
        LightGCN 前向传播计算
        
        Args:
            groups_emb (torch.Tensor): 群组嵌入向量
            items_emb (torch.Tensor): 物品嵌入向量
            
        Returns:
            torch.Tensor: 更新后的群组嵌入向量
            
        Note:
            将群组和物品嵌入拼接后进行多层图卷积
            最终取所有层输出的平均值作为最终表示
        """
        all_emb = torch.cat([groups_emb, items_emb])
        embeddings = [all_emb]

        for _ in range(self.layers):
            all_emb = torch.sparse.mm(self.graph, all_emb)
            embeddings.append(all_emb)
        embeddings = torch.mean(torch.stack(embeddings, dim=1), dim=1)
        groups, _ = torch.split(embeddings, [self.num_groups, self.num_items])
        return groups

    def forward(self, groups_emb, items_emb):
        """
        前向传播接口
        
        Args:
            groups_emb (torch.Tensor): 群组嵌入向量
            items_emb (torch.Tensor): 物品嵌入向量
            
        Returns:
            torch.Tensor: 更新后的群组嵌入向量
        """
        return self.compute(groups_emb, items_emb)


class HyperGraphBasicConvolution(nn.Module):
    """
    基础超图卷积层
    
    功能:
        实现单层超图卷积操作
        融合用户、物品和群组的信息进行超图消息传递
        
    Args:
        input_dim (int): 输入嵌入向量维度
    """
    def __init__(self, input_dim):
        super(HyperGraphBasicConvolution, self).__init__()
        self.aggregation = nn.Linear(3 * input_dim, input_dim)

    def forward(self, user_emb, item_emb, group_emb, user_hyper_graph, item_hyper_graph, full_hyper):
        """
        前向传播
        
        Args:
            user_emb (torch.Tensor): 用户嵌入向量
            item_emb (torch.Tensor): 物品嵌入向量  
            group_emb (torch.Tensor): 群组嵌入向量
            user_hyper_graph (torch.sparse.FloatTensor): 用户超图
            item_hyper_graph (torch.sparse.FloatTensor): 物品超图
            full_hyper (torch.sparse.FloatTensor): 完整超图
            
        Returns:
            tuple: (更新后的节点表示, 超边表示)
                - norm_emb: 经过超图卷积更新的节点表示
                - msg: 超边(群组)的表示
                
        Note:
            超图卷积分为两个阶段:
            1. 节点到超边: 聚合超边内的节点信息
            2. 超边到节点: 将超边信息传播回节点
        """
        # 第一阶段: 节点到超边的信息聚合
        user_msg = torch.sparse.mm(user_hyper_graph, user_emb)
        item_msg = torch.sparse.mm(item_hyper_graph, item_emb)

        # 融合用户、物品和群组信息
        item_group_element = item_msg * group_emb
        msg = self.aggregation(torch.cat([user_msg, item_msg, item_group_element], dim=1))
        
        # 第二阶段: 超边到节点的信息传播
        norm_emb = torch.mm(full_hyper, msg)
        # norm_emb (refined node representations)，msg (hyperedges' representations)
        return norm_emb, msg


class HyperGraphConvolution(nn.Module):
    """
    多层超图卷积网络 - 用于成员级别超图的卷积操作
    
    功能:
        堆叠多个基础超图卷积层
        实现深层次的超图信息传播
        
    Args:
        user_hyper_graph (torch.sparse.FloatTensor): 用户超图
        item_hyper_graph (torch.sparse.FloatTensor): 物品超图
        full_hyper (torch.sparse.FloatTensor): 完整超图
        layers (int): 卷积层数
        input_dim (int): 输入嵌入维度
        device (torch.device): 计算设备
    """

    def __init__(self, user_hyper_graph, item_hyper_graph, full_hyper, layers,
                 input_dim, device):
        super(HyperGraphConvolution, self).__init__()
        self.layers = layers
        self.user_hyper, self.item_hyper, self.full_hyper_graph = user_hyper_graph, item_hyper_graph, full_hyper
        self.hgnns = [HyperGraphBasicConvolution(input_dim).to(device) for _ in range(layers)]

    def forward(self, user_emb, item_emb, group_emb, num_users, num_items):
        """
        多层超图卷积前向传播
        
        Args:
            user_emb (torch.Tensor): 初始用户嵌入
            item_emb (torch.Tensor): 初始物品嵌入
            group_emb (torch.Tensor): 初始群组嵌入
            num_users (int): 用户数量
            num_items (int): 物品数量
            
        Returns:
            tuple: (最终节点表示, 最终超边表示)
                - final_emb: 所有层输出的节点表示之和
                - final_he: 所有层输出的超边表示之和
                
        Note:
            使用残差连接思想，将所有层的输出累加
            这有助于缓解深层网络的梯度消失问题
        """
        final = [torch.cat([user_emb, item_emb], dim=0)]
        final_he = [group_emb]
        
        for i in range(len(self.hgnns)):
            hgnn = self.hgnns[i]
            emb, he_msg = hgnn(user_emb, item_emb, group_emb, self.user_hyper, self.item_hyper, self.full_hyper_graph)
            user_emb, item_emb = torch.split(emb, [num_users, num_items])
            final.append(emb)
            final_he.append(he_msg)

        final_emb = torch.sum(torch.stack(final), dim=0)
        final_he = torch.sum(torch.stack(final_he), dim=0)
        # Final nodes' (users and items) representations and final hyper-edges' (groups) representations
        return final_emb, final_he


class ConsRec(nn.Module):
    """
    ConsRec: 基于共识的群体推荐模型
    
    核心思想:
        融合三种不同视角的信息来捕捉群体共识:
        1. 成员级聚合 (超图神经网络)
        2. 物品级品味 (LightGCN)  
        3. 群体级偏好 (重叠图卷积)
        
    Args:
        num_users (int): 用户总数
        num_items (int): 物品总数
        num_groups (int): 群组总数
        args: 命令行参数对象
        user_hyper_graph: 用户超图
        item_hyper_graph: 物品超图
        full_hyper: 完整超图
        overlap_graph: 群组重叠图
        device: 计算设备
        light_gcn_graph: LightGCN图
        num_lgcn_item: LightGCN中的物品数量
    """

    def __init__(self, num_users, num_items, num_groups, args, user_hyper_graph, item_hyper_graph,
                 full_hyper, overlap_graph, device, light_gcn_graph, num_lgcn_item):
        super(ConsRec, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.num_groups = num_groups

        # 超参数
        self.emb_dim = args.emb_dim
        self.layers = args.layers
        self.device = args.device
        self.predictor_type = args.predictor

        # 图结构
        self.overlap_graph = overlap_graph
        
        # 嵌入层初始化
        self.user_embedding = nn.Embedding(num_users, self.emb_dim)
        self.item_embedding = nn.Embedding(num_items, self.emb_dim)
        self.group_embedding = nn.Embedding(num_groups, self.emb_dim)

        # Xavier均匀初始化
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.group_embedding.weight)

        # 自适应融合门控机制
        self.overlap_gate = nn.Sequential(nn.Linear(self.emb_dim, 1), nn.Sigmoid())
        self.hyper_gate = nn.Sequential(nn.Linear(self.emb_dim, 1), nn.Sigmoid())
        self.lightgcn_gate = nn.Sequential(nn.Linear(self.emb_dim, 1), nn.Sigmoid())

        # 三种图卷积组件
        # 1. 超图卷积 (成员级聚合)
        self.hyper_graph_conv = HyperGraphConvolution(user_hyper_graph, item_hyper_graph, full_hyper, self.layers,
                                                      self.emb_dim, device)

        # 2. 重叠图卷积 (群体级偏好)
        self.overlap_graph_conv = OverlapGraphConvolution(self.layers)

        # 3. LightGCN (物品级品味)
        self.light_gcn = LightGCN(num_groups, num_lgcn_item, self.layers, light_gcn_graph)
        self.num_lgcn_item = num_lgcn_item

        # 预测层
        self.predict = PredictLayer(self.emb_dim)

    def forward(self, group_inputs, user_inputs, item_inputs):
        """
        模型前向传播路由
        
        Args:
            group_inputs: 群组输入ID (群组推荐时使用)
            user_inputs: 用户输入ID (用户推荐时使用)
            item_inputs: 物品输入ID
            
        Returns:
            torch.Tensor: 预测分数
            
        Note:
            根据输入参数自动选择群组推荐或用户推荐分支
        """
        if (group_inputs is not None) and (user_inputs is None):
            return self.group_forward(group_inputs, item_inputs)
        else:
            return self.user_forward(user_inputs, item_inputs)

    def group_forward(self, group_inputs, item_inputs):
        """
        群组推荐前向传播
        
        Args:
            group_inputs (torch.Tensor): 群组ID
            item_inputs (torch.Tensor): 物品ID
            
        Returns:
            torch.Tensor: 群组对物品的预测分数
            
        核心流程:
        1. 群组级别图计算 (重叠图卷积)
        2. 成员级别图计算 (超图卷积)  
        3. 物品级别图计算 (LightGCN)
        4. 自适应融合三种视角
        5. 预测分数计算
        """
        # 1. 群组级别图计算 (群体级偏好)
        group_emb = self.overlap_graph_conv(self.group_embedding.weight, self.overlap_graph)

        # 2. 成员级别图计算 (成员级聚合)
        ui_emb, he_emb = self.hyper_graph_conv(self.user_embedding.weight, self.item_embedding.weight,
                                               group_emb,
                                               self.num_users, self.num_items)
        _, i_emb = torch.split(ui_emb, [self.num_users, self.num_items])

        # 3. 物品级别图计算 (物品级品味)
        light_gcn_group_emb = self.light_gcn(self.group_embedding.weight,
                                             self.item_embedding.weight[:self.num_lgcn_item, :])

        # 4. 自适应融合三种视角
        overlap_coef, hyper_coef, lightgcn_coef = self.overlap_gate(group_emb), self.hyper_gate(
            he_emb), self.lightgcn_gate(light_gcn_group_emb)

        group_ui_emb = overlap_coef * group_emb + hyper_coef * he_emb + lightgcn_coef * light_gcn_group_emb
        
        # 5. 获取对应的嵌入向量
        i_emb = i_emb[item_inputs]
        g_emb = group_ui_emb[group_inputs]

        # 6. 预测分数计算
        # For CAMRa2011, we use DOT mode to avoid the dead ReLU
        if self.predictor_type == "MLP":
            return torch.sigmoid(self.predict(g_emb * i_emb))
        else:
            return torch.sum(g_emb * i_emb, dim=-1)

    def user_forward(self, user_inputs, item_inputs):
        """
        用户推荐前向传播
        
        Args:
            user_inputs (torch.Tensor): 用户ID
            item_inputs (torch.Tensor): 物品ID
            
        Returns:
            torch.Tensor: 用户对物品的预测分数
            
        Note:
            用户推荐使用简单的嵌入向量点积或MLP预测
            不涉及复杂的图卷积操作
        """
        u_emb = self.user_embedding(user_inputs)
        i_emb = self.item_embedding(item_inputs)
        if self.predictor_type == "MLP":
            return torch.sigmoid(self.predict(u_emb * i_emb))
        else:
            return torch.sum(u_emb * i_emb, dim=-1)
