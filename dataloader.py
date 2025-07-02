"""群体推荐数据加载器模块 - 提供数据集类和数据加载功能"""
from datautil import load_rating_file_to_matrix, load_rating_file_to_list, load_negative_file, \
    load_group_member_to_dict, build_hyper_graph, build_group_graph, build_light_gcn_graph
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


class GroupDataset(object):
    """
    群体推荐数据集类
    
    功能:
        - 加载用户和群组的训练/测试数据
        - 构建超图、重叠图和二分图结构
        - 提供训练数据的DataLoader
        
    Attributes:
        num_users (int): 用户总数
        num_items (int): 物品总数  
        num_groups (int): 群组总数
        num_group_net_items (int): 群组网络中的物品总数
        user_hyper_graph (torch.sparse.FloatTensor): 用户超图
        item_hyper_graph (torch.sparse.FloatTensor): 物品超图
        full_hg (torch.sparse.FloatTensor): 完整超图
        overlap_graph (numpy.ndarray): 群组重叠图
        light_gcn_graph (torch.sparse.FloatTensor): LightGCN二分图
    """
    
    def __init__(self, user_path, group_path, num_negatives, dataset="Mafengwo"):
        """
        初始化群体推荐数据集
        
        Args:
            user_path (str): 用户数据文件路径前缀 (如 "./data/Mafengwo/userRating")
            group_path (str): 群组数据文件路径前缀 (如 "./data/Mafengwo/groupRating") 
            num_negatives (int): 每个正样本对应的负样本数量
            dataset (str): 数据集名称，支持 "Mafengwo", "CAMRa2011", "MafengwoS"
            
        Note:
            会自动加载以下文件:
            - {user_path}Train.txt, {user_path}Test.txt, {user_path}Negative.txt
            - {group_path}Train.txt, {group_path}Test.txt, {group_path}Negative.txt
            - ./data/{dataset}/groupMember.txt
        """
        print(f"[{dataset.upper()}] loading...")
        self.num_negatives = num_negatives

        # User data
        if dataset == "MafengwoS":
            self.user_train_matrix = load_rating_file_to_matrix(user_path + "Train.txt", num_users=11026, num_items=1235)
        else:
            self.user_train_matrix = load_rating_file_to_matrix(user_path + "Train.txt")
        self.user_test_ratings = load_rating_file_to_list(user_path + "Test.txt")
        self.user_test_negatives = load_negative_file(user_path + "Negative.txt")
        self.num_users, self.num_items = self.user_train_matrix.shape

        print(f"UserItem: {self.user_train_matrix.shape} with {len(self.user_train_matrix.keys())} "
              f"interactions, sparsity: {(1-len(self.user_train_matrix.keys()) / self.num_users / self.num_items):.5f}")

        # Group data
        self.group_train_matrix = load_rating_file_to_matrix(group_path + "Train.txt")
        self.group_test_ratings = load_rating_file_to_list(group_path + "Test.txt")
        self.group_test_negatives = load_negative_file(group_path + "Negative.txt")
        self.num_groups, self.num_group_net_items = self.group_train_matrix.shape
        self.group_member_dict = load_group_member_to_dict(f"./data/{dataset}/groupMember.txt")

        print(f"GroupItem: {self.group_train_matrix.shape} with {len(self.group_train_matrix.keys())} interactions, spa"
              f"rsity: {(1-len(self.group_train_matrix.keys()) / self.num_groups / self.group_train_matrix.shape[1]):.5f}")

        # Member-level Hyper-graph
        self.user_hyper_graph, self.item_hyper_graph, self.full_hg, group_data = build_hyper_graph(
            self.group_member_dict, group_path + "Train.txt", self.num_users, self.num_items, self.num_groups)
        # Group-level graph
        self.overlap_graph = build_group_graph(group_data, self.num_groups)
        # Item-level graph
        self.light_gcn_graph = build_light_gcn_graph(self.group_train_matrix, self.num_groups, self.num_group_net_items)
        print(f"\033[0;30;43m{dataset.upper()} finish loading!\033[0m", end='')

    def get_train_instances(self, train):
        """
        生成训练样本 (实体ID, 正样本物品, 负样本物品)
        
        Args:
            train (scipy.sparse matrix): 训练交互矩阵，可以是用户-物品或群组-物品矩阵
            
        Returns:
            tuple: (实体ID列表, 正负样本对列表)
                - 实体ID列表: 用户ID或群组ID的列表
                - 正负样本对列表: [[正样本ID, 负样本ID], ...] 的列表
                
        Note:
            对每个正样本，会随机采样 self.num_negatives 个负样本
            负样本通过随机采样确保不在训练集中出现
        """
        users, pos_items, neg_items = [], [], []

        num_users, num_items = train.shape[0], train.shape[1]

        for (u, i) in train.keys():
            for _ in range(self.num_negatives):
                users.append(u)
                pos_items.append(i)

                j = np.random.randint(num_items)
                while (u, j) in train:
                    j = np.random.randint(num_items)
                neg_items.append(j)
        pos_neg_items = [[pos_item, neg_item] for pos_item, neg_item in zip(pos_items, neg_items)]
        return users, pos_neg_items

    def get_user_dataloader(self, batch_size):
        """
        获取用户训练数据的DataLoader
        
        Args:
            batch_size (int): 批处理大小
            
        Returns:
            torch.utils.data.DataLoader: 用户训练数据加载器
                每个batch包含 (用户ID, [正样本ID, 负样本ID])
                
        Note:
            数据会被自动打乱(shuffle=True)以提高训练效果
        """
        users, pos_neg_items = self.get_train_instances(self.user_train_matrix)
        train_data = TensorDataset(torch.LongTensor(users), torch.LongTensor(pos_neg_items))
        return DataLoader(train_data, batch_size=batch_size, shuffle=True)

    def get_group_dataloader(self, batch_size):
        """
        获取群组训练数据的DataLoader
        
        Args:
            batch_size (int): 批处理大小
            
        Returns:
            torch.utils.data.DataLoader: 群组训练数据加载器
                每个batch包含 (群组ID, [正样本ID, 负样本ID])
                
        Note:
            数据会被自动打乱(shuffle=True)以提高训练效果
        """
        groups, pos_neg_items = self.get_train_instances(self.group_train_matrix)
        train_data = TensorDataset(torch.LongTensor(groups), torch.LongTensor(pos_neg_items))
        return DataLoader(train_data, batch_size=batch_size, shuffle=True)
