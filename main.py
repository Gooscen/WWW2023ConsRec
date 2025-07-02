"""
ConsRec 群体推荐主程序
基于共识的群体推荐系统，融合三种视角：成员级聚合、物品级品味、群体级偏好
"""
import torch
import random
import torch.optim as optim
import numpy as np
from metrics import evaluate
from model import ConsRec
from datetime import datetime
import argparse
import time
from dataloader import GroupDataset
from tensorboardX import SummaryWriter
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def set_seed(seed):
    """
    设置随机种子以确保实验结果可重现
    
    Args:
        seed (int): 随机种子值
        
    Note:
        设置numpy、python random、torch(CPU和GPU)以及cudnn的随机种子
        确保在相同环境下运行结果一致
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True


def training(train_loader, epoch, type_m="group"):
    """
    执行一个epoch的训练过程
    
    Args:
        train_loader (DataLoader): 训练数据加载器
        epoch (int): 当前训练轮数
        type_m (str): 训练类型，"group"表示群组推荐，"user"表示用户推荐
        
    Returns:
        float: 当前epoch的平均损失值
        
    Note:
        使用BPR (Bayesian Personalized Ranking) 损失函数
        通过对比正样本和负样本的预测分数来优化排序
    """
    st_time = time.time()
    lr = args.learning_rate
    optimizer = optim.RMSprop(train_model.parameters(), lr=lr)
    losses = []

    for batch_id, (u, pi_ni) in enumerate(train_loader):
        user_input = torch.LongTensor(u).to(running_device)
        pos_items_input, neg_items_input = pi_ni[:, 0].to(running_device), pi_ni[:, 1].to(running_device)

        if type_m == 'user':
            pos_prediction = train_model(None, user_input, pos_items_input)
            neg_prediction = train_model(None, user_input, neg_items_input)
        else:
            pos_prediction = train_model(user_input, None, pos_items_input)
            neg_prediction = train_model(user_input, None, neg_items_input)

        optimizer.zero_grad()
        if args.loss_type == "BPR":
            loss = torch.mean(torch.nn.functional.softplus(neg_prediction - pos_prediction))
        else:
            loss = torch.mean((pos_prediction - neg_prediction - 1) ** 2)

        losses.append(loss)
        loss.backward()
        optimizer.step()

    print(
        f'Epoch {epoch}, {type_m} loss: {torch.mean(torch.stack(losses)):.5f}, Cost time: {time.time() - st_time:4.2f}s')
    return torch.mean(torch.stack(losses)).item()


if __name__ == "__main__":
    """
    主程序入口
    
    执行流程:
    1. 解析命令行参数
    2. 设置随机种子和计算设备
    3. 加载数据集并构建各种图结构
    4. 初始化ConsRec模型
    5. 执行训练循环：
       - 群组推荐训练
       - 用户推荐训练  
       - 群组推荐评估
       - 用户推荐评估
    6. 输出最终结果
    """
    # 命令行参数解析
    parser = argparse.ArgumentParser()
    
    # 基础参数
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--dataset", type=str, help="数据集名称 [Mafengwo, CAMRa2011]", default="Mafengwo")
    parser.add_argument("--device", type=str, help="计算设备 [cuda:0, ..., cpu]", default="cuda:0")

    # 模型架构参数
    parser.add_argument("--layers", type=int, help="超图卷积和重叠图卷积的层数", default=3)
    parser.add_argument("--emb_dim", type=int, help="用户/物品/群组嵌入向量维度", default=32)
    parser.add_argument("--num_negatives", type=int, help="每个正样本对应的负样本数量", default=8)
    parser.add_argument("--topK", type=list, help="Top-K评估指标的K值列表", default=[1, 5, 10])

    # 训练参数
    parser.add_argument("--epoch", type=int, default=100, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, help="学习率", default=0.001)
    parser.add_argument("--batch_size", type=float, help="批处理大小", default=512)
    parser.add_argument("--patience", type=int, help="早停耐心值", default=4)
    parser.add_argument("--predictor", type=str, help="预测器类型 [MLP, DOT]", default="MLP")
    parser.add_argument("--loss_type", type=str, help="损失函数类型 [BPR, MSE]", default="BPR")

    args = parser.parse_args()
    set_seed(args.seed)

    print('= ' * 20)
    print('## Starting Time:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print(args)

    # TensorBoard日志记录(已注释)
    writer_dir = f"ckpts/{args.dataset}/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(writer_dir)

    # 设置计算设备
    running_device = torch.device(args.device)

    # 数据集加载
    user_path, group_path = f"./data/{args.dataset}/userRating", f"./data/{args.dataset}/groupRating"
    dataset = GroupDataset(user_path, group_path, num_negatives=args.num_negatives, dataset=args.dataset)
    num_users, num_items, num_groups = dataset.num_users, dataset.num_items, dataset.num_groups
    print(f" #Users {num_users}, #Items {num_items}, #Groups {num_groups}\n")

    # 将图结构转移到指定设备
    user_hg, item_hg, full_hg = dataset.user_hyper_graph.to(running_device), dataset.item_hyper_graph.to(
        running_device), dataset.full_hg.to(running_device)
    overlap_graph = torch.Tensor(dataset.overlap_graph).to(running_device)
    light_gcn_graph = dataset.light_gcn_graph.to(running_device)

    # 模型初始化
    train_model = ConsRec(num_users, num_items, num_groups, args, user_hg, item_hg,
                          full_hg, overlap_graph, running_device, light_gcn_graph, dataset.num_group_net_items)
    train_model.to(running_device)

    # 训练循环
    for epoch_id in range(args.epoch):
        # 设置模型为训练模式
        train_model.train()
        
        # 群组推荐训练
        group_loss = training(dataset.get_group_dataloader(args.batch_size), epoch_id, "group")
        writer.add_scalar("Group Loss", group_loss, epoch_id)
        
        # 用户推荐训练
        user_loss = training(dataset.get_user_dataloader(args.batch_size), epoch_id, "user")
        writer.add_scalar("User Loss", user_loss, epoch_id)

        # 群组推荐评估
        hits, ndcgs = evaluate(train_model, dataset.group_test_ratings, dataset.group_test_negatives, running_device,
                               args.topK, 'group')

        print(f"[Epoch {epoch_id}] Group, Hit@{args.topK}: {hits}, NDCG@{args.topK}: {ndcgs}")
        writer.add_scalars(f'Group/Hit@{args.topK}', {str(args.topK[i]): hits[i] for i in range(len(args.topK))}, epoch_id)
        writer.add_scalars(f'Group/NDCG@{args.topK}', {str(args.topK[i]): ndcgs[i] for i in range(len(args.topK))}, epoch_id)

        # 用户推荐评估
        hrs, ngs = evaluate(train_model, dataset.user_test_ratings, dataset.user_test_negatives, running_device,
                            args.topK, 'user')

        print(f"[Epoch {epoch_id}] User, Hit@{args.topK}: {hrs}, NDCG@{args.topK}: {ngs}")
        print()
        writer.add_scalars(f'User/Hit@{args.topK}', {str(args.topK[i]): hrs[i] for i in range(len(args.topK))}, epoch_id)
        writer.add_scalars(f'User/NDCG@{args.topK}', {str(args.topK[i]): ndcgs[i] for i in range(len(args.topK))}, epoch_id)

    # 关闭TensorBoard writer
    writer.close()
    
    print()
    print('## Finishing Time:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print('= ' * 20)
    print("Done!")
