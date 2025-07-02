"""
超边覆盖度中心性（HCC）评估示例脚本
演示如何使用HCC指标分析推荐系统中的超图结构
"""

import torch
import numpy as np
from dataloader import GroupDataset
from metrics import calculate_hyperedge_coverage_centrality, evaluate_hyperedge_coverage_centrality
import argparse
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_hcc_for_dataset(dataset_name="yelp", alpha_values=[0.1, 0.3, 0.5, 0.7, 0.9]):
    """
    分析特定数据集的超边覆盖度中心性
    
    Args:
        dataset_name (str): 数据集名称
        alpha_values (list): 要测试的α参数值列表
    """
    print(f"正在加载 {dataset_name} 数据集...")
    
    # 加载数据集
    user_path = f"./data/{dataset_name}/userRating"
    group_path = f"./data/{dataset_name}/groupRating"
    dataset = GroupDataset(user_path, group_path, num_negatives=8, dataset=dataset_name)
    
    print(f"数据集统计信息:")
    print(f"  用户数: {dataset.num_users}")
    print(f"  物品数: {dataset.num_items}")
    print(f"  群组数: {dataset.num_groups}")
    
    # 准备超图数据
    hypergraphs = {
        'user_hypergraph': dataset.user_hyper_graph,
        'item_hypergraph': dataset.item_hyper_graph,
        'full_hypergraph': dataset.full_hg
    }
    
    print("\n开始计算超边覆盖度中心性指标...")
    
    # 计算HCC指标
    hcc_results = evaluate_hyperedge_coverage_centrality(hypergraphs, alpha_values)
    
    # 分析结果
    print("\n=== HCC分析结果 ===")
    
    for graph_name, alpha_results in hcc_results.items():
        print(f"\n{graph_name} 超图分析:")
        print("-" * 50)
        
        for alpha, hcc_scores in alpha_results.items():
            # 基本统计
            mean_hcc = np.mean(hcc_scores)
            std_hcc = np.std(hcc_scores)
            max_hcc = np.max(hcc_scores)
            min_hcc = np.min(hcc_scores)
            
            # 找到最重要的节点
            top_k = min(10, len(hcc_scores))
            top_nodes = np.argsort(hcc_scores)[-top_k:][::-1]
            
            print(f"  α={alpha:.1f}:")
            print(f"    平均HCC: {mean_hcc:.4f}")
            print(f"    标准差: {std_hcc:.4f}")
            print(f"    最大值: {max_hcc:.4f} (节点 {np.argmax(hcc_scores)})")
            print(f"    最小值: {min_hcc:.4f} (节点 {np.argmin(hcc_scores)})")
            print(f"    Top-{top_k}重要节点: {top_nodes.tolist()}")
    
    return hcc_results


def visualize_hcc_results(hcc_results, save_path="hcc_analysis.png"):
    """
    可视化HCC分析结果
    
    Args:
        hcc_results (dict): HCC计算结果
        save_path (str): 图片保存路径
    """
    plt.figure(figsize=(15, 10))
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    graph_names = list(hcc_results.keys())
    alpha_values = list(hcc_results[graph_names[0]].keys())
    
    # 子图1: 不同α值下的平均HCC
    plt.subplot(2, 3, 1)
    for graph_name in graph_names:
        mean_hcc_values = [np.mean(hcc_results[graph_name][alpha]) for alpha in alpha_values]
        plt.plot(alpha_values, mean_hcc_values, marker='o', label=graph_name)
    plt.xlabel('α 参数')
    plt.ylabel('平均HCC值')
    plt.title('不同α参数下的平均HCC')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2: HCC值分布（α=0.5）
    plt.subplot(2, 3, 2)
    alpha_mid = 0.5
    hcc_distributions = []
    labels = []
    for graph_name in graph_names:
        if alpha_mid in hcc_results[graph_name]:
            hcc_distributions.append(hcc_results[graph_name][alpha_mid])
            labels.append(graph_name)
    
    plt.boxplot(hcc_distributions, labels=labels)
    plt.ylabel('HCC值')
    plt.title(f'HCC分布 (α={alpha_mid})')
    plt.xticks(rotation=45)
    
    # 子图3: HCC标准差比较
    plt.subplot(2, 3, 3)
    for graph_name in graph_names:
        std_hcc_values = [np.std(hcc_results[graph_name][alpha]) for alpha in alpha_values]
        plt.plot(alpha_values, std_hcc_values, marker='s', label=graph_name)
    plt.xlabel('α 参数')
    plt.ylabel('HCC标准差')
    plt.title('不同α参数下的HCC标准差')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图4-6: 每个超图的HCC分布热图
    for i, graph_name in enumerate(graph_names):
        plt.subplot(2, 3, 4+i)
        hcc_matrix = []
        for alpha in alpha_values:
            hcc_scores = hcc_results[graph_name][alpha]
            # 取前50个节点的HCC值进行可视化
            sample_size = min(50, len(hcc_scores))
            sampled_hcc = hcc_scores[:sample_size]
            hcc_matrix.append(sampled_hcc)
        
        hcc_matrix = np.array(hcc_matrix)
        sns.heatmap(hcc_matrix, xticklabels=False, yticklabels=[f'α={a}' for a in alpha_values], 
                   cmap='viridis', cbar=True)
        plt.title(f'{graph_name} HCC热图')
        plt.xlabel('节点索引')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"可视化结果已保存到: {save_path}")
    plt.show()


def compare_node_importance(hcc_results, top_k=20):
    """
    比较不同超图中节点的重要性排序
    
    Args:
        hcc_results (dict): HCC计算结果
        top_k (int): 显示前K个重要节点
    """
    print(f"\n=== 节点重要性分析 (Top-{top_k}) ===")
    
    alpha = 0.5  # 使用中等α值进行比较
    
    for graph_name, alpha_results in hcc_results.items():
        if alpha in alpha_results:
            hcc_scores = alpha_results[alpha]
            
            # 获取最重要的节点
            top_indices = np.argsort(hcc_scores)[-top_k:][::-1]
            top_scores = hcc_scores[top_indices]
            
            print(f"\n{graph_name} (α={alpha}):")
            print("排名\t节点ID\tHCC值")
            print("-" * 30)
            for rank, (node_id, score) in enumerate(zip(top_indices, top_scores), 1):
                print(f"{rank:2d}\t{node_id:6d}\t{score:.4f}")


if __name__ == "__main__":
    """
    主程序：演示HCC指标的使用
    """
    parser = argparse.ArgumentParser(description="超边覆盖度中心性（HCC）评估")
    parser.add_argument("--dataset", type=str, default="yelp", help="数据集名称")
    parser.add_argument("--alpha_min", type=float, default=0.1, help="最小α值")
    parser.add_argument("--alpha_max", type=float, default=0.9, help="最大α值")
    parser.add_argument("--alpha_step", type=float, default=0.2, help="α值步长")
    parser.add_argument("--visualize", action="store_true", help="是否生成可视化图表")
    parser.add_argument("--top_k", type=int, default=20, help="显示前K个重要节点")
    
    args = parser.parse_args()
    
    # 生成α值列表
    alpha_values = np.arange(args.alpha_min, args.alpha_max + args.alpha_step, args.alpha_step)
    alpha_values = np.round(alpha_values, 1).tolist()
    
    print("超边覆盖度中心性（HCC）评估工具")
    print("=" * 50)
    print(f"数据集: {args.dataset}")
    print(f"α参数范围: {alpha_values}")
    
    try:
        # 执行HCC分析
        hcc_results = analyze_hcc_for_dataset(args.dataset, alpha_values)
        
        # 比较节点重要性
        compare_node_importance(hcc_results, args.top_k)
        
        # 可视化结果（如果请求）
        if args.visualize:
            try:
                visualize_hcc_results(hcc_results, f"hcc_analysis_{args.dataset}.png")
            except ImportError:
                print("警告: 无法导入matplotlib/seaborn，跳过可视化")
            except Exception as e:
                print(f"可视化过程中出现错误: {e}")
        
        print("\nHCC分析完成！")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        print("请检查数据集路径和参数设置")