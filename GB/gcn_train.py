import argparse
import math
import os
import cv2
import numpy as np
from operator import add
from functools import reduce
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
import time
from collections import Counter
import networkx as nx
import scipy.sparse as sp
import itertools
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from torch_geometric.utils import degree

from img2graph import granular_balls_generate, cal_bound
from model import GCN_8_plus, GCN_8_plus2, GCN_Layer_4, GCN, PNA_GNN

random.seed(0)


def read_train_npz(img, mode="train"):
    base_name = os.path.splitext(img)[0]
    if mode == "train":
        npz_path = os.path.join(r"E:\dataset\SPINE\train\granular_features25_3\images_256_npz", base_name + ".npz")
    else:
        npz_path = os.path.join(r"E:\dataset\SPINE\val\granular_features25_3\images_256_npz", base_name + ".npz")

    if not os.path.exists(npz_path):
        print(f"❌ 文件不存在: {npz_path}")
        return None
    data = np.load(npz_path)
    center_array = data['center_array']
    adj = data['adj']
    edge_attr = data['edge_attr']
    center_ = data['center_']

    return center_array, adj, edge_attr, center_


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss



# 为粒球分配多类标签
def assign_labels(center_, mask):
    labels = []
    for center in center_:
        x, y, Rx, Ry = center
        left, right, up, down = cal_bound(mask, [int(x), int(y)], Rx, Ry)
        region = mask[up:down + 1, left:right + 1]
        # 统计区域内各类别像素的分布，选择占比最高的类别
        counts = Counter(region.flatten())
        # 移除无效值（如果有），选择最多的类别
        max_count = -1
        label = -1
        for k, v in counts.items():
            if v > max_count and k != -1:  # 忽略无效值（如果有）
                max_count = v
                label = k
        labels.append(label)
    # 将标签映射到连续的整数（从0开始）
    unique_labels = sorted(np.unique(labels))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels = [label_map[l] for l in labels]
    return np.array(labels), len(unique_labels)


def filter_background_nodes(data):
    """
    删除与前景节点完全无连接的背景节点
    """
    edge_index = data.edge_index.cpu().numpy()
    labels = data.y.cpu().numpy()

    num_nodes = data.num_nodes
    is_useful = np.zeros(num_nodes, dtype=bool)

    # 标记前景点为保留
    is_useful[labels != 0] = True

    # 构建邻接图
    adj_list = [[] for _ in range(num_nodes)]
    for src, dst in zip(*edge_index):
        adj_list[src].append(dst)
        adj_list[dst].append(src)

    # 背景节点若与前景连接也保留
    for i in range(num_nodes):
        if labels[i] == 0:
            neighbors = adj_list[i]
            if any(labels[n] != 0 for n in neighbors):
                is_useful[i] = True

    # 创建新节点索引映射
    old2new = {old: new for new, old in enumerate(np.where(is_useful)[0])}

    # 更新图结构
    new_edge_index = []
    new_edge_attr = []
    for i, (src, dst) in enumerate(zip(*edge_index)):
        if is_useful[src] and is_useful[dst]:
            new_edge_index.append([old2new[src], old2new[dst]])
            new_edge_attr.append(data.edge_attr[i])

    # 生成新 Data 对象
    data.x = data.x[is_useful]
    data.y = data.y[is_useful]
    data.edge_index = torch.tensor(new_edge_index, dtype=torch.long).t().contiguous()
    data.edge_attr = torch.stack(new_edge_attr)

    return data


# 创建图数据对象
def create_graph_data(img, mask, mode="train"):
    # center_array, adj, edge_attr, center_ = granular_balls_generate(img)

    center_array, adj, edge_attr, center_ = read_train_npz(img, mode)

    labels, num_classes = assign_labels(center_, mask)
    x = torch.tensor(center_array, dtype=torch.float)
    edge_index = torch.tensor(adj, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_classes=num_classes)

    # if mode == "train":
    #     data = filter_background_nodes(data)  # ✅ 只在训练时使用

    return data


# 数据加载和预处理
def load_data(train_image_dir, train_mask_dir, val_image_dir, val_mask_dir):
    train_data_list = []
    train_image_files = sorted([f for f in os.listdir(train_image_dir) if f.endswith('.png') or f.endswith('.jpg')])
    for img_file in train_image_files:
        img_path = os.path.join(train_image_dir, img_file)
        mask_path = os.path.join(train_mask_dir, img_file)
        if mask_path.endswith('.jpg'):
            mask_path = mask_path.replace('.jpg', '.png')
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            print(f"无法加载图像或掩码: {img_file}")
            continue
        # 归一化图像到[0, 255]
        # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        # data = create_graph_data(img, mask, purity, threshold, var_threshold)
        data = create_graph_data(img_file, mask, mode = "train") # 使用读取的npz数据
        train_data_list.append(data)

    val_data_list = []
    val_image_files = sorted([f for f in os.listdir(val_image_dir) if f.endswith('.png') or f.endswith('.jpg')])
    for img_file in val_image_files:
        img_path = os.path.join(val_image_dir, img_file)
        mask_path = os.path.join(val_mask_dir, img_file)
        if mask_path.endswith('.jpg'):
            mask_path = mask_path.replace('.jpg', '.png')
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            print(f"无法加载图像或掩码: {img_file}")
            continue
        # 归一化图像到[0, 255]
        # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        # data = create_graph_data(img, mask, purity, threshold, var_threshold)
        data = create_graph_data(img_file, mask, mode = "val") # 使用读取的npz数据
        val_data_list.append(data)

    return train_data_list, val_data_list


# 训练和评估
def train_and_evaluate(train_data_list, val_data_list, hidden_dim=64, epochs=100, lr=0.01, deg_hist=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 划分训练集和测试集
    train_loader = DataLoader(train_data_list, batch_size=4, shuffle=True)
    test_loader = DataLoader(val_data_list, batch_size=4, shuffle=False)

    # 获取类别数
    num_classes = max([data.num_classes for data in train_data_list])



    # # 计算类别权重
    # label_counts = Counter()
    # for data in train_data:
    #     label_counts.update(data.y.cpu().numpy())
    # total_nodes = sum(label_counts.values())
    # class_weights = torch.zeros(num_classes, dtype=torch.float)
    # for cls, count in label_counts.items():
    #     class_weights[cls] = total_nodes / (num_classes * count)  # 逆频率权重
    # class_weights = class_weights.to(device)

    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    # criterion = FocalLoss(gamma=2, alpha=class_weights)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 初始化模型
    model = GCN_8_plus(num_features=25, num_classes=num_classes, initdim=16, inithead=16, edge_dim=3).to(device)
    # model = PNA_GNN(num_features=31, num_classes=num_classes, edge_dim=13, hidden_dim=64, deg=deg_hist).to(device)
    # model = GCN_8_plus2(num_features=25, num_classes=num_classes, initdim=32, inithead=8, edge_dim=12).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_f1 = 0
    patience = 50
    counter = 0

    # 训练
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_y_true, train_y_pred = [], []
        for data in tqdm(train_loader, desc=f"训练 Epoch {epoch+1}", total=len(train_loader)):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            pred = out.argmax(dim=1)
            train_y_true.extend(data.y.cpu().numpy())
            train_y_pred.extend(pred.cpu().numpy())

        # 计算训练集评估指标
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = accuracy_score(train_y_true, train_y_pred)
        train_precision = precision_score(train_y_true, train_y_pred, average='macro', zero_division=0)
        train_recall = recall_score(train_y_true, train_y_pred, average='macro', zero_division=0)
        train_f1 = f1_score(train_y_true, train_y_pred, average='macro', zero_division=0)

        # 验证
        model.eval()
        val_loss = 0
        val_y_true, val_y_pred = [], []
        with torch.no_grad():
            for data in tqdm(test_loader, desc=f"验证 Epoch {epoch + 1}", total=len(test_loader)):
                data = data.to(device)
                out = model(data)

                loss = criterion(out, data.y)
                val_loss += loss.item()

                pred = out.argmax(dim=1)
                val_y_true.extend(data.y.cpu().numpy())
                val_y_pred.extend(pred.cpu().numpy())

        # 计算验证集评估指标
        avg_val_loss = val_loss / len(test_loader)
        val_accuracy = accuracy_score(val_y_true, val_y_pred)
        val_precision = precision_score(val_y_true, val_y_pred, average='macro', zero_division=0)
        val_recall = recall_score(val_y_true, val_y_pred, average='macro', zero_division=0)
        val_f1 = f1_score(val_y_true, val_y_pred, average='macro', zero_division=0)

        # 打印训练和验证结果
        print(f'轮次 {epoch+1}/{epochs}, '
              f'训练损失: {avg_train_loss:.4f}, '
              f'训练准确率: {train_accuracy:.4f}, 训练精确率: {train_precision:.4f}, '
              f'训练召回率: {train_recall:.4f}, 训练F1分数: {train_f1:.4f}, '
              f'验证损失: {avg_val_loss:.4f}, '
              f'验证准确率: {val_accuracy:.4f}, 验证精确率: {val_precision:.4f}, '
              f'验证召回率: {val_recall:.4f}, 验证F1分数: {val_f1:.4f}')


        # 保存最佳模型（基于验证的F1分数）
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_path = f'checkpoint/best_model_975_11.pth'
            torch.save(model.state_dict(), best_model_path)
            print(f'轮次 {epoch+1}: 保存最佳模型至 {best_model_path}, 验证的F1分数: {best_val_f1:.4f}')
            counter = 0
        else:
            counter += 1
        if counter >= patience:
            print(f'轮次 {epoch+1}: 验证的F1分数没有提升，提前停止训练。')
            break


# 主函数
def main():
    parser = argparse.ArgumentParser(description='GCN for Spine CT Multiclass Classification')
    parser.add_argument('--train_image_dir', default=r"E:\dataset\SPINE\train\granular_features25_3\images_256", type=str, help='CT图像目录')
    parser.add_argument('--train_mask_dir', default=r"E:\dataset\SPINE\train\granular_features25_3\masks_11_256", type=str, help='掩码标签目录')
    parser.add_argument('--val_image_dir', default=r"E:\dataset\SPINE\val\granular_features25_3\images_256", type=str, help='CT图像目录')
    parser.add_argument('--val_mask_dir', default=r"E:\dataset\SPINE\val\granular_features25_3\masks_11_256", type=str, help='掩码标签目录')
    parser.add_argument('--purity', type=float, default=0.9, help='粒球纯度')
    parser.add_argument('--threshold', type=float, default=10, help='异类点阈值')
    parser.add_argument('--var_threshold', type=float, default=20, help='方差阈值')
    parser.add_argument('--epochs', type=int, default=500, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--hidden_dim', type=int, default=64, help='隐藏层维度')
    args = parser.parse_args()

    # 加载数据
    train_data_list, val_data_list = load_data(args.train_image_dir, args.train_mask_dir, args.val_image_dir, args.val_mask_dir)
    if not train_data_list or not val_data_list:
        print("没有加载到有效数据！")
        return

    # deg_hist = compute_degree_histogram(train_data_list)


    # 训练和评估
    train_and_evaluate(train_data_list, val_data_list, hidden_dim=args.hidden_dim, epochs=args.epochs, lr=args.lr, deg_hist=None)


def compute_degree_histogram(data_list, max_deg=100):
    deg_hist = torch.zeros(max_deg, dtype=torch.long)
    for data in data_list:
        deg = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long)
        deg = deg.clamp(max=max_deg - 1)
        deg_hist += torch.bincount(deg, minlength=max_deg)
    return deg_hist

if __name__ == '__main__':
    main()