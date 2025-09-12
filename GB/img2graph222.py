import cv2
import networkx as nx
import scipy.sparse as sp
import random
from sklearn.neighbors import NearestNeighbors
import numpy as np
from typing import Tuple
import math
from skimage.feature import local_binary_pattern
from scipy.stats import entropy as scipy_entropy

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)

def get_grad(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobely = cv2.convertScaleAbs(sobely)
    sobelxy2 = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    return sobelxy2

def cal_bound(img, center, Rx, Ry):
    h, w = img.shape
    left = max(0, center[1] - Rx)
    right = min(w - 1, center[1] + Rx)
    up = max(0, center[0] - Ry)
    down = min(h - 1, center[0] + Ry)
    return int(left), int(right), int(up), int(down)

def center_select(img_grad, img_label):
    minPix_xy = np.where(img_grad == img_grad.min())
    while True:
        pos = random.randint(0, len(minPix_xy[0]) - 1)
        if img_label[minPix_xy[0][pos], minPix_xy[1][pos]] != 1:
            return [minPix_xy[0][pos], minPix_xy[1][pos]]

def cal_Radius_1(img, center, purity, threshold, var_threshold):
    Rx = Ry = item_count = 0
    flag_x = flag_y = True
    temp_pixNum = 0
    center_value = int(img[center[0], center[1]])
    while True:
        if flag_x and flag_y:
            item_count += 1
        else:
            item_count = 1 if flag_x else 2
        if flag_x and item_count % 2 != 0:
            Rx += 1
        if flag_y and item_count % 2 == 0:
            Ry += 1
        left, right, up, down = cal_bound(img, center, Rx, Ry)
        pixNum = (down - up + 1) * (right - left + 1)
        if pixNum == temp_pixNum:
            return Rx, Ry, 1
        count = len(np.where(abs(np.int_(img[up:down + 1, left:right + 1]) - center_value) > threshold)[0])
        temp_purity = 1 - count / pixNum
        var = np.var(img[up:down + 1, left:right + 1])
        temp_pixNum = pixNum
        if temp_purity > purity and var < var_threshold:
            purity = min(purity * 1.005, 0.99)
            flag = True
        else:
            flag = False
        if not flag and item_count % 2 != 0:
            flag_x = False
            Rx -= 1
        if not flag and item_count % 2 == 0:
            flag_y = False
            Ry -= 1
        if not flag_x and not flag_y:
            return Rx, Ry, temp_purity

def calulate_weight(img, center_1, center_2):
    l1, r1, u1, d1 = cal_bound(img, center_1[0], center_1[1], center_1[2])
    l2, r2, u2, d2 = cal_bound(img, center_2[0], center_2[1], center_2[2])
    x_list = sorted([u1, u2, d1, d2])
    y_list = sorted([l1, l2, r1, r2])
    if x_list[2] < x_list[1] or y_list[2] < y_list[1]:
        return 0
    return (x_list[2] - x_list[1] + 1) * (y_list[2] - y_list[1] + 1)

def calulate_A_and_B(img, center_1, center_2):
    x1, x2, y1, y2 = cal_bound(img, center_1[0], center_1[1], center_1[2])
    x3, x4, y3, y4 = cal_bound(img, center_2[0], center_2[1], center_2[2])
    return (x2 - x1 + 1) * (y2 - y1 + 1) + (x4 - x3 + 1) * (y4 - y3 + 1)

def granular_balls_generate(img, purity_1=0.9, threshold=10, var_threshold_1=20, gray_thresh=20, knn_k=6):
    img_gray = cv2.cvtColor(cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0), cv2.COLOR_GRAY2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_label = np.zeros(img.shape)
    img_grad = get_grad(img)
    max_Grad = img_grad.max()
    h, w = img.shape
    center = []

    while 0 in img_label:
        temp_center = center_select(img_grad, img_label)
        Rx, Ry, _ = cal_Radius_1(img, temp_center, purity_1, threshold, var_threshold_1)
        left, right, up, down = cal_bound(img, temp_center, Rx, Ry)
        cx, cy = temp_center
        patch = img[max(cx - 1, 0): min(cx + 2, h), max(cy - 1, 0): min(cy + 2, w)]
        patch_flat = patch.flatten().astype(np.float32)
        patch_padded = np.pad(patch_flat, (0, 9 - len(patch_flat)), mode='constant', constant_values=0)
        region = img[up:down + 1, left:right + 1]
        center.append((temp_center,
                       Rx, Ry,
                       img.mean(),
                       img_grad[cx, cy],
                       region.max(), region.min(), region.mean(),
                       img[cx, cy],
                       *patch_padded,
                       math.sqrt(cx ** 2 + cy ** 2),
                       math.atan2(cy, cx),
                       img[cx, cy] / (region.mean() + 1e-5),
                       img.mean() - region.mean(),
                       cx * img[cx, cy],
                       cy * img.mean()))
        img_label[up:down + 1, left:right + 1] = 1
        img_grad[up:down + 1, left:right + 1] = max_Grad

    g = nx.Graph()
    for i in range(len(center)):
        g.add_node(str(i))

    for i in range(len(center)):
        for j in range(i + 1, len(center)):
            c1, c2 = center[i], center[j]
            if (abs(c1[0][0] - c2[0][0]) - 1) < c1[2] + c2[2] and (abs(c1[0][1] - c2[0][1]) - 1) < c1[1] + c2[1]:
                g.add_edge(str(i), str(j))

    positions = np.array([c[0] for c in center])
    gray_values = np.array([int(c[8]) for c in center], dtype=np.int16)
    knn = NearestNeighbors(n_neighbors=knn_k + 1).fit(positions)
    distances, indices = knn.kneighbors(positions)

    for i in range(len(center)):
        for j in indices[i][1:]:
            if abs(gray_values[i] - gray_values[j]) < gray_thresh:
                g.add_edge(str(i), str(j))

    adj = sp.coo_matrix(nx.to_numpy_array(g))
    adj = np.vstack((adj.row, adj.col))

    center_array = np.zeros((len(center), 25))
    center_ = np.zeros((len(center), 4))

    radius = normalize(np.array([c[-6] for c in center]))
    angle = (np.array([c[-5] for c in center]) + np.pi) / (2 * np.pi)
    pixel_ratio = normalize(np.array([c[-4] for c in center]))
    global_diff = normalize(np.array([c[-3] for c in center]))
    x_pixel_interact = normalize(np.array([c[-2] for c in center]))
    y_global_interact = normalize(np.array([c[-1] for c in center]))

    for id in range(len(center)):
        center_array[id, :19] = [
            center[id][0][0], center[id][0][1], center[id][1], center[id][2], center[id][3], center[id][4],
            center[id][5], center[id][6], center[id][7], center[id][8], center[id][9], center[id][10],
            center[id][11], center[id][12], center[id][13], center[id][14], center[id][15], center[id][16],
            center[id][17]
        ]
        center_array[id, 19:] = [
            radius[id], angle[id], pixel_ratio[id], global_diff[id], x_pixel_interact[id], y_global_interact[id]
        ]
        center_[id] = [center[id][0][0], center[id][0][1], center[id][1], center[id][2]]

    # 调用13维边特征生成函数
    edge_attr = compute_13d_edge_attr(center, adj, img, img_grad)

    # 获取增强的6维节点特征
    extra_feats, extra_names = extract_texture_entropy_edge(center, img)
    # 合并到节点特征矩阵
    center_array = np.hstack((center_array, extra_feats))


    return center_array, adj, edge_attr, center_






def extract_texture_entropy_edge(center, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    为每个节点提取额外6维特征：
    1. LBP 均值
    2. LBP 方差
    3. 区域熵
    4. 是否靠近图像边缘
    5. 中心梯度方向 atan2
    6. 区域与周围像素差均值

    :param center: 你的原始粒球结构（list of tuples）
    :param img: 原始灰度图像
    :return: 额外的 [N, 6] 特征矩阵，和新的字段说明
    """
    h, w = img.shape
    extra_feats = []

    for c in center:
        cx, cy = c[0]
        Rx, Ry = c[1], c[2]
        left, right, up, down = max(0, cy - Rx), min(w, cy + Rx + 1), max(0, cx - Ry), min(h, cx + Ry + 1)
        region = img[up:down, left:right]

        # 1-2. LBP 均值和方差
        lbp = local_binary_pattern(region, P=8, R=1, method='uniform')
        lbp_mean = lbp.mean()
        lbp_std = lbp.std()

        # 3. 区域熵
        hist, _ = np.histogram(region.flatten(), bins=256, range=(0, 256), density=True)
        region_entropy = scipy_entropy(hist + 1e-8, base=2)

        # 4. 是否靠近图像边界
        is_edge = int(cx < h * 0.05 or cx > h * 0.95 or cy < w * 0.05 or cy > w * 0.95)

        # 5. 中心点梯度方向
        grad_angle = math.atan2(int(img[min(cx + 1, h - 1), cy]) - int(img[max(cx - 1, 0), cy]),
                                int(img[cx, min(cy + 1, w - 1)]) - int(img[cx, max(cy - 1, 0)]))
        grad_angle = (grad_angle + np.pi) / (2 * np.pi)

        # 6. 区域灰度与图像其余区域灰度差均值
        full_mean = img.mean()
        diff_to_global = abs(region.mean() - full_mean) / 255.0

        extra_feats.append([lbp_mean, lbp_std, region_entropy / 10.0, is_edge, grad_angle, diff_to_global])

    return np.array(extra_feats), np.array(["lbp_mean", "lbp_std", "entropy", "is_edge", "grad_angle", "diff_to_global"])





def compute_13d_edge_attr(center, adj, img, img_grad):
    edge_attr = np.zeros((len(adj[0]), 13))
    h, w = img.shape

    for idx in range(len(adj[0])):
        i, j = adj[0][idx], adj[1][idx]
        c1, c2 = center[i], center[j]

        # 粒球边界
        l1, r1, u1, d1 = cal_bound(img, c1[0], c1[1], c1[2])
        l2, r2, u2, d2 = cal_bound(img, c2[0], c2[1], c2[2])
        x_start, x_end = max(u1, u2), min(d1, d2)
        y_start, y_end = max(l1, l2), min(r1, r2)

        # 1. overlap & gradient in overlap
        if x_end >= x_start and y_end >= y_start:
            inter = (x_end - x_start + 1) * (y_end - y_start + 1)
            inter_gray = img[x_start:x_end + 1, y_start:y_end + 1]
            inter_grad = img_grad[x_start:x_end + 1, y_start:y_end + 1]
        else:
            inter = 0
            inter_gray = np.array([])
            inter_grad = np.array([])

        union = (r1 - l1 + 1) * (d1 - u1 + 1) + (r2 - l2 + 1) * (d2 - u2 + 1)
        iou = inter / (union + 1e-8)

        # 2. 中心欧氏距离
        dist = math.sqrt((c1[0][0] - c2[0][0])**2 + (c1[0][1] - c2[0][1])**2)

        # 3. 粒心灰度差
        gray_diff = abs(np.int16(c1[8]) - np.int16(c2[8])) / 255.0

        # 4. 平均灰度差
        mean1 = img[u1:d1+1, l1:r1+1].mean()
        mean2 = img[u2:d2+1, l2:r2+1].mean()
        mean_diff = abs(mean1 - mean2) / 255.0

        # 5. 梯度差（粒心）
        grad_diff = abs(np.int16(c1[4]) - np.int16(c2[4])) / (max(c1[4], c2[4], 1e-5))

        # 6. 连线角度
        dx = c2[0][1] - c1[0][1]
        dy = c2[0][0] - c1[0][0]
        angle = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi)

        # 7. 半径比
        r1_avg = (c1[1] + c1[2]) / 2
        r2_avg = (c2[1] + c2[2]) / 2
        radius_ratio = min(r1_avg, r2_avg) / (max(r1_avg, r2_avg) + 1e-5)

        # 8. 共享邻居数（占位，设为 0）
        shared_neighbors = 0

        # 9. 相交区域灰度方差
        var_inter = np.var(inter_gray) / (255.0**2) if inter_gray.size > 0 else 0

        # 10. LBP 差异
        region1 = img[u1:d1+1, l1:r1+1]
        region2 = img[u2:d2+1, l2:r2+1]
        lbp1 = local_binary_pattern(region1, P=8, R=1, method='uniform')
        lbp2 = local_binary_pattern(region2, P=8, R=1, method='uniform')
        h1, _ = np.histogram(lbp1, bins=10, range=(0, 10), density=True)
        h2, _ = np.histogram(lbp2, bins=10, range=(0, 10), density=True)
        lbp_diff = np.sum(np.abs(h1 - h2)) / 2.0

        # 11. 熵差异
        r1_flat = region1.flatten()
        r2_flat = region2.flatten()
        h1, _ = np.histogram(r1_flat, bins=256, range=(0, 256), density=True)
        h2, _ = np.histogram(r2_flat, bins=256, range=(0, 256), density=True)
        e1 = -np.sum(h1 * np.log2(h1 + 1e-10))
        e2 = -np.sum(h2 * np.log2(h2 + 1e-10))
        entropy_diff = abs(e1 - e2) / 10.0

        # 12. overlap 区域平均梯度
        grad_mean = inter_grad.mean() / (img_grad.max() + 1e-8) if inter_grad.size > 0 else 0

        # 13. 是否靠近图像边缘
        is_edge = int(
            min(l1, l2) < w * 0.05 or
            max(r1, r2) > w * 0.95 or
            min(u1, u2) < h * 0.05 or
            max(d1, d2) > h * 0.95
        )

        edge_attr[idx] = [
            iou, dist / math.sqrt(h**2 + w**2),
            gray_diff, mean_diff, grad_diff,
            angle, radius_ratio, shared_neighbors,
            var_inter, lbp_diff, entropy_diff,
            grad_mean, is_edge
        ]

    # 对部分特征进行归一化
    for col in [0, 1, 3, 4, 8, 9, 10, 11, 12]:
        maxv, minv = edge_attr[:, col].max(), edge_attr[:, col].min()
        if maxv > minv:
            edge_attr[:, col] = (edge_attr[:, col] - minv) / (maxv - minv + 1e-8)

    return edge_attr
