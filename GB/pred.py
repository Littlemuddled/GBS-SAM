import argparse
import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from img2graph import granular_balls_generate, cal_bound
from model import GCN_8_plus

# 设置随机种子以确保一致性
random.seed(0)
torch.manual_seed(0)

def visualize_pixel_matrix(pixel_matrix, title='gray Image'):
    plt.figure(figsize=(6, 3))
    plt.imshow(pixel_matrix, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

colors = np.array([
    [0, 0, 0],         # 0: 背景（黑色）
    [255, 0, 0],       # 1: 红色
    [0, 255, 0],       # 2: 绿色
    [0, 0, 255],       # 3: 蓝色
    [255, 255, 0],     # 4: 黄色
    [255, 0, 255],     # 5: 品红
    [0, 255, 255],     # 6: 青色
    [255, 105, 180],  # 16: 粉红色
    [128, 0, 0],  # 7: 深红色
    [0, 128, 0],       # 8: 深绿色
    [0, 0, 128],       # 9: 深蓝色
    [128, 128, 0],     # 10: 深黄色
    [128, 0, 128],     # 11: 深品红
    [0, 128, 128],     # 12: 深青色
    [192, 192, 192],   # 13: 银色
    [128, 128, 128],   # 14: 灰色
    [255, 165, 0],     # 15: 橙色
    [128, 0, 128],     # 17: 深紫色
], dtype=np.uint8)

def read_npz(img_path):
    """
    从预生成的.npz文件中加载粒球图数据。
    :param img_path: 图像文件路径
    :return: center_array, adj, edge_attr, center_ 或 None（如果文件不存在）
    """
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    npz_path = os.path.join(r"E:\dataset\SPINE\val\granular_features25_3\images_256_npz", base_name + ".npz")
    if not os.path.exists(npz_path):
        print(f"❌ 文件不存在: {npz_path}")
        return None
    data = np.load(npz_path)
    return data['center_array'], data['adj'], data['edge_attr'], data['center_']


def create_graph_data(img_path, purity=0.9, threshold=10, var_threshold=20):
    """
    创建图数据对象，用于预测。
    :param mask: 掩码（numpy数组，可选）
    :param purity, threshold, var_threshold: 粒球生成参数
    :return: PyTorch Geometric Data对象
    """
    # 尝试加载预生成的.npz文件
    npz_data = read_npz(img_path)
    if npz_data is not None:
        center_array, adj, edge_attr, center_ = npz_data
    else:
        img_data = cv2.imread(img_path)
        if img_data is None:
            raise ValueError(f"无法加载图像: {img_path}")
        center_array, adj, edge_attr, center_ = granular_balls_generate(
            img_data, purity, threshold, var_threshold
        )

    # 转换为PyTorch Geometric格式
    x = torch.tensor(center_array, dtype=torch.float)
    edge_index = torch.tensor(adj, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # 如果提供了掩码，生成标签（用于评估）
    y = None
    num_classes = None

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_classes=num_classes)
    data.center_ = center_  # 保存center_用于可视化
    return data


def predict_image(model, data, device):
    """
    对单个图数据进行预测。
    :param model: 训练好的GCN模型
    :param data: PyTorch Geometric Data对象
    :param device: 计算设备
    :return: 预测类别（numpy数组）
    """
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1).cpu().numpy()
    return pred


def predictions_matrix(img, center_, pred_labels):
    """
    生成两个掩码矩阵：
    - point_matrix: 粒球中心点为类别值
    - region_matrix: 粒球区域赋值为类别值
    所有矩阵大小与输入图像相同。

    :param img: 输入图像 (H, W, 3)
    :param center_: 粒球中心信息列表，每个元素为 (x, y, Rx, Ry, ...)
    :param pred_labels: 每个粒球的预测类别（list 或 numpy数组）
    :return: point_matrix, region_matrix
    """
    h, w = img.shape[:2]
    point_matrix = np.zeros((h, w), dtype=np.uint8)
    region_matrix = np.zeros((h, w), dtype=np.uint8)

    for i, (x, y, Rx, Ry, *_) in enumerate(center_):
        x, y = int(x), int(y)
        Rx, Ry = int(Rx), int(Ry)
        cls = int(pred_labels[i])

        # 中心点赋值
        if 0 <= x < h and 0 <= y < w:
            point_matrix[x, y] = cls

        # 区域赋值
        left = max(0, y - Rx)
        right = min(w - 1, y + Rx)
        up = max(0, x - Ry)
        down = min(h - 1, x + Ry)
        region_matrix[up:down + 1, left:right + 1] = cls

    return point_matrix, region_matrix



def visualize_predictions(img_path, point_matrix, region_matrix, save_path, base_name):

    output_img_P_L = os.path.join(save_path, base_name + "_P_L.png")
    output_img_P_RGB = os.path.join(save_path, base_name + "_P_RGB.png")
    output_img_L = os.path.join(save_path, base_name + "_L.png")
    output_img_RGB = os.path.join(save_path, base_name + "_RGB.png")

    # 保存为灰度图（L 模式）   点图
    Image.fromarray(point_matrix * 40, mode="L").save(output_img_P_L)
    Image.fromarray(region_matrix * 40, mode="L").save(output_img_L)


    new_path = save_path.replace("975_11_predictions", "975_11_mask_pred")
    os.makedirs(new_path, exist_ok=True)
    Image.fromarray(region_matrix, mode="L").save(os.path.join(new_path, base_name + ".png"))



    ## 保存高质量掩码
    image = Image.open(img_path)
    image = np.array(image.convert("RGB"))

    color_mask = colors[point_matrix]
    # 将彩色点图叠加到原图上
    # 假设 img_point 中值为 0 的像素是背景，不覆盖原图
    vis_image = image.copy()  # 复制原图
    mask = point_matrix != 0  # 非背景像素的掩码
    vis_image[mask] = color_mask[mask]  # 将非背景像素替换为彩色点
    # 保存叠加后的图像
    Image.fromarray(vis_image, mode='RGB').save(output_img_P_RGB)

    color_mask = colors[region_matrix]
    vis_image = image.copy()  # 复制原图
    mask = region_matrix != 0  # 非背景像素的掩码
    vis_image[mask] = color_mask[mask]  # 将非背景像素替换为彩色点
    # 保存叠加后的图像
    Image.fromarray(vis_image, mode='RGB').save(output_img_RGB)

    print(f"已保存点图像: {output_img_L}")
    print(f"已保存点图像: {output_img_P_L}")
    print(f"已保存叠加图像: {output_img_P_RGB}")
    print(f"已保存叠加图像: {output_img_RGB}")




def main():
    parser = argparse.ArgumentParser(description='GCN Batch Prediction for Spine CT Multiclass Classification')
    parser.add_argument('--image_dir', default=r"E:\dataset\SPINE\val\granular_features25_3\images_256", type=str, help='输入CT图像目录')
    parser.add_argument('--model_path', type=str, default='checkpoint/best_model_975_11.pth', help='模型权重路径')
    parser.add_argument('--purity', type=float, default=0.9, help='粒球纯度')
    parser.add_argument('--threshold', type=float, default=10, help='异类点阈值')
    parser.add_argument('--var_threshold', type=float, default=20, help='方差阈值')
    parser.add_argument('--output_dir', type=str, default=r'E:\dataset\SPINE\val\granular_features25_3/975_11_predictions', help='输出目录')
    args = parser.parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    num_classes = 12  #（需根据训练数据调整）
    model = GCN_8_plus(num_features=25, num_classes=num_classes, initdim=16, inithead=16, edge_dim=3).to(device)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"成功加载模型权重: {args.model_path}")
    except Exception as e:
        print(f"加载模型权重失败: {e}")
        return

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取图像文件列表
    image_files = sorted([f for f in os.listdir(args.image_dir) if f.endswith(('.png', '.jpg'))])
    if not image_files:
        print(f"图像目录 {args.image_dir} 中没有找到.png或.jpg文件")
        return


    # 批量预测
    for img_file in tqdm(image_files, desc="预测图像"):
        img_path = os.path.join(args.image_dir, img_file)

        # 加载图像和掩码
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法加载图像: {img_path}")
            continue

        # 创建图数据
        try:
            data = create_graph_data(img_path, args.purity, args.threshold, args.var_threshold)
        except ValueError as e:
            print(f"处理图像 {img_file} 失败: {e}")
            continue

        # 进行预测
        pred_labels = predict_image(model, data, device)

        # 保存预测结果
        base_name = os.path.splitext(img_file)[0]
        # output_path = os.path.join(args.output_dir, f"{base_name}_pred.npy")
        # np.save(output_path, pred_labels)
        # print(f"预测标签已保存至: {output_path}")

        # 可视化
        point_matrix, region_matrix = predictions_matrix(img, data.center_, pred_labels)

        visualize_predictions(img_path, point_matrix, region_matrix, args.output_dir, base_name)
        print(f"预测可视化已保存至: {args.output_dir}")



if __name__ == '__main__':
    main()