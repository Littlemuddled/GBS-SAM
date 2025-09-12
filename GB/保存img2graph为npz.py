import os
import numpy as np
import cv2
# from img2graph222 import granular_balls_generate
from img2graph import granular_balls_generate

def process_and_save_granular_features(image_dir, output_dir, purity=0.9, threshold=10, var_threshold=20):
    """
    对 image_dir 下的每张图像运行 granular_balls_generate 并保存结果到 output_dir
    每个文件保存为 .npz 格式，包含 center_array, adj, edge_attr, center_
    """
    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for file in image_files:
        image_path = os.path.join(image_dir, file)
        img = cv2.imread(image_path)

        if img is None:
            print(f"❌ 无法读取图像: {image_path}")
            continue

        print(f"✅ 正在处理图像: {file}")
        center_array, adj, edge_attr, center_ = granular_balls_generate(img, purity, threshold, var_threshold)

        # 保存为 .npz 文件
        base_name = os.path.splitext(file)[0]
        save_path = os.path.join(output_dir, base_name + ".npz")
        np.savez_compressed(save_path,
                            center_array=center_array,
                            adj=adj,
                            edge_attr=edge_attr,
                            center_=center_)
        print(f"✅ 已保存特征: {save_path}")


# def read_npz():
#     data = np.load('data/train/granular_features/Patient_001_I5000000.npz')
#     center_array = data['center_array']
#     adj = data['adj']
#     edge_attr = data['edge_attr']
#     center_ = data['center_']
#
#     print("center_array shape:", center_array.shape)




if __name__ == '__main__':
    image_dir = r'E:\dataset\SPINE\train\granular_features25_3\images_256'
    output_dir = r'E:\dataset\SPINE\train\granular_features25_3\images_256_npz'
    process_and_save_granular_features(image_dir, output_dir)

    # read_npz()  # 测试读取 npz 文件