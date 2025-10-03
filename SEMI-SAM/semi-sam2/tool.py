import csv
import datetime
import json
import time
import os
import cv2
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from finch import FINCH


np.random.seed(0)

colors = np.array([
    [0, 0, 0],         # 0: 背景（黑色）
    [255, 0, 0],       # 1: 红色
    [0, 255, 0],       # 2: 绿色
    [0, 0, 255],       # 3: 蓝色
    [255, 255, 0],     # 4: 黄色
    [255, 0, 255],     # 5: 品红
    [0, 255, 255],     # 6: 青色
    [128, 0, 0],       # 7: 深红色
    [0, 128, 0],       # 8: 深绿色
    [0, 0, 128],       # 9: 深蓝色
    [128, 128, 0],     # 10: 深黄色
    [128, 0, 128],     # 11: 深品红
    [0, 128, 128],     # 12: 深青色
    [192, 192, 192],   # 13: 银色
    [128, 128, 128],   # 14: 灰色
    [255, 165, 0],     # 15: 橙色
    [255, 105, 180],   # 16: 粉红色
    [128, 0, 128],     # 17: 深紫色
], dtype=np.uint8)


def load_image_mask_points(image_dir, mask_dir, json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    dataset = []
    for entry in data:
        image_name = entry["image"]
        if image_name.startswith("ISIC"):
            image_path = os.path.join(image_dir, image_name + ".jpg")  # TODO
        else:
            image_path = os.path.join(image_dir, image_name + ".png") # TODO
        mask_path = os.path.join(mask_dir, image_name + ".png")

        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"[Warning] 图像未找到: {image_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 读取掩码
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[Warning] 掩码未找到: {mask_path}")
            continue

        # 点提示
        points = entry.get("points", [])

        dataset.append({
            "image": image,
            "mask": mask,
            "points": points  # [{"class": 1, "coord": [x, y]}, ...]
        })
    return dataset


def read_batch2(entry, image_root, target_size=1024):
    image_name = entry["image"]
    if image_name.startswith("ISIC"):
        image_path = os.path.join(image_root, image_name + ".jpg")
    else:
        image_path = os.path.join(image_root, image_name + ".png")  # 根据实际情况调整扩展名

    # === 加载图像和掩码 ===
    img = cv2.imread(image_path)
    if img is None:
        print(f"[Error] 读取图像失败: {image_path}")
        return None, None, None, 0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # === 缩放 ===
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # === Padding 到 1024×1024 ===
    pad_w = target_size - new_w
    pad_h = target_size - new_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    img_padded = cv2.copyMakeBorder(img_resized, pad_top, pad_bottom, pad_left, pad_right,borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # === 构建 masks 和 points ===
    points = []
    for p in entry["points"]:
        cls = p["class"]
        x, y = p["coord"]

        # 缩放 + padding 后的新点位置
        x_new = int(x * scale + pad_left)
        y_new = int(y * scale + pad_top)

        points.append([[x_new, y_new]])

    points = np.array(points)              # [N, 1, 2]

    # return img_padded, masks, points, len(masks), scale, pad_left, pad_top
    return img_padded, points, len(points)


def read_batch(entry, target_size=1024):
    image = entry["image"]
    mask = entry["mask"]

    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # === 缩放 ===
    img_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # === Padding 到 1024×1024 ===
    pad_w = target_size - new_w
    pad_h = target_size - new_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    img_padded = cv2.copyMakeBorder(img_resized, pad_top, pad_bottom, pad_left, pad_right,
                                    borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    mask_padded = cv2.copyMakeBorder(mask_resized, pad_top, pad_bottom, pad_left, pad_right,
                                     borderType=cv2.BORDER_CONSTANT, value=0)

    # === 构建 masks 和 points ===
    masks = []
    points = []
    for p in entry["points"]:
        cls = p["class"]
        x, y = p["coord"]

        # 缩放 + padding 后的新点位置
        x_new = int(x * scale + pad_left)
        y_new = int(y * scale + pad_top)

        mask = (mask_padded == cls).astype(np.uint8)
        if mask.sum() == 0:
            continue
        masks.append(mask[np.newaxis, ...])
        points.append([[x_new, y_new]])

    if not masks:
        return None, None, None, 0

    masks = np.concatenate(masks, axis=0)  # [N, H, W]
    points = np.array(points)              # [N, 1, 2]

    # return img_padded, masks, points, len(masks), scale, pad_left, pad_top
    return img_padded, masks, points, len(masks)






"""
[1: [坐标], 2:[坐标], 3:[坐标], 4:[坐标], 5:[坐标], 6:[坐标], 7:[坐标], 8:[坐标], 9:[坐标], 10:[坐标], 11:[坐标]]

ann_map = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
inds = np.unique(ann_map)[1:]
for ind in inds:
    mask = (ann_map == ind).astype(np.uint8)
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        print(f"{mask_path}没有点")
        continue
    yx = coords[np.random.randint(len(coords))]
    points.append({ind:[yx[1], yx[0]})
"""


# ==== 伪标签生成 ====
# def generate_pseudo_labels(predictor, unlabeled_images_dir, json_path, output_dir, score_threshold=0.9):
#     predictor.model.eval()
#     os.makedirs(output_dir, exist_ok=True)
#     os.makedirs(output_dir + "_RGB", exist_ok=True)
#
#     unlabeled_images = os.listdir(unlabeled_images_dir)
#
#     # 加载提示点数据
#     with open(json_path, "r") as f:
#         data_entries = json.load(f)
#
#     pseudo_masks_data = []
#     for entry in tqdm(data_entries, desc="Generating Pseudo Labels"):
#         if entry['image'] + '.png' not in unlabeled_images:
#             continue
#         # 读取图像和提示点
#         image, input_points, num_instances = read_batch2(entry, image_root=unlabeled_images_dir)
#         if image is None or num_instances == 0:
#             continue
#
#         input_labels = np.ones((num_instances, 1), dtype=np.int32)
#
#         with torch.no_grad():
#             predictor.set_image(image)
#             mask_input, unnorm_coords, labels, _ = predictor._prep_prompts(
#                 input_points, input_labels, box=None, mask_logits=None, normalize_coords=True
#             )
#             if unnorm_coords is None or unnorm_coords.shape[0] == 0:
#                 continue
#
#             sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
#                 points=(unnorm_coords, labels), boxes=None, masks=None
#             )
#             batched_mode = unnorm_coords.shape[0] > 1
#             high_res_features = [feat[-1].unsqueeze(0) for feat in predictor._features["high_res_feats"]]
#
#             low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
#                 image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
#                 image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
#                 sparse_prompt_embeddings=sparse_embeddings,
#                 dense_prompt_embeddings=dense_embeddings,
#                 multimask_output=True,
#                 repeat_image=batched_mode,
#                 high_res_features=high_res_features,
#             )
#
#             prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
#             prd_masks = torch.sigmoid(prd_masks[:, 0])
#
#         # 构建伪标签掩码图
#         pseudo_masks = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
#         tag = False
#         for i in range(num_instances):
#             if prd_scores[i, 0] > score_threshold:
#                 tag = True
#                 binary_mask = (prd_masks[i] > 0.5).cpu().numpy().astype(np.uint8)
#                 if binary_mask.sum() > 2000:
#                     pseudo_masks[(pseudo_masks == 0) & (binary_mask == 1)] = i + 1
#                 else:
#                     print(f"[Skip] 掩码 {i} 太小，跳过：{binary_mask.sum()} 像素")
#
#         if tag:  # 有得分超过  0.9
#             base_name = entry["image"] + ".png"
#             mask_path = os.path.join(output_dir, base_name)
#
#             ## 保存高质量掩码
#             vis_image = image.copy()
#             color_mask = colors[pseudo_masks]
#             mask = pseudo_masks > 0
#             vis_image[mask] = color_mask[mask]
#             mask_path_RGB = os.path.join(output_dir + "_RGB", base_name)
#             cv2.imwrite(mask_path_RGB, vis_image[..., ::-1])
#
#             pseudo_masks = cv2.resize(pseudo_masks, (512, 512), interpolation=cv2.INTER_NEAREST)
#             cv2.imwrite(mask_path, pseudo_masks)
#             pseudo_masks_data.append({"image": os.path.join(unlabeled_images_dir,base_name), "annotation": mask_path})
#
#     return pseudo_masks_data


def generate_pseudo_labels(predictor, unlabeled_images_dir, json_path, output_dir, score_threshold=0.9):
    predictor.model.eval()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir + "_RGB", exist_ok=True)

    unlabeled_images = os.listdir(unlabeled_images_dir)

    # 加载提示点数据
    with open(json_path, "r") as f:
        data_entries = json.load(f)

    dataset = []
    for entry in tqdm(data_entries, desc="生成伪标签"):
        if entry['image'].startswith("ISIC"):
            if entry['image'] + '.jpg' not in unlabeled_images:
                continue
        else:
            if entry['image'] + '.png' not in unlabeled_images:
                continue

        # 读取图像和提示点
        image, input_points, num_instances = read_batch2(entry, image_root=unlabeled_images_dir)
        if image is None or num_instances == 0:
            continue

        input_labels = np.ones((num_instances, 1), dtype=np.int32)

        with torch.no_grad():
            predictor.set_image(image)
            mask_input, unnorm_coords, labels, _ = predictor._prep_prompts(
                input_points, input_labels, box=None, mask_logits=None, normalize_coords=True
            )
            if unnorm_coords is None or unnorm_coords.shape[0] == 0:
                continue

            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=(unnorm_coords, labels), boxes=None, masks=None
            )
            batched_mode = unnorm_coords.shape[0] > 1
            high_res_features = [feat[-1].unsqueeze(0) for feat in predictor._features["high_res_feats"]]

            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=batched_mode,
                high_res_features=high_res_features,
            )

            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
            prd_masks = torch.sigmoid(prd_masks[:, 0])

        # 构建伪标签掩码图
        pseudo_masks = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        tag = False
        valid_classes = []
        valid_classes_point = []
        for i in range(num_instances):
            if prd_scores[i, 0] > score_threshold:
                tag = True
                binary_mask = (prd_masks[i] > 0.5).cpu().numpy().astype(np.uint8)
                if binary_mask.sum() > 2000:
                    pseudo_masks[(pseudo_masks == 0) & (binary_mask == 1)] = i + 1
                    valid_classes.append(i + 1)  # 记录有效类别
                    valid_classes_point.append(entry['points'][i])  # 记录有效点
                else:
                    print(f"[跳过] 掩码 {i} 太小，跳过：{binary_mask.sum()} 像素")

        if tag:  # 有得分超过 score_threshold
            if entry['image'].startswith("ISIC"):
                base_name = entry["image"] + ".jpg"
            else:
                base_name = entry["image"] + ".png"
            mask_path = os.path.join(output_dir, entry["image"] + ".png")

            # 保存高质量掩码
            image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
            pseudo_masks = cv2.resize(pseudo_masks, (512, 512), interpolation=cv2.INTER_NEAREST)

            # 保存掩码图
            vis_image = image.copy()
            color_mask = colors[pseudo_masks]
            mask = pseudo_masks > 0
            vis_image[mask] = color_mask[mask]
            mask_path_RGB = os.path.join(output_dir + "_RGB", base_name)

            cv2.imwrite(mask_path_RGB, vis_image[..., ::-1])
            cv2.imwrite(mask_path, pseudo_masks)

            # 添加到数据集
            dataset.append({
                "image": image,
                "mask": pseudo_masks,
                "points": valid_classes_point
            })

    return dataset



# ==== 验证函数 ====
# def evaluate_validation_set(predictor, val_data_list):
#     predictor.model.eval()
#
#     total_iou, total_dice, count = 0, 0, 0
#     with torch.no_grad():
#         for ent in tqdm(val_data_list, desc="Validation", total=len(val_data_list), leave=False, position=0):
#             image, masks, input_points, num_instances = read_batch(ent)
#             if image is None or masks is None or num_instances == 0:
#                 continue
#             input_labels = np.ones((num_instances, 1))
#             if input_points.size == 0 or input_labels.size == 0:
#                 continue
#             predictor.set_image(image)
#             mask_input, unnorm_coords, labels, _ = predictor._prep_prompts(
#                 input_points, input_labels, box=None, mask_logits=None, normalize_coords=True
#             )
#             if unnorm_coords is None or unnorm_coords.shape[0] == 0:
#                 continue
#             sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
#                 points=(unnorm_coords, labels), boxes=None, masks=None
#             )
#             batched_mode = unnorm_coords.shape[0] > 1
#             high_res_features = [feat[-1].unsqueeze(0) for feat in predictor._features["high_res_feats"]]
#             low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
#                 image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
#                 image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
#                 sparse_prompt_embeddings=sparse_embeddings,
#                 dense_prompt_embeddings=dense_embeddings,
#                 multimask_output=True,
#                 repeat_image=batched_mode,
#                 high_res_features=high_res_features,
#             )
#             prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
#             prd_masks = torch.sigmoid(prd_masks[:, 0])
#             gt_masks = torch.tensor(masks.astype(np.float32)).cuda()
#             iou, dice = 0, 0
#             for i in range(num_instances):
#                 prd_mask = prd_masks[i:i+1]
#                 gt_mask = gt_masks[i:i+1]
#                 inter = (gt_mask * (prd_mask > 0.5)).sum()
#                 union = gt_mask.sum() + (prd_mask > 0.5).sum() - inter
#                 iou += inter / (union + 1e-6)
#                 dice_inter = (gt_mask * (prd_mask > 0.5)).sum()
#                 dice += (2 * dice_inter + 1e-6) / (gt_mask.sum() + (prd_mask > 0.5).sum() + 1e-6)
#             iou = iou / max(1, num_instances)
#             dice = dice / max(1, num_instances)
#             total_iou += iou.item()
#             total_dice += dice.item()
#             count += 1
#     predictor.model.train()
#     return total_iou / max(1, count), total_dice / max(1, count)




# ==== 验证函数（扩展指标）====
def evaluate_validation_set(predictor, val_data_list):
    """
    验证集评估函数，计算 IoU, Dice, Precision, Recall, F1
    不计算 AUC
    """
    predictor.model.eval()

    total_iou, total_dice = 0, 0
    total_precision, total_recall, total_f1 = 0, 0, 0
    count = 0

    with torch.no_grad():
        for ent in tqdm(val_data_list, desc="Validation", total=len(val_data_list), leave=False, position=0):
            image, masks, input_points, num_instances = read_batch(ent)
            if image is None or masks is None or num_instances == 0:
                continue
            input_labels = np.ones((num_instances, 1))
            if input_points.size == 0 or input_labels.size == 0:
                continue

            # === 前向传播 ===
            predictor.set_image(image)
            mask_input, unnorm_coords, labels, _ = predictor._prep_prompts(
                input_points, input_labels, box=None, mask_logits=None, normalize_coords=True
            )
            if unnorm_coords is None or unnorm_coords.shape[0] == 0:
                continue
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=(unnorm_coords, labels), boxes=None, masks=None
            )
            batched_mode = unnorm_coords.shape[0] > 1
            high_res_features = [feat[-1].unsqueeze(0) for feat in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=batched_mode,
                high_res_features=high_res_features,
            )
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
            prd_probs = torch.sigmoid(prd_masks[:, 0])  # 概率图
            prd_binary = (prd_probs > 0.5).float()      # 二值图

            gt_masks = torch.tensor(masks.astype(np.float32)).cuda()
            iou = dice = precision = recall = 0

            # === 按实例统计指标 ===
            for i in range(num_instances):
                prd_mask = prd_binary[i]
                prd_prob = prd_probs[i]
                gt_mask = gt_masks[i]

                # IoU
                inter = (gt_mask * prd_mask).sum()
                union = gt_mask.sum() + prd_mask.sum() - inter
                iou += inter / (union + 1e-6)

                # Dice
                dice += (2 * inter + 1e-6) / (gt_mask.sum() + prd_mask.sum() + 1e-6)

                # Precision & Recall
                tp = inter
                fp = prd_mask.sum() - tp
                fn = gt_mask.sum() - tp
                precision += tp / (tp + fp + 1e-6)
                recall += tp / (tp + fn + 1e-6)

            # === 平均化每个实例的指标 ===
            num = max(1, num_instances)
            avg_prec = (precision / num).item()
            avg_rec = (recall / num).item()
            f1 = (2 * avg_prec * avg_rec) / (avg_prec + avg_rec + 1e-6)

            total_iou += (iou / num).item()
            total_dice += (dice / num).item()
            total_precision += avg_prec
            total_recall += avg_rec
            total_f1 += f1
            count += 1

    predictor.model.train()

    return (total_iou / max(1, count),
            total_dice / max(1, count),
            total_precision / max(1, count),
            total_recall / max(1, count),
            total_f1 / max(1, count))


def compute_mmd(x, y, kernel='rbf', sigma=1.0):
    """
    计算 MMD 损失 (Maximum Mean Discrepancy)
    x: [N, C] 或 [C]，目标特征
    y: [M, C] 或 [C]，源特征
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)  # [1, C]
    if y.dim() == 1:
        y = y.unsqueeze(0)  # [1, C]

    xx = torch.mm(x, x.t())  # [N, N]
    yy = torch.mm(y, y.t())  # [M, M]
    xy = torch.mm(x, y.t())  # [N, M]

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    # RBF kernel
    K_xx = torch.exp(- (rx.t() + rx - 2*xx) / (2*sigma**2))
    K_yy = torch.exp(- (ry.t() + ry - 2*yy) / (2*sigma**2))
    K_xy = torch.exp(- (rx.t() + ry - 2*xy) / (2*sigma**2))

    mmd = K_xx.mean() + K_yy.mean() - 2*K_xy.mean()
    return mmd





# ==== 边界损失 ====
def boundary_loss(pred, gt):
    # 横向梯度：W 方向变化
    dx_pred = torch.abs(pred[:, :, 1:] - pred[:, :, :-1])     # [B, H, W-1]
    dx_gt   = torch.abs(gt[:, :, 1:] - gt[:, :, :-1])         # [B, H, W-1]

    # 纵向梯度：H 方向变化
    dy_pred = torch.abs(pred[:, 1:, :] - pred[:, :-1, :])     # [B, H-1, W]
    dy_gt   = torch.abs(gt[:, 1:, :] - gt[:, :-1, :])         # [B, H-1, W]

    # 为了统一 shape，我们可以将 dx 和 dy 裁成 [B, H-1, W-1]
    dx_pred = dx_pred[:, :-1, :]  # -> [B, H-1, W-1]
    dx_gt   = dx_gt[:, :-1, :]
    dy_pred = dy_pred[:, :, :-1]  # -> [B, H-1, W-1]
    dy_gt   = dy_gt[:, :, :-1]

    pred_grad = dx_pred + dy_pred
    gt_grad = dx_gt + dy_gt

    return torch.mean(torch.abs(pred_grad - gt_grad))


def distillation_loss(student_logits, teacher_logits, temperature=1.0, alpha=0.5):
    """
    计算学生模型和教师模型之间的蒸馏损失。

    参数:
        student_logits: 学生模型的输出logits，形状为 [B, H, W]
        teacher_logits: 教师模型的输出logits，形状为 [B, H, W]
        temperature: 蒸馏温度，用于软化概率分布
        alpha: 平衡软目标和硬目标的权重参数

    返回:
        标量张量，表示计算得到的蒸馏损失
    """
    # 确保输入张量具有相同的形状
    assert student_logits.shape == teacher_logits.shape, "学生和教师的logits形状必须相同"

    # 计算软标签的KL散度损失
    soft_student_probs = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    soft_loss = F.kl_div(soft_student_probs, soft_teacher_probs, reduction='batchmean') * (temperature ** 2)

    # 计算硬标签的交叉熵损失（使用教师的预测作为目标）
    hard_targets = teacher_logits.argmax(dim=1)
    hard_loss = F.cross_entropy(student_logits, hard_targets)

    # 结合软损失和硬损失
    total_loss = alpha * soft_loss + (1 - alpha) * hard_loss

    return total_loss


def cosine_similarity(vec1, vec2, device):
    vec1 = torch.tensor(vec1).to(device).float()
    vec2 = torch.tensor(vec2).to(device).float()
    dot_product = torch.dot(vec1, vec2)
    norm1 = torch.norm(vec1)
    norm2 = torch.norm(vec2)
    similarity = dot_product / (norm1 * norm2)
    return similarity
def Matching_Loss(pred_prototypes, target_prototypes, device):
    """
    pred_prototypes: list of predicted prototypes
    target_prototypes: list of target prototypes
    """
    num_pred = len(pred_prototypes)
    num_target = len(target_prototypes)

    num = min(num_pred, num_target)
    cost_matrix = torch.zeros((num_pred, num_target))
    for i, pred_proto in enumerate(pred_prototypes):
        for j, target_proto in enumerate(target_prototypes):
            cos_sim = cosine_similarity(pred_proto, target_proto, device)
            cost_matrix[i, j] = 1 - cos_sim

    row_indices, col_indices = linear_sum_assignment(cost_matrix.numpy())

    total_loss = 0
    for row, col in zip(row_indices, col_indices):
        total_loss += cost_matrix[row, col].item()

    return total_loss / len(row_indices)


# ==== 可视化函数 ====
def visualize_pixel_matrix(pixel_matrix, title='gray Image'):
    plt.figure(figsize=(6, 3))
    plt.imshow(pixel_matrix, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def show_image(image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('on')
    plt.show()


def vis_evaluate(save_dir, steps_list, loss_list, train_iou_list, train_dice_list, val_iou_list, val_dice_list, val_precision_list, val_recall_list, val_f1_list):
    # === 1. Loss 曲线 ===
    plt.figure()
    plt.plot(steps_list, loss_list, label="Loss", color="blue")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

    # === 2. IoU & Dice (Train vs Val) ===
    plt.figure()
    plt.plot(steps_list, train_iou_list, label="Train IoU", color="orange")
    plt.plot(steps_list, val_iou_list, label="Val IoU", color="red")
    plt.plot(steps_list, train_dice_list, label="Train Dice", color="green")
    plt.plot(steps_list, val_dice_list, label="Val Dice", color="purple")
    plt.xlabel("Step")
    plt.ylabel("Score")
    plt.title("IoU & Dice (Train vs Val)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, "iou_dice_curve.png"))
    plt.close()

    # === 3. Val Precision, Recall, F1 ===
    plt.figure()
    plt.plot(steps_list, val_precision_list, label="Val Precision", color="blue")
    plt.plot(steps_list, val_recall_list, label="Val Recall", color="orange")
    plt.plot(steps_list, val_f1_list, label="Val F1", color="green")
    plt.xlabel("Step")
    plt.ylabel("Score")
    plt.title("Validation Metrics (Precision, Recall, F1)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, "val_metrics_curve.png"))
    plt.close()

