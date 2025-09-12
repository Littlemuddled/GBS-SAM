import csv
import datetime
import random
import time
import os
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm
from collections import deque

from finch import FINCH
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tool import read_batch, generate_pseudo_labels, boundary_loss, evaluate_validation_set, vis_evaluate, Matching_Loss, load_image_mask_points

np.random.seed(0)

def train_stage2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # ==== 配置路径 ====
    checkpoint_path = r"../checkpoints\sam2.1_hiera_small.pt"
    model_cfg = r"configs\sam2.1_hiera_s.yaml"
    data_dir_1 = r"E:\dataset\GB_SEMI_SAM\SPINE_SEMI\stage1"
    data_dir = r"E:\dataset\GB_SEMI_SAM\SPINE_SEMI\stage2"
    data_dir = r"E:\dataset\GB_SEMI_SAM\ISIC2016_SEMI\stage2"
    data_dir = r"E:\dataset\GB_SEMI_SAM\COVID_SEMI\stage2"

    images_dir_train = os.path.join(data_dir_1, 'train', 'images')
    masks_dir_train = os.path.join(data_dir_1, 'train', 'masks')
    json_dir_train = os.path.join(data_dir_1, "train", "mask_points.json")

    images_dir_train2 = os.path.join(data_dir, 'train', 'images')
    masks_dir_train2 = os.path.join(data_dir, 'train', 'masks')
    json_dir_train2 = os.path.join(data_dir, "train", "mask_points.json")

    images_dir_val = os.path.join(data_dir, "val", "images")
    masks_dir_val = os.path.join(data_dir, "val", "masks")
    json_dir_val = os.path.join(data_dir, "val", "mask_points.json")


    save_dir = "result_semi/semi_spine_2"
    save_dir = "result_semi/semi_ISIC_2"
    save_dir = "result_semi/semi_COVID_2"
    csv_path = os.path.join(save_dir, "metrics_2.csv")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Val Dice', 'Val IoU', 'Val Precision', 'Val Recall', 'Val F1'])

    stage1_path = "result_semi/semi_spine_1"
    stage1_path = "result_semi/semi_ISIC_1"
    stage1_path = "result_semi/semi_COVID_1"

    # 加载源域特征
    source_feature_bank = torch.load(os.path.join(stage1_path, "best_val_feature_bank.pt"))
    source_feature_bank = torch.stack(source_feature_bank).to(device)  # [N, C]

    # 加载提示特征并进行聚类
    source_prompt_feats = torch.load(os.path.join(stage1_path, "best_val_prompt_bank.pt"))
    fin = FINCH(verbose=True)
    pts = torch.stack(source_prompt_feats).cpu().numpy()
    res = fin.fit(pts)
    last_key = list(res.partitions.keys())[-1]
    target_pts = {'target_pts': res.partitions[last_key]['cluster_centers']}

    # ==== 构建模型 ====
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint_path, device="cuda"))
    predictor.model.load_state_dict(torch.load(os.path.join(stage1_path, "last.pt")))
    predictor.model.sam_mask_decoder.train(True)
    predictor.model.sam_prompt_encoder.train(True)
    predictor.model.image_encoder.train(True)


    # ==== 加载数据 ====
    train_data_list = load_image_mask_points(images_dir_train, masks_dir_train, json_dir_train)
    val_data_list = load_image_mask_points(images_dir_val, masks_dir_val, json_dir_val)

    # 预测高质量伪标签
    output_dir = os.path.join(data_dir, "train", "pseudo_labels")
    add_data_list = generate_pseudo_labels(predictor, images_dir_train2, json_dir_train2, output_dir, score_threshold = 0.9)

    train_data_list.extend(add_data_list)
    random.shuffle(train_data_list)  # 打乱顺序


    # 优化器与学习率调度
    optimizer = torch.optim.AdamW([
        {'params': predictor.model.sam_mask_decoder.parameters(), 'lr': 1e-5},
        {'params': predictor.model.sam_prompt_encoder.parameters(), 'lr': 1e-5},
        {'params': predictor.model.image_encoder.parameters(), 'lr': 1e-6},
    ], weight_decay=1e-3)
    scaler = torch.cuda.amp.GradScaler()

    accumulation_steps = 4
    num_epochs = 50
    steps_per_epoch = len(train_data_list)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * steps_per_epoch / accumulation_steps, eta_min=1e-6)

    patience = 10 * steps_per_epoch
    counter = 0
    global_step = 0
    best_val_iou, best_val_dice = 0.0, 0.0
    loss_list, iou_list, dice_list, val_iou_list, val_dice_list, val_precision_list, val_recall_list, val_f1_list,  steps_list = [], [], [], [], [], [], [], [], []
    mean_iou, mean_dice = 0.0, 0.0
    source_prompt_queue = deque(maxlen=512)  # 存储提示特征，最大长度336*11

    # 初始验证
    print(f"Direct val on the Stage 1")
    val_iou, val_dice, val_precision, val_recall, val_f1 = evaluate_validation_set(predictor, val_data_list)
    print(f"Direct val on the original SAM: "
          f"Val IoU = {val_iou:.4f}, "
          f"Val Dice = {val_dice:.4f}, "
          f"Val Precision = {val_precision:.4f}, "
          f"Val Recall = {val_recall:.4f}, "
          f"Val F1 = {val_f1:.4f}, ")


    # ==== 训练主循环 ====
    for epoch in range(num_epochs):
        for ent in tqdm(train_data_list, desc=f"Epoch {epoch + 1}/{num_epochs}", total=len(train_data_list)):
            global_step += 1
            with torch.cuda.amp.autocast():
                image, masks, input_points, num_instances = read_batch(ent)
                if image is None or masks is None or num_instances == 0:
                    continue
                if masks.shape[0] != num_instances:
                    print(f"掩码数量不匹配: {masks.shape[0]} != {num_instances}，跳过此批次")
                    continue
                input_labels = np.ones((num_instances, 1))
                if input_points.size == 0 or input_labels.size == 0:
                    continue
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
                prd_masks_logits = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
                prd_masks = torch.sigmoid(prd_masks_logits[:, 0])
                gt_masks = torch.tensor(masks.astype(np.float32)).cuda()


                # 提取目标特征并进行语义对齐
                tgt_feat = high_res_features[-1].squeeze(0)  # [C, H, W]
                tgt_feat = F.adaptive_avg_pool2d(tgt_feat, 1).flatten()  # [C]
                tgt_feat = F.normalize(tgt_feat, p=2, dim=0)  # L2归一化
                # 计算余弦相似度
                sims = F.cosine_similarity(tgt_feat.unsqueeze(0), source_feature_bank, dim=1)
                # # 特征蒸馏损失
                nearest_feat = source_feature_bank[sims.argmax()]  # 最近的源特征
                distill_loss = F.mse_loss(tgt_feat, nearest_feat)  # MSE损失

                # topk_indices = torch.topk(sims, k=3).indices
                # nearest_feats = source_feature_bank[topk_indices]  # [k, C]
                # mean_feat = nearest_feats.mean(dim=0)
                # distill_loss = F.mse_loss(tgt_feat, mean_feat)


                seg_loss, dice_loss, bound_loss, loss_match, iou, dice = 0, 0, 0, 0, 0, 0
                for i in range(num_instances):
                    prd_mask = prd_masks[i:i+1]
                    gt_mask = gt_masks[i:i+1]
                    seg_loss += (-gt_mask * torch.log(prd_mask + 1e-6) - (1 - gt_mask) * torch.log(1 - prd_mask + 1e-6)).mean()
                    intersection = (gt_mask * prd_mask).sum()
                    dice_loss += 1 - (2 * intersection + 1e-6) / (gt_mask.sum() + prd_mask.sum() + 1e-6)
                    bound_loss += boundary_loss(prd_mask, gt_mask)
                    inter = (gt_mask * (prd_mask > 0.5)).sum()
                    union = gt_mask.sum() + (prd_mask > 0.5).sum() - inter
                    iou += inter / (union + 1e-6)
                    dice_inter = (gt_mask * (prd_mask > 0.5)).sum()
                    dice += (2 * dice_inter + 1e-6) / (gt_mask.sum() + (prd_mask > 0.5).sum() + 1e-6)
                    feat = sparse_embeddings[i, 0, :].detach().cpu()
                    source_prompt_queue.append(feat)

                if len(source_prompt_queue) == source_prompt_queue.maxlen:
                    source_prompt_tensor = torch.stack(list(source_prompt_queue)).cpu().numpy()
                    target_pts_ = target_pts['target_pts']
                    fin = FINCH(verbose=False)
                    results = fin.fit(source_prompt_tensor)
                    last_key = list(results.partitions.keys())[-1]
                    pred_pts = results.partitions[last_key]['cluster_centers']
                    loss_match += Matching_Loss(pred_pts, target_pts_, device)

                seg_loss = seg_loss / max(1, num_instances)
                dice_loss = dice_loss / max(1, num_instances)
                iou = iou / max(1, num_instances)
                dice = dice / max(1, num_instances)
                score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
                bound_loss /= max(1, num_instances)
                loss = seg_loss + dice_loss + 0.05 * score_loss + 0.1 * bound_loss + 0.1 * loss_match + 0.1 * distill_loss  # 总损失

            loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), 1.0)

            if global_step % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                predictor.model.zero_grad()
                scheduler.step()

            mean_iou = mean_iou * 0.99 + 0.01 * iou.item()
            mean_dice = mean_dice * 0.99 + 0.01 * dice.item()

            if global_step % steps_per_epoch == 0:
                steps_list.append(global_step)
                loss_list.append(loss.item() * accumulation_steps)
                iou_list.append(mean_iou)
                dice_list.append(mean_dice)
                val_iou, val_dice, val_precision, val_recall, val_f1 = evaluate_validation_set(predictor, val_data_list)
                val_iou_list.append(val_iou)
                val_dice_list.append(val_dice)
                val_precision_list.append(val_precision)
                val_recall_list.append(val_recall)
                val_f1_list.append(val_f1)
                print(f"Step {global_step}: Train IoU = {mean_iou:.4f}, "
                      f"Train Dice = {mean_dice:.4f}, "
                      f"Val IoU = {val_iou:.4f}, "
                      f"Val Dice = {val_dice:.4f}, "
                      f"Val Precision = {val_precision:.4f}, "
                      f"Val Recall = {val_recall:.4f}, "
                      f"Val F1 = {val_f1:.4f}")

                # ✅ 将每个 epoch 的指标写入 CSV 文件
                with open(csv_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch + 1,
                        round(val_dice, 4),
                        round(val_iou, 4),
                        round(val_precision, 4),
                        round(val_recall, 4),
                        round(val_f1, 4),
                    ])

                if val_dice > best_val_dice:
                    best_val_dice = val_dice
                    counter = 0
                    torch.save(predictor.model.state_dict(), os.path.join(save_dir, "best_val.pt"))
                    print(f"New best model saved at Step {global_step} with Val IoU={val_iou:.4f}, Val Dice={val_dice:.4f}")
                else:
                    counter += steps_per_epoch
                    if counter >= patience:
                        print(f"Early stopping at step {global_step}")
                        break

        if counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    torch.save(predictor.model.state_dict(), os.path.join(save_dir, "last.pt"))
    vis_evaluate(save_dir, steps_list, loss_list, iou_list, dice_list, val_iou_list, val_dice_list, val_precision_list, val_recall_list, val_f1_list)


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

if __name__ == '__main__':
    start_time = time.time()
    train_stage2()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total time {total_time_str}")