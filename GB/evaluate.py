import os
import numpy as np
from sklearn.metrics import accuracy_score
import cv2


def compute_segmentation_metrics(true_mask, pred_mask, num_classes=3):
    """
    计算多类别语义分割评估指标：Accuracy、Precision、Recall、F1、Dice、IoU（忽略背景类0）

    参数：
        true_mask: H x W numpy array，真实标签
        pred_mask: H x W numpy array，预测标签
        num_classes: 类别数（像素值范围 0~num_classes-1）

    返回：
        字典形式的各项指标（每类+均值，不包括背景类）
    """
    metrics = {
        'accuracy': 0.0,
        'precision': [],
        'recall': [],
        'f1': [],
        'dice': [],
        'iou': []
    }

    # 展平
    true_flat = true_mask.flatten()
    pred_flat = pred_mask.flatten()

    # 总体准确率（只考虑非背景类）
    non_bg_mask = true_flat != 0
    if np.sum(non_bg_mask) > 0:
        metrics['accuracy'] = accuracy_score(true_flat[non_bg_mask], pred_flat[non_bg_mask])
    else:
        metrics['accuracy'] = 0.0

    # 从类1开始，忽略背景类0
    for cls in range(1, num_classes):
        true_cls = (true_flat == cls).astype(np.uint8)
        pred_cls = (pred_flat == cls).astype(np.uint8)

        TP = np.sum((true_cls == 1) & (pred_cls == 1))
        FP = np.sum((true_cls == 0) & (pred_cls == 1))
        FN = np.sum((true_cls == 1) & (pred_cls == 0))

        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        dice = 2 * TP / (2 * TP + FP + FN + 1e-8)
        iou = TP / (TP + FP + FN + 1e-8)

        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)
        metrics['dice'].append(dice)
        metrics['iou'].append(iou)

    # 计算均值
    metrics['mean_precision'] = np.mean(metrics['precision'])
    metrics['mean_recall'] = np.mean(metrics['recall'])
    metrics['mean_f1'] = np.mean(metrics['f1'])
    metrics['mean_dice'] = np.mean(metrics['dice'])
    metrics['mean_iou'] = np.mean(metrics['iou'])

    return metrics


def evaluate_all(true_dir, pred_dir, num_classes=3):
    results = []

    file_list = sorted(os.listdir(true_dir))
    for file in file_list:
        true_path = os.path.join(true_dir, file)
        pred_path = os.path.join(pred_dir, file)
        if not os.path.exists(pred_path):
            print(f"❌ 预测文件不存在: {file}")
            continue

        true_mask = cv2.imread(true_path, cv2.IMREAD_GRAYSCALE)
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        if true_mask.shape != pred_mask.shape:
            print(f"⚠️ 尺寸不匹配: {file}")
            continue

        metrics = compute_segmentation_metrics(true_mask, pred_mask, num_classes)
        result = {
            'file': file,
            'accuracy': metrics['accuracy'],
            'mean_precision': metrics['mean_precision'],
            'mean_recall': metrics['mean_recall'],
            'mean_f1': metrics['mean_f1'],
            'mean_dice': metrics['mean_dice'],
            'mean_iou': metrics['mean_iou'],
        }

        # 记录非背景类的指标
        for i, cls in enumerate(range(1, num_classes)):
            result[f'precision_class_{cls}'] = metrics['precision'][i]
            result[f'recall_class_{cls}'] = metrics['recall'][i]
            result[f'f1_class_{cls}'] = metrics['f1'][i]
            result[f'dice_class_{cls}'] = metrics['dice'][i]
            result[f'iou_class_{cls}'] = metrics['iou'][i]

        results.append(result)

    # 打印平均指标
    print("\n📊 平均指标（Across all images, ignoring background class）：")
    avg_metrics = {
        'accuracy': np.mean([r['accuracy'] for r in results]),
        'mean_precision': np.mean([r['mean_precision'] for r in results]),
        'mean_recall': np.mean([r['mean_recall'] for r in results]),
        'mean_f1': np.mean([r['mean_f1'] for r in results]),
        'mean_dice': np.mean([r['mean_dice'] for r in results]),
        'mean_iou': np.mean([r['mean_iou'] for r in results]),
    }
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == '__main__':
    true_dir = r"data_covid/val/masks"
    pred_dir = r"data_covid/val/mask_pred"
    evaluate_all(true_dir, pred_dir, num_classes=3)