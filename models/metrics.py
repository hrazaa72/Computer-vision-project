"""Mask IoU, COCO-style mask AP, precision and recall for a one-class, one-object-per-image task."""
import numpy as np
import torch

IOU_THRESHOLDS = np.arange(0.5, 0.96, 0.05)


def mask_iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def average_precision(scores, ious, threshold, n_images):
    """All-point interpolated AP. scores[i] and ious[i] = (image index, IoU) describe prediction i.
    Each image has one ground-truth object, so a prediction is a true positive only if it is the
    first of its image to reach the threshold; later ones on the same image are false positives."""
    order = np.argsort(-np.asarray(scores))
    seen, tp = set(), []
    for i in order:
        image, iou = ious[i]
        hit = iou >= threshold and image not in seen
        if hit:
            seen.add(image)
        tp.append(hit)
    tp = np.array(tp, float)
    cum_tp, cum_fp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = cum_tp / n_images
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
    recall = np.concatenate([[0], recall, [recall[-1]]])
    precision = np.concatenate([[1], precision, [0]])
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    steps = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[steps + 1] - recall[steps]) * precision[steps + 1]))


@torch.no_grad()
def evaluate(model, dataset, device, score_threshold=0.5):
    model.eval()
    scores, ious, top_ious = [], [], []
    for index in range(len(dataset)):
        image, target = dataset[index]
        output = model([image.to(device)])[0]
        truth = target["masks"][0].numpy().astype(bool)
        predicted = (output["masks"][:, 0] > 0.5).cpu().numpy()
        for score, mask in zip(output["scores"].cpu().numpy(), predicted):
            scores.append(float(score))
            ious.append((index, mask_iou(mask, truth)))
        top_ious.append(mask_iou(predicted[0], truth) if len(predicted) else 0.0)

    n = len(dataset)
    result = {"images": n, "mask_iou_mean": float(np.mean(top_ious)),
              "mask_iou_median": float(np.median(top_ious)), "mask_iou_min": float(np.min(top_ious))}
    if scores:
        aps = [average_precision(scores, ious, t, n) for t in IOU_THRESHOLDS]
        result["mAP@0.5"], result["mAP@0.5:0.95"] = aps[0], float(np.mean(aps))
        kept = [(s, i) for s, i in zip(scores, ious) if s >= score_threshold]
        true_pos = len({i[0] for _, i in kept if i[1] >= 0.5})
        result["precision@0.5"] = true_pos / len(kept) if kept else 0.0
        result["recall@0.5"] = true_pos / n
    else:
        result.update({"mAP@0.5": 0.0, "mAP@0.5:0.95": 0.0, "precision@0.5": 0.0, "recall@0.5": 0.0})
    result["score_threshold"] = score_threshold
    return result
