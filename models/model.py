"""Mask R-CNN builder.

Mask R-CNN (ResNet-50 FPN, COCO-pretrained) is used because it gives one mask, one box and one
confidence score per object. The score feeds the refusal rule, and the box + mask pair allows
standard COCO mask AP. The two heads are replaced for one foreground class and fine-tuned.
"""
import torch
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

NUM_CLASSES = 2  # background + med_box
MIN_SIZE, MAX_SIZE = 640, 1152  # images are 720x1280 portrait


def build_model(pretrained=True):
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn(
        weights=weights, weights_backbone=None, min_size=MIN_SIZE, max_size=MAX_SIZE,
        box_detections_per_img=5)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    in_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels, 256, NUM_CLASSES)
    return model


def load_checkpoint(path, device):
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()
