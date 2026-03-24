import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.image_list import ImageList
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms.functional import to_tensor
from PIL import Image
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from model.network import FasterRCNN
from model.rpn import RPN

def load_backbone():
    """Load ResNet50 backbone with frozen weights."""
    resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    backbone = nn.Sequential(
        *list(resnet.children())[:-4]
    )
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone
def inference_rpn(model_path, image, device=device, score_threshold=0.5, 
                  nms_iou_threshold=0.7, transform=None):
    backbone = load_backbone().to(device).eval()
    rpn = RPN(in_channels=512, image_size=(224, 224))
    rpn.load_state_dict(torch.load(model_path, map_location=device))
    rpn = rpn.to(device)
    rpn.eval()
    if isinstance(image, (Image.Image, np.ndarray)):
        image = transform(image)
    if image.dim() == 3 :
        image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        features = backbone(image)
        rpn_logits, rpn_deltas, rois, all_anchors = rpn(features, image)
        
        # rois are already filtered by RPN's NMS and top-k selection
        # We can only compute meaningful scores from the selected proposals
        # by using the RPN logits at inference time for visualization
        proposals = rois.cpu()  # [M, 4]
        
        # All proposals passed RPN filtering, so assign confidence = 1.0
        # (they're the top confidence anchors after NMS)
        scores = torch.ones(len(proposals))
        
    return proposals, scores

def inference_faster_rcnn(model_path, rpn_path,image, device=device, score_threshold=0.5, 
                  nms_iou_threshold=0.7, transform=None):
    model = FasterRCNN(num_classes=21).to(device)
    rpn_weight = torch.load(rpn_path, map_location=device)
    model.rpn.load_state_dict(rpn_weight)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    if isinstance(image, (Image.Image, np.ndarray)):
        image = transform(image)
    if image.dim() == 3 :
        image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        cls_scores, bbox_preds, proposals = model(image)
        
        # Apply softmax to get class probabilities
        probs = F.softmax(cls_scores, dim=1)
        scores, labels = torch.max(probs, dim=1)

        # Filter by score threshold
        keep = scores > score_threshold
        proposals = proposals[keep]
        scores = scores[keep]
        labels = labels[keep]

        # Apply NMS
        keep_indices = torchvision.ops.nms(proposals, scores, nms_iou_threshold)
        final_boxes = proposals[keep_indices]
        final_scores = scores[keep_indices]
        final_labels = labels[keep_indices]

    return final_boxes.cpu(), final_scores.cpu(), final_labels.cpu()

if __name__ == "__main__":
    from load_data import default_transform
    image_path = "test_images/00a42fc158bbdc03.jpg"
    rpn_path = "rpn_trained_stage1.pth"
    image = Image.open(image_path).convert("RGB")
    final_boxes, final_scores, final_labels = inference_faster_rcnn("faster_rcnn_trained.pth", rpn_path,image, device=device, transform=default_transform)
    print("Faster R-CNN Final Boxes:", final_boxes)
    print("Faster R-CNN Final Scores:", final_scores)
    print("Faster R-CNN Final Labels:", final_labels)