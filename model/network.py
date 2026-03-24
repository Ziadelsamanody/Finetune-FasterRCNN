from .rpn import RPN
import torch 
import torch.nn as nn 
import torch.nn.functional as f 
from torchvision.ops import roi_pool, nms, Conv2dNormActivation
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection.image_list import ImageList
class DetectionHead(nn.Module): 
    def __init__(self, in_channels, num_classes):
        super(DetectionHead, self).__init__()
        self.fc1 = nn.Linear(in_channels * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)

    def forward(self, x): 
        x = x.view(x.size(0), -1)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        return self.cls_score(x), self.bbox_pred(x)
    
class FasterRCNN(nn.Module):
    def __init__(self,num_classes, image_size= (224,224)):
        super(FasterRCNN, self).__init__()
        resnet = resnet50(weights = ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[: -4])
        self.rpn = RPN(in_channels=512, image_size=image_size)
        self.roi_pool_output_size = (7,7)
        self.detection_head = DetectionHead(in_channels=512, num_classes=num_classes)
        self.image_size = image_size
        self.spatial_scale = None
    def forward(self, x): 
        backbone = self.backbone(x)
        self.spatial_scale = backbone.size(2) / self.image_size[0]

        N,C,H,W = x.shape 
        image_sizes = [(H,W)] * N 
        image_list = ImageList(x, image_sizes)
        rpn_logits, rpn_box_deltas, rois, _ = self.rpn(backbone, image_list)
        
        # Add batch indices to rois: [M, 4] -> [M, 5]
        batch_indices = torch.zeros((rois.size(0), 1), dtype=rois.dtype, device=rois.device)
        rois_with_batch = torch.cat([batch_indices, rois], dim=1)
        
        pooled_rois = roi_pool(backbone, rois_with_batch, self.roi_pool_output_size, self.spatial_scale)
        cls_score, bbox_preds = self.detection_head(pooled_rois)

        return cls_score, bbox_preds, rois
    

import torch
import torch.nn as nn

class FasterRCNNLoss(nn.Module):
    def __init__(self, pos_iou_threshold=0.5, neg_iou_threshold=0.3):
        super(FasterRCNNLoss, self).__init__()
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.bbox_reg = nn.SmoothL1Loss(reduction='mean')
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_threshold = neg_iou_threshold

    def forward(self, cls_score, bbox_pred, proposals, gt_labels, gt_boxes):
        """
        Args:
            cls_score: [num_proposals, num_classes] - classification scores
            bbox_pred: [num_proposals, num_classes * 4] - bbox predictions
            proposals: [num_proposals, 4] - RPN proposals
            gt_labels: [num_gt_boxes] - GT class labels
            gt_boxes: [num_gt_boxes, 4] - GT boxes
        """
        from torchvision.ops import box_iou
        
        num_proposals = cls_score.size(0)
        num_gt = gt_boxes.size(0)
        num_classes = cls_score.size(1)
        
        # Create proposal labels (0 = background)
        proposal_labels = torch.zeros(num_proposals, dtype=torch.long, device=cls_score.device)
        
        if num_gt > 0:
            # Compute IoU between proposals and GT boxes
            ious = box_iou(proposals, gt_boxes)  # [num_proposals, num_gt]
            max_ious, max_gt_idx = ious.max(dim=1)  # [num_proposals]
            
            # Assign labels: positive if IoU > threshold, negative if < threshold
            pos_mask = max_ious > self.pos_iou_threshold
            proposal_labels[pos_mask] = gt_labels[max_gt_idx[pos_mask]]
            
            # Negative labels stay as 0 (background)
            # Proposals with IoU between thresholds are ignored (-1, but we use 0 for simplicity)
        
        # Classification loss
        cls_loss = self.cls_loss_fn(cls_score, proposal_labels)
        
        # Regression loss (only on positive proposals)
        num_pos = (proposal_labels > 0).sum().item()
        if num_pos > 0:
            pos_mask = proposal_labels > 0
            pos_bbox_pred = bbox_pred[pos_mask]
            # For positive proposals, create dummy regression targets
            # In full implementation, would encode GT boxes relative to proposals
            target_size = pos_bbox_pred.size(0) * pos_bbox_pred.size(1)
            pos_targets = torch.zeros_like(pos_bbox_pred)
            reg_loss = self.bbox_reg(pos_bbox_pred, pos_targets)
        else:
            reg_loss = torch.tensor(0.0, device=cls_score.device, requires_grad=True)
        
        total_loss = cls_loss + reg_loss
        return cls_loss, reg_loss, total_loss