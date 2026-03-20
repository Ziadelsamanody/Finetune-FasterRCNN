import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.ops import roi_pool, nms, Conv2dNormActivation
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.rpn import AnchorGenerator


class RPN(nn.Module):
    def __init__(self, in_channels, image_size, mid_channels=256):
        super(RPN, self).__init__()
        self.anchor_generator = AnchorGenerator(
            sizes= ((32, 64, 128),),
            aspect_ratios= ((0.5, 1.0, 2.0))
            )
        
        self.num_anchors = self.anchor_generator.num_anchors_per_location()[0]
        self.image_size = image_size 

        self.conv = Conv2dNormActivation(in_channels, mid_channels, kernel_size=3, norm_layer=None)
        self.cls_logits = nn.Conv2d(
            mid_channels, self.num_anchors *2 , kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(mid_channels, self.num_anchors *4 , kernel_size=1, stride=1)

    def forward(self, feature_map, image_list, num_rois=200):
        bsz, _, hh, ww = feature_map.shape 
        x = F.relu(self.conv(feature_map))
        rpn_logits = self.cls_logits(x).permute(0, 2, 3, 1).contiguous().view(bsz, -1, 2)
        rpn_box_delta = self.bbox_pred(x).permute(0,2,3,1).contiguous().view(bsz, -1, 4)

        anchor_list = self.anchor_generator(image_list, [feature_map])
        all_rois = []
        # convert proposal and select the best one 
        for i in range(bsz):
            anchors = anchor_list[i]
            proposal = self._generate_proposals(rpn_box_delta[i], anchors)
            proposal =  self._select_rois(proposal, rpn_logits[i], num_rois)
            proposal[:, 0] = i 
            all_rois.append(proposal)

        rois =  torch.cat(proposal, dim=1)
        # Stop Gradient from ROIS to logits and deltas 
        rois = rois.detach()
        return rpn_logits, rpn_box_delta,  rois
    def _generate_proposals(self, rpn_box_delta, anchors) : 
        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        ctr_x =  anchors[:, 0] + 0.5 * widths
        ctr_y = anchors[:, 1] + 0.5 * heights
        # decode predictions 
        dx, dy = rpn_box_delta[:, 0], rpn_box_delta[:, 1]
        dw, dh = rpn_box_delta[:, 2], rpn_box_delta[:, 3]

        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights
        # converts format (center, width, hegiht) to (x1, y1, x2, y2)
        pred_boxes = torch.zeros_like(rpn_box_delta)
        pred_boxes[:, 0] = pred_ctr_x - 0.5 * (pred_w - 1.0)
        pred_boxes[:, 1] = pred_ctr_y - 0.5 * (pred_h-1.0)
        pred_boxes[:, 2] = pred_ctr_x + 0.5 * (pred_w - 1.0)
        pred_boxes[:, 3] = pred_ctr_y + 0.5 * (pred_h - 1.0)
        # clip boxes to ensure boxes don't go outside the image
        pred_boxes[:, 0::2] = pred_boxes[:, 0::2].clamp(min=0, max=self.image_size[1] - 1)
        pred_boxes[:, 1::2] = pred_boxes[:, 1::2].clamp(min=0, max= self.image_size[0] - 1)

        batch_indices = torch.zeros((pred_boxes.size(0), 1), device=pred_boxes.device)
        return torch.cat([batch_indices, pred_boxes], dim=1)
    
    def _select_rois(self, proposals,logits, num_rois = 200, ious_threshold=0.7):
        scores = torch.softmax(logits, dim=1)[:, 1] # foreground scores 
        #remove batch size 
        proposals = proposals[:, 1:5] 
        keep = nms(proposals, scores, ious_threshold)
        keep = keep[: num_rois]
        return proposals[keep]