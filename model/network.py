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
        rpn_logits, rpn_box_detas , rois = self.rpn(backbone, image_list)
        pooled_rois = roi_pool(backbone, rois, self.roi_pool_output_size, self.spatial_scale)
        cls_score, bbox_preds = self.detection_head(pooled_rois)

        return cls_score, bbox_preds