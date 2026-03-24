import torch 
from torch.utils.data import DataLoader
from load_data import VocDetection
from model.network import FasterRCNN, FasterRCNNLoss
from model.rpn import RPN
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from utils.utils import (
    assign_anchors_to_gt,
    subsample_labels,
    encode_propsals,
)
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection.image_list import ImageList
from torchvision.utils import draw_bounding_boxes
from PIL import Image
import numpy as np 
dataset = VocDetection()
from torch_snippets import show
from torchvision.transforms import transforms
# First baseline Train in two steps: 
#  train RPN  
# Freeze RPN , Train Detection Head 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

default_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


dataloader = DataLoader(
        dataset, 
        batch_size= 4,
        collate_fn=dataset.collate_fn,
        pin_memory=True 
    )
# def load_backbone():
#     resnet = resnet50(weights= ResNet50_Weights.DEFAULT)
#     backbone = nn.Sequential(
#         *list(resnet.children())[:-4]
#     )
#     for p in backbone.parameters():
#         p.requires_grad = False
#     return backbone

def compute_rpn_loss(rpn_logits, rpn_deltas, anchor_list, targets, device=device):
    cls_loss = 0.0
    reg_loss = 0.0
    total_pos = 0 
    num_images = len(anchor_list)

    for i in range(num_images):
        anchors = anchor_list[i]
        gt_boxes = targets[i]['boxes'].to(device)

        labels, matched_gt_idx = assign_anchors_to_gt(anchors, gt_boxes)
        # Subsamble labels
        labels = subsample_labels(labels)
        pos_mask = (labels ==1)
        valid_mask = (labels >= 0)
        if valid_mask.any():
            cls_loss_i = f.cross_entropy(rpn_logits[i][valid_mask],
                                       labels[valid_mask].long(),
                                       reduction='mean')
        else : 
            cls_loss_i = torch.tensor(0.0, device=device) 
        if pos_mask.any(): 
            pos_anchors = anchors[pos_mask]
            pos_deltas = rpn_deltas[i][pos_mask]
            pos_gt = gt_boxes[matched_gt_idx[pos_mask]]
            target_deltas = encode_propsals(pos_anchors, pos_gt)

            reg_loss_i = f.smooth_l1_loss(pos_deltas, target_deltas, beta=0.111,
                                          reduction='mean')
            num_pos = pos_mask.sum().item()
        else : 
            reg_loss_i = torch.tensor(0.0, device=device)
            num_pos = 0
        cls_loss += cls_loss_i
        reg_loss += reg_loss_i
        total_pos += num_pos 
    cls_loss /= max(num_images, 1)
    if total_pos > 0 :
        reg_loss /= total_pos
    return {
        'loss_rpn_cls' : cls_loss,
        'loss_rpn_delta': reg_loss,
        'total_loss' : cls_loss + reg_loss
    }

def train_rpn(epochs= 10):
    backbone = load_backbone()
    backbone = backbone.to(device)
    backbone.eval()
    rpn = RPN(in_channels=512, image_size=(224, 224))
    rpn = rpn.to(device)
    rpn.train()
    optimizer = optim.SGD(rpn.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)


    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx , (images, target) in enumerate(dataloader):
            # Stack images if they come as a tuple from collate_fn
            if isinstance(images, (list, tuple)):
                images = torch.stack(images)
            images = images.to(device)

            with torch.no_grad():
                features = backbone(images)
            
            # forward pass
            rpn_logits, rpn_box_deltas, rois, anchors_list = rpn(features, images)

            loss_dict = compute_rpn_loss(
                rpn_logits=rpn_logits,
                rpn_deltas=rpn_box_deltas,
                anchor_list=anchors_list,
                targets=target,
                device=device
            )

            loss = loss_dict['total_loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            print(f'Epoch : [{epoch +1}/ {epochs}] | '
                  f'batch : [{batch_idx}] | '
                  f'cls: [{loss_dict['loss_rpn_cls'].item() : .4f}] |'
                  f'box : [{loss_dict['loss_rpn_delta'].item() : .4f}] |'
                  f'total : [{loss.item()}]')
            torch.save(rpn.state_dict(), "rpn_trained_stage1.pth")
        
        avg_loss = total_loss / (batch_idx + 1)
        print(f"Epoch {epoch+1} finished | Avg loss: {avg_loss:.4f}\n")





def trainFasterRCNN(criterion, rpn_path, epochs=10, dataloader=dataloader, device=device):
    model = FasterRCNN(num_classes=21).to(device)
    rpn_weights = torch.load(rpn_path, map_location=device)
    model.rpn.load_state_dict(rpn_weights)
  
    for param in model.backbone.parameters():
        param.requires_grad = False 
    for p in model.rpn.parameters():
        p.requires_grad = False
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(
            trainable_params, 
            lr=0.005,           # Learning rate (Adjust based on your batch size)
            momentum=0.9,       # Standard momentum
            weight_decay=0.0005 # L2 regularization to prevent overfitting
        )
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_idx, (image, targets) in enumerate(dataloader):
            # Stack images if they come as a tuple from collate_fn
            if isinstance(image, (list, tuple)):
                image = torch.stack(image)
            image = image.to(device)
            
            # Handle targets: collate_fn returns tuple of dicts, stack labels and boxes
            if isinstance(targets, (list, tuple)):
                labels = torch.cat([t['labels'] for t in targets]).to(device)
                boxes = torch.cat([t['boxes'] for t in targets]).to(device)
            else:
                labels = targets['labels'].to(device)
                boxes = targets['boxes'].to(device)

            cls_score, bbox_pred, proposals = model(image)
            

            cls_loss, reg_loss, total_losses = criterion(cls_score, bbox_pred, proposals, labels, boxes)
            total_loss += total_losses.item()
        
            optimizer.zero_grad()
            total_losses.backward()
            optimizer.step()
            
            print(f'Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}] '
                  f'cls_loss: {cls_loss.item():.4f} reg_loss: {reg_loss.item():.4f} '
                  f'total_loss: {total_losses.item():.4f}')
        
        avg_loss = total_loss / (batch_idx + 1)
        print(f"Epoch {epoch+1} finished | Avg loss: {avg_loss:.4f}\n")
        torch.save(model.state_dict(), "faster_rcnn_trained.pth")
if __name__ == '__main__':
    rpn_path = 'rpn_trained_stage1.pth'
    criterion = FasterRCNNLoss()
    trainFasterRCNN(criterion, rpn_path)