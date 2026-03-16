import torch 
from torch.utils.data import DataLoader
from load_data import VocDetection
from model.network import FasterRCNN
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.optim as optim

class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.reg_loss_fn = nn.SmoothL1Loss()
    def forward(self, cls_score, bbox_pred, targets):
        # Implement the loss calculation for both classification and regression
        cls_loss = self.cls_loss_fn(cls_score, targets['labels'])
        reg_loss = self.reg_loss_fn(bbox_pred, targets['boxes'])
        return cls_loss, reg_loss
    
dataloader = VocDetection()
model = FasterRCNN(num_classes=21)
optimizer= optim.SGD(model.parameters(), lr=0.001)

num_epochs = 1 

for p in model.parameters:
    p.requires_grad = False 
    print(p.numel())
# for epochs in range(num_epochs):


