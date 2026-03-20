import torch 
from torch.utils.data import DataLoader
from load_data import VocDetection
from model.network import FasterRCNN
from model.rpn import RPN
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
# First baseline Train in two steps: 
#  train RPN  
# Freeze RPN , Train Detection Head 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def load_backbone():
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    backbone = nn.Sequential(*list(model.children())[:-4])
    # freeze model 
    for param in backbone.parameters():
        param.requires_grad = False
    return backbone
model = RPN(in_channels= 512, image_size=(224, 224)) 
optimizer = optim.SGD(model.parameters(), lr=0.001)

def train_rpn(model, dataloader, optimizer, device):
    model.train()

