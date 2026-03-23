import torchvision
import torch
import torchvision.transforms as transforms 
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader, Dataset
from dotenv import load_dotenv 
import os 

load_dotenv()

data_path = os.environ.get('data_path')

# Default transform: convert PIL image to tensor
default_transform = transforms.Compose([
    transforms.ToTensor()
])
class VocDetection(Dataset):
    def __init__(self, root=data_path, image_set='train', transform=default_transform):
        super().__init__()
        self.data = VOCDetection(root=root, year='2007', image_set=image_set,  download=False )
        self.transform = transform 
        VOC_CLASSES = [
    'aeroplane','bicycle','bird','boat','bottle',
    'bus','car','cat','chair','cow',
    'diningtable','dog','horse','motorbike','person',
    'pottedplant','sheep','sofa','train','tvmonitor'
]
        self.class_to_idx = {clss:  idx + 1 for idx, clss in  enumerate(VOC_CLASSES)}
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        image, target = self.data[idx]
        if self.transform is not None:
            image = self.transform(image)
        
        objs = target['annotation']['object']
        if not isinstance(objs, list):
            objs = [objs]
        boxes = []
        labels = []
        for obj in objs : 
            bbox = obj['bndbox']
            xmin = float(bbox['xmin'])
            ymin = float(bbox['ymin'])
            xmax = float(bbox['xmax'])
            ymax = float(bbox['ymax'])
            boxes.append([xmin,ymin,xmax, ymax])
            
            name = obj['name']
            label = self.class_to_idx[name]
            labels.append(label)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels= torch.tensor(labels, dtype=torch.int64)

        new_target = {
            'boxes' : boxes,
            'labels' : labels,
            'image_id': torch.tensor([idx])
        
        }

        return image, new_target
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch)) # ([imgs], [targts])
    

