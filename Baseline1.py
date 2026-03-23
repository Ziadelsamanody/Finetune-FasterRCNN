import torch 
from torch.utils.data import DataLoader
from load_data import VocDetection
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
model = RPN(in_channels=512, image_size=(224, 224))
optimizer = optim.SGD(model.parameters(), lr=0.001)


def train_rpn(model, dataloader, optimizer, device):
    """Train the RPN for one epoch and save the weights.

    This function assumes a detection-style dataloader where each batch is
    (images, targets) and, for now, effectively operates on batch_size = 1.
    """

    model.to(device)
    model.train()

    backbone = load_backbone().to(device)
    backbone.eval()

    total_loss = 0.0
    num_batches = 0

    for batch_idx, (images, targets) in enumerate(dataloader):
        # Handle images coming either as a list/tuple of tensors or a batched tensor
        if isinstance(images, (list, tuple)):
            if len(images) != 1:
                raise ValueError(
                    "train_rpn currently expects batch_size = 1 when images is a list/tuple."
                )
            image = images[0]
        else:
            # images is a tensor, possibly with batch dim
            if images.dim() == 4:  # [B, C, H, W]
                if images.size(0) != 1:
                    raise ValueError(
                        "train_rpn currently expects batch_size = 1 when images is a tensor."
                    )
                image = images[0]
            else:
                image = images

        if not isinstance(image, torch.Tensor):
            image = ToTensor()(image)

        if image.dim() == 3:  # [C, H, W]
            image = image.unsqueeze(0)  # [1, C, H, W]

        image = image.to(device)

        # Handle targets coming either as a list/tuple of dicts or a single dict
        if isinstance(targets, (list, tuple)):
            if len(targets) != 1:
                raise ValueError(
                    "train_rpn currently expects batch_size = 1 when targets is a list/tuple."
                )
            target = targets[0]
        else:
            target = targets

        gt_boxes = target["boxes"].to(device)

        with torch.no_grad():
            features = backbone(image)

        # RPN expects a feature map and images; it will build
        # an ImageList internally and generate anchors for all locations.
        cls_logits, bbox_deltas, rois, anchors = model(features, image)

        # For batch size 1, squeeze the batch dimension from RPN outputs
        cls_logits = cls_logits[0]  # [N, 2]
        bbox_deltas = bbox_deltas[0]  # [N, 4]
        anchors = anchors[0]         # [N, 4]

        # Compute labels and regression targets per-anchor so shapes match
        labels, max_gt_idx = assign_anchors_to_gt(anchors, gt_boxes)
        labels = subsample_labels(labels)
        target_deltas = encode_propsals(anchors, gt_boxes[max_gt_idx])

        # Classification loss: foreground vs background on valid anchors
        fg_logits = cls_logits[:, 1]  # [N]
        valid_mask = labels >= 0  # ignore = -1
        if valid_mask.any():
            cls_loss = f.binary_cross_entropy_with_logits(
                fg_logits[valid_mask], labels[valid_mask].float()
            )
        else:
            cls_loss = torch.tensor(0.0, device=device)

        # Regression loss: only on positive anchors
        pos_mask = labels == 1
        if pos_mask.any():
            reg_loss = f.smooth_l1_loss(
                bbox_deltas[pos_mask], target_deltas[pos_mask]
            )
        else:
            reg_loss = torch.tensor(0.0, device=device)

        loss = cls_loss + reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        print(
            f"Batch {batch_idx + 1} - "
            f"cls_loss: {cls_loss.item():.4f}, "
            f"reg_loss: {reg_loss.item():.4f}, "
            f"total_loss: {loss.item():.4f}"
        )

    if num_batches > 0:
        avg_loss = total_loss / num_batches
        print(f"Average RPN loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "rpn_model.pth")
    print("Saved RPN model to rpn_model.pth")

    
if __name__ == "__main__":
    dataset = VocDetection()
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=VocDetection.collate_fn,
    )
    train_rpn(model, dataloader, optimizer, device)