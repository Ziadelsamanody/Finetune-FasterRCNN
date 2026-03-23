import matplotlib.pyplot as plt 
import matplotlib.patches as patches 
from PIL import Image
import numpy as np 
import torch 
import math 
import  torchvision.transforms as transform 
from torchvision.ops import box_iou , box_convert#  vectorized version

def parse_voc_target(target, device=None):
    """Extract GT boxes + class names from a torchvision VOCDetection target.

    Returns:
        gt_boxes: FloatTensor of shape (M, 4) in (xmin, ymin, xmax, ymax)
        gt_labels: list[str] length M (VOC class names)
    """
    if device is None:
        device = "cpu"

    ann = target["annotation"] if isinstance(target, dict) else target
    objs = ann.get("object", [])
    if isinstance(objs, dict):
        objs = [objs]
    if not objs:
        return torch.zeros((0, 4), dtype=torch.float32, device=device), []

    boxes = []
    labels = []
    for obj in objs:
        bbox = obj["bndbox"]
        xmin = float(bbox["xmin"])
        ymin = float(bbox["ymin"])
        xmax = float(bbox["xmax"])
        ymax = float(bbox["ymax"])
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(obj["name"])

    return torch.tensor(boxes, dtype=torch.float32, device=device), labels
def visulize_image_with_gt(dataset, idx):
    image, target = dataset[idx]

    # Convert image to numpy array 
    # chw -- > hwc
    img_np = np.array(image.permute(1,2,0))
    
    fig, ax = plt.subplots(1, figsize=(12,9))
    ax.imshow(img_np)

    # draw gt boxes  target['annotations']
    if isinstance(target, dict): 
        annos = target['annotation']['object']
        # Handle single object case (VOC returns dict instead of list)
        if isinstance(annos, dict):
            annos = [annos]
    else : 
        annos = target['annotation']['object']
    
    for anno in annos : 
        bbox = anno['bndbox']
        x, y, w, h = int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']) - int(bbox['xmin']), int(bbox['ymax']) - int(bbox['ymin'])
        label = anno['name']
        # draw rectangle 
        rect = patches.Rectangle((x,y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x, y-10, label, fontsize=10, bbox=dict(facecolor='white', alpha= 0.5))
    plt.axis('off')
    plt.show()

def iou(boxA, boxB):
    # Get intersection coordinates (inner bounds)
    xmin = max(boxA[0], boxB[0])
    ymin = max(boxA[1], boxB[1])
    xmax = min(boxA[2], boxB[2])
    ymax = min(boxA[3], boxB[3])

    # Calculate intersection area
    width = max(0, xmax - xmin)
    height = max(0, ymax - ymin)
    intersection = width * height
    
    # Calculate individual box areas
    area_a = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    area_b = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    # Calculate union and IoU
    union = area_a + area_b - intersection
    iou = intersection / union if union > 0 else 0
    return iou 


def transform(image, target):
    image = transform.ToTensor(image)
    return image, target 



def assign_anchors_to_gt(anchors, gt_boxes, pos_iou_thesh=0.7, neg_iou_thres=0.3):
    # anchors N,4
    # Return labels : N,  contain 1 +ve 0 -ve -1 ignore 
    N = anchors.size(0)
    M = gt_boxes.size(0)
    labels = torch.full((N,), -1, dtype=torch.int64, device=anchors.device)
    if M == 0 :  
        labels[:] = 0 
        return labels , torch.zeros(N, dtype=torch.int64, device=anchors.device)# return dummy data
    ious = box_iou(anchors, gt_boxes) # (N,M)
    max_ious, max_gt_idx = ious.max(dim=1) # for each anchor
    labels[max_ious < neg_iou_thres] = 0
    labels[max_ious >= pos_iou_thesh] = 1
    # ensure evey box has at least one anchor (best match)
    gt_max_iou, gt_argmax_anchors = ious.max(dim=0) # for each Gt box

    for gt_idx, anchor_idx in enumerate(gt_argmax_anchors):
        labels[anchor_idx] = 1 # force it to be positive 
    return labels , max_gt_idx


def subsample_labels(labels, num_samples=256, pos_fraction=0.5):
    pos_inds = torch.nonzero(labels == 1).squeeze(1)
    neg_inds = torch.nonzero(labels == 0).squeeze(1)

    num_pos  = int(num_samples * pos_fraction)
    num_pos = min(pos_inds.numel(), num_pos)
    num_neg = num_samples - num_pos
    num_neg = min(neg_inds.numel(), num_neg)
    
    # Randomly disable extra positive s 
    if pos_inds.numel() > num_pos :
        p = torch.randperm(pos_inds.numel())[:pos_inds.numel() - num_pos]
        disable_pos = pos_inds[p]
        labels[disable_pos] = -1
    
    # randomly disable extras negatives 
    if neg_inds.numel() > num_neg :
        p = torch.randperm(neg_inds.numel())[:neg_inds.numel() -  num_neg]
        disable_neg = neg_inds[p]
        labels[disable_neg] = -1
    return labels 


def encode_propsals(anchors, gt_boxes):
    # anchors = [xmin, ymin, xmax, ymax]
    # gt_boxes = [xmin, ymin, xmax, ymax]
    # return targets for regression (dx, dy, dw, dh)
    ax, ay, ax2, ay2 = anchors.unbind(-1)
    gx, gy, gx2, gy2 = gt_boxes.unbind(-1)
    anchor_width = ax2 - ax
    anchor_height = ay2 -ay
    gt_width = gx2 - gx
    gt_height = gy2 - gy
    # first convert boxes to (ctr_x, ctr_y, w, h)
    cx_anchor = ax + 0.5 * anchor_width 
    cy_anchor = ay + 0.5 * anchor_height
    cx_gt = gx + 0.5 * gt_width
    cy_gt = gy + 0.5 * gt_height 

    tx = (cx_gt - cx_anchor) / anchor_width
    ty = (cy_gt - cy_anchor) / anchor_height
    tw = torch.log(gt_width / anchor_width)
    th = torch.log(gt_height / anchor_height)
    return torch.stack([tx,ty,tw,th], dim=1)



def encode_propsals(anchors, gt_boxes):
    # anchors = [xmin, ymin, xmax, ymax]
    # gt_boxes = [xmin, ymin, xmax, ymax]
    # return targets for regression (dx, dy, dw, dh)
    ax, ay, ax2, ay2 = anchors.unbind(-1)
    gx, gy, gx2, gy2 = gt_boxes.unbind(-1)
    
    # add it to pervent zero division 
    anchor_width = torch.clamp(ax2 - ax, min=1e-5)
    anchor_height = torch.clamp(ay2 - ay, min=1e-5)
    gt_width = torch.clamp(gx2 - gx, min=1e-5)
    gt_height = torch.clamp(gy2 - gy, min=1e-5)
    
    # first convert boxes to (ctr_x, ctr_y, w, h)
    cx_anchor = ax + 0.5 * anchor_width 
    cy_anchor = ay + 0.5 * anchor_height
    cx_gt = gx + 0.5 * gt_width
    cy_gt = gy + 0.5 * gt_height 

    tx = (cx_gt - cx_anchor) / anchor_width
    ty = (cy_gt - cy_anchor) / anchor_height
    tw = torch.log(gt_width / anchor_width)
    th = torch.log(gt_height / anchor_height)
    
    return torch.stack([tx, ty, tw, th], dim=1)