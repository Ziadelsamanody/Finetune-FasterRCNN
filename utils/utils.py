import torch 
from torchvision.ops import box_iou 


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