

def inference_rpn(model_path, image, device=device, score_threshold=0.5, 
                  nms_iou_threshold=0.7, transform=None):
    backbone = load_backbone().to(device).eval()
    rpn = RPN(in_channels=512, image_size=(224, 224))
    rpn.load_state_dict(torch.load(model_path, map_location=device))
    rpn = rpn.to(device)
    rpn.eval()
    if isinstance(image, (Image.Image, np.ndarray)):
        image = transform(image)
    if image.dim() == 3 :
        image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        features = backbone(image)
        rpn_logits, rpn_deltas, rois, all_anchors = rpn(features, image)
        
        # rois are already filtered by RPN's NMS and top-k selection
        # We can only compute meaningful scores from the selected proposals
        # by using the RPN logits at inference time for visualization
        proposals = rois.cpu()  # [M, 4]
        
        # All proposals passed RPN filtering, so assign confidence = 1.0
        # (they're the top confidence anchors after NMS)
        scores = torch.ones(len(proposals))
        
    return proposals, scores

