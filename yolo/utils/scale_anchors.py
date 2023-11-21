import torch

def scale_anchors(anchors, scale, img_w, img_h, device="cpu"):
    scaler = torch.tensor([
        img_w / min(scale, img_w), 
        img_h / min(scale, img_h)
    ]).to(device)
    scaled_anchors = anchors * scaler
    return scaled_anchors
