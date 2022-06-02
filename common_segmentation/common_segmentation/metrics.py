import torch


@torch.no_grad()
def calculate_iou(p, t):
    num_classes = p.size(1)
    p = p.argmax(1)
    res = []
    for i in range(num_classes):
        pc = p == i
        tc = t == i
        nom = (pc & tc).sum().item()
        denom = (pc | tc).sum().item() + 1e-9
        res.append(nom / denom)
    return res
