# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area

def paired_box_to_score(x, type="min-size"):
    """
    x: N*8, N*[subject_bbox, object_bbox], [cx,cy,w,h, cx,cy,w,h]
    """
    subject_sizes = x[:, 2] * x[:, 3]
    object_sizes = x[:, 6] * x[:, 7]
    sizes = torch.stack([subject_sizes, object_sizes], dim=1)
    if type == "min-size":
        scores = torch.min(sizes, dim=1)[0]
    elif type == "max-size":
        scores = torch.max(sizes, dim=1)[0]
    elif type == "center-dis":
        dis = torch.cdist(x[:, 0:2], x[:, 4:6], p=2)
        scores = torch.diag(dis)
    elif type == "rel-center-dis":
        absulute_dis = torch.cdist(x[:, 0:2], x[:, 4:6], p=2)
        subject_bboxs = x[:, 0:4] # [cx,cy,w,h]
        object_bboxs = x[:, 4:8] # [cx,cy,w,h]
        subject_bboxs = box_cxcywh_to_xyxy(subject_bboxs)
        object_bboxs = box_cxcywh_to_xyxy(object_bboxs)
        assert (subject_bboxs[:, 2:] >= subject_bboxs[:, :2]).all()
        assert (object_bboxs[:, 2:] >= object_bboxs[:, :2]).all()
        iou, union = box_iou(subject_bboxs, object_bboxs)
        relative_dis = absulute_dis / torch.sqrt(union)
        scores = torch.diag(relative_dis)
    else:
        raise NotImplementedError
    return scores

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
