import torch


def compute_iou(annotation, segmentation, eps=1e-7):
    if segmentation.shape[1] > 1:
        segmentation = torch.softmax(segmentation, dim=1)
        segmentation = torch.argmax(segmentation, dim=1, keepdim=True)
    if len(annotation.shape) == 4:
        annotation = annotation.view(-1)
    if len(segmentation.shape) == 4:
        segmentation = segmentation.view(-1)
    annotation = annotation.long()
    segmentation = segmentation.long()
    assert annotation.size() == segmentation.size()
    return torch.sum((annotation & segmentation)) / (torch.sum((annotation | segmentation)) + eps)
