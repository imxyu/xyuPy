import torch

def is_batch(input):
    if len(input.shape) == 3:
        return input.shape[0]
    else:
        return 1


def iou_loss(input, target):
    smooth = 1.
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((intersection + smooth) /
                (iflat.sum() + tflat.sum() - intersection + smooth))


def dice_loss(input, target):
    smooth = 1.
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth).float() /
                (iflat.sum() + tflat.sum() + smooth))
