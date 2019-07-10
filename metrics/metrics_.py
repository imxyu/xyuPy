import torch
import numpy as np
import torch.nn as nn
import math

def is_batch(input):
    if len(input.shape) == 3:
        return input.shape[0]
    else:
        return 1

# def IoU(prediction, target, is_training=False):
#     if is_training:
#         prediction = prediction.cpu().detach().numpy()
#         target = target.cpu().detach().numpy()
#     prediction = np.uint8(prediction)
#     target = np.uint8(target)
#     delta = 1e-10
#     IoU = ((prediction * target).sum() + delta) / (prediction.sum() + target.sum() - (prediction * target).sum() + delta)
#     return IoU * is_batch(prediction)

def IoU(prediction, target, is_training=False):
    if is_training:
        iflat = prediction.view(-1).float()
        tflat = target.view(-1).float()
    else:
        iflat = np.reshape(prediction, -1).astype(np.float)
        tflat = np.reshape(target, -1).astype(np.float)
    delta = 1.0e-10
    intersection = (iflat * tflat).sum()


    return ((intersection + delta) /
                (iflat.sum() + tflat.sum() - intersection + delta))

def Dice(prediction, target, is_training=False):
    if is_training:
        iflat = prediction.view(-1).float()
        tflat = target.view(-1).float()
    else:
        iflat = np.reshape(prediction, -1).astype(np.float)
        tflat = np.reshape(target, -1).astype(np.float)
    delta = 1.0e-10
    intersection = (iflat * tflat).sum()
    return (2 * intersection + delta) / (prediction.sum() + target.sum() + delta)




def PA(prediction, target):
    prediction = np.uint8(prediction)
    target = np.uint8(target)
    return np.mean(prediction == target)* is_batch(prediction)



def VOE(prediction,target,is_training=False):
    return 1-IoU(prediction,target,is_training)



def RVD(prediction,target,is_training=False):
    if is_training:
        prediction = prediction.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
    prediction = np.uint8(prediction)
    target = np.uint8(target)
    delta = 1e-10

    RVD = (((target.sum())-(prediction.sum()) + delta) / (prediction.sum()+delta))


    return RVD


def dvs(A,B,minn=True):

    A.flatten()
    B.flatten()
    la = A.size
    lb = B.size

    ret_dis = -1

    for i in range(la):
        for j in range(lb):
            dis = math.sqrt((A[i]-B[j])*(A[i]-B[j]))
            
            if minn:
                if(dis < ret_dis):
                    ret_dis = dis
            else:
                if(dis > ret_dis):
                    ret_dis = dis

    return ret_dis


def ASD(prediction,target,is_training=False):
    if is_training:
        prediction = prediction.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
    prediction = np.uint8(prediction)
    target = np.uint8(target)
    delta = 1e-10

    sa = prediction.sum()
    sb = target.sum()

    dvsAB = dvs(prediction,target)
    dvsBA = dvs(target,prediction)

    return ((dvsAB+dvsBA+delta)/(sa+sb+delta))


def MSD(prediction,target,is_training=False):
    if is_training:
        prediction = prediction.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
    prediction = np.uint8(prediction)
    target = np.uint8(target)
    

    dvsAB_max = dvs(prediction,target,False)
    dvsBA_max = dvs(target,prediction,False)

    return max(dvsAB_max,dvsBA_max)





def RMSE(prediction,target,is_training=False):
    if is_training:
        prediction = prediction.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
    prediction = np.uint8(prediction)
    target = np.uint8(target)
    delta = 1e-10


    pre = prediction.flatten()
    tar = target.flatten()

    length = pre.size
    su = 0
    for i in range(length):
        su += (pre[i]-tar[i])*(pre[i]-tar[i])

    return math.sqrt((su+delta) / (length+delta))
