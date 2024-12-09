
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import cv2
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
from skimage.io import imread
import imageio


def get_hd(mask_name,predict):
    image_mask = cv2.imread(mask_name, 0)
    # print(mask_name)
    # print(image_mask)
    if np.all(image_mask == None):
        image_mask = imageio.mimread(mask_name)
        image_mask = np.array(image_mask)[0]
        image_mask = cv2.resize(image_mask,(576,576))
    #image_mask = mask
    height = predict.shape[0]
    weight = predict.shape[1]
    o = 0
    for row in range(height):
        for col in range(weight):
            if predict[row, col] < 0.5:  # 由于输出的predit是0~1范围的，其中值越靠近1越被网络认为是肝脏目标，所以取0.5为阈值
                predict[row, col] = 0
            else:
                predict[row, col] = 1
            if predict[row, col] == 0 or predict[row, col] == 1:
                o += 1
    height_mask = image_mask.shape[0]
    weight_mask = image_mask.shape[1]
    for row in range(height_mask):
        for col in range(weight_mask):
            if image_mask[row, col] < 125:  # 由于mask图是黑白的灰度图，所以少于125的可以看作是黑色
                image_mask[row, col] = 0
            else:
                image_mask[row, col] = 1
            if image_mask[row, col] == 0 or image_mask[row, col] == 1:
                o += 1
    hd1 = directed_hausdorff(image_mask, predict)[0]
    hd2 = directed_hausdorff(predict, image_mask)[0]
    res = None
    if hd1>hd2 or hd1 == hd2:
        res=hd1
        return res
    else:
        res=hd2
        return res



def get_mean_iou(mask, pred, classes = 2):
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    pred = torch.squeeze(pred).cpu().numpy()  # (512,512)
    mask = torch.squeeze(mask).cpu().numpy()  # (512,512)
    miou = 0
    for i in range(classes):
        intersection = np.logical_and(mask == i, pred == i)
        union = np.logical_or(mask == i, pred == i)
        temp = np.sum(intersection) / np.sum(union)
        miou += temp
    return  miou/classes

def get_acc(mask, pred):
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    pred = torch.squeeze(pred).cpu().numpy()  # (512,512)  .cpu()
    mask = torch.squeeze(mask).cpu().numpy()  # (512,512)
    matrix = confusion_matrix(y_true=np.array(mask).flatten(), y_pred=np.array(pred).flatten())
    acc = np.diag(matrix).sum() / matrix.sum()
    return acc

def get_recall(mask,pred):
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    pred = torch.squeeze(pred).cpu().numpy()  # (512,512)
    mask = torch.squeeze(mask).cpu().numpy()  # (512,512)
    pred = np.atleast_1d(pred.astype(bool))  # np.
    mask = np.atleast_1d(mask.astype(bool))  # np.
    tp = np.count_nonzero(pred & mask)
    fn = np.count_nonzero(~pred & mask)
    recall = tp / float(tp + fn+1e-6)
    return recall

# pred和target的shape为【batch_size,channels,...】，数据类型为tensor
def get_dice(pred, target):
    pred[pred>=0.5]=1       # 将张量元素转换成0/1，使用这一步，一般结果会优于没有使用这一步
    pred[pred<0.5]=0        #
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)



def get_metrics(mask,pred):
    # 二值分割图是一个波段的黑白图，正样本值为1，负样本值为0
    # 通过矩阵的逻辑运算分别计算出tp,tn,fp,fn
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    pred = torch.squeeze(pred).cpu().numpy()  # (512,512)  .cpu()
    mask = torch.squeeze(mask).cpu().numpy()  # (512,512)

    seg_inv, gt_inv = np.logical_not(pred), np.logical_not(mask)

    true_pos = float(np.logical_and(pred, mask).sum())  # float for division
    true_neg = np.logical_and(seg_inv, gt_inv).sum()
    false_pos = np.logical_and(pred, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, mask).sum()

    # 然后根据公式分别计算出这几种指标
    Precision = true_pos / (true_pos + false_pos + 1e-6)
    Recall = true_pos / (true_pos + false_neg + 1e-6)
    #Accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg + 1e-6)
    # 它的最大值是1，最小值是0，值越大意味着模型越好。
    #F1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    # IoU正例 IoU = TP / (TP + FP + FN)
    IoU = true_pos / (true_pos + false_neg + false_pos + 1e-6)
    # IoU反例 IoU_= TN/(TN + FN + FP)
    IoU_=true_neg/(true_neg+false_neg+false_pos)
    # MIoU = (IoU正例p + IoU反例n) / 2 = [ TP/(TP + FP + FN) +TN/(TN + FN + FP) ]/2
    mIou=(IoU+ IoU_ ) / 2

    #return Precision,Recall,Accuracy,F1,IoU,MIoU
    return Precision,Recall,1-IoU


# 计算DICE系数，即DSI
def calDSI(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    DSI_s, DSI_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
                DSI_s += 1
            if binary_GT[i][j] == 255:
                DSI_t += 1
            if binary_R[i][j] == 255:
                DSI_t += 1
    DSI = 2 * DSI_s / DSI_t
    # print(DSI)
    return DSI


# 计算VOE系数，即VOE
def calVOE(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    VOE_s, VOE_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255:
                VOE_s += 1
            if binary_R[i][j] == 255:
                VOE_t += 1
    VOE = 2 * (VOE_t - VOE_s) / (VOE_t + VOE_s)
    return VOE


# 计算RVD系数，即RVD
def calRVD(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    RVD_s, RVD_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255:
                RVD_s += 1
            if binary_R[i][j] == 255:
                RVD_t += 1
    RVD = RVD_t / RVD_s - 1
    return RVD


# 计算Prevision系数，即Precison
def calPrecision(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    P_s, P_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
                P_s += 1
            if binary_R[i][j] == 255:
                P_t += 1

    Precision = P_s / P_t
    return Precision


# 计算Recall系数，即Recall
def calRecall(binary_GT, binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    R_s, R_t = 0, 0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
                R_s += 1
            if binary_GT[i][j] == 255:
                R_t += 1

    Recall = R_s / R_t
    return Recall



# 显示预测图
def show(predict):
    height = predict.shape[0]
    weight = predict.shape[1]
    for row in range(height):
        for col in range(weight):
            predict[row, col] *= 255
    plt.imshow(predict)
    plt.show()







