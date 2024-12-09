import cv2
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
from skimage.io import imread
import imageio


def get_iou(mask_name,predict):
    image_mask = cv2.imread(mask_name,0)
    if np.all(image_mask == None):
        image_mask = imageio.mimread(mask_name)
        image_mask = np.array(image_mask)[0]
        image_mask = cv2.resize(image_mask,(576,576))
    #image_mask = mask
    # print(image.shape)
    height = predict.shape[0]
    weight = predict.shape[1]
    # print(height*weight)
    o = 0
    for row in range(height):
            for col in range(weight):
                if predict[row, col] < 0.5:  #由于输出的predit是0~1范围的，其中值越靠近1越被网络认为是肝脏目标，所以取0.5为阈值
                    predict[row, col] = 0
                else:
                    predict[row, col] = 1
                if predict[row, col] == 0 or predict[row, col] == 1:
                    o += 1
    height_mask = image_mask.shape[0]
    weight_mask = image_mask.shape[1]
    for row in range(height_mask):
            for col in range(weight_mask):
                if image_mask[row, col] < 125:   #由于mask图是黑白的灰度图，所以少于125的可以看作是黑色
                    image_mask[row, col] = 0
                else:
                    image_mask[row, col] = 1
                if image_mask[row, col] == 0 or image_mask[row, col] == 1:
                    o += 1
    predict = predict.astype(np.int16)

    interArea = np.multiply(predict, image_mask)
    tem = predict + image_mask
    unionArea = tem - interArea
    inter = np.sum(interArea)
    union = np.sum(unionArea)
    iou_tem = inter / union

    # Iou = IOUMetric(2)  #2表示类别，肝脏类+背景类
    # Iou.add_batch(predict, image_mask)
    # a, b, c, d, e= Iou.evaluate()
    # 打印当前图片以及预测的iou
    #print('%s:iou=%f' % (mask_name,iou_tem))

    return iou_tem

# 评价指标对比的是原始图像和预测结果
def get_dice(mask_name,predict):    # 3维张量   ？？通道
    image_mask = cv2.imread(mask_name, 0)    # 读取训练样本标签，0：灰度图，1：彩色图，-1：完整图，反回（高度，宽度，通道数）的元组。
    # all()函数用于判断整个数组中的元素的值是否全部满足条件，如果满足条件返回True，否则返回False。
    if np.all(image_mask == None):
        image_mask = imageio.mimread(mask_name)      # 从指定文件中读取多个图像。返回numpy数组，
        image_mask = np.array(image_mask)[0]         # 转换成array
        image_mask = cv2.resize(image_mask,(576,576))
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
    predict = predict.astype(np.int16)
    intersection = (predict*image_mask).sum()
    dice = (2. *intersection) /(predict.sum()+image_mask.sum())
    return dice



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





# 显示预测图
def show(predict):
    height = predict.shape[0]
    weight = predict.shape[1]
    for row in range(height):
        for col in range(weight):
            predict[row, col] *= 255
    plt.imshow(predict)
    plt.show()