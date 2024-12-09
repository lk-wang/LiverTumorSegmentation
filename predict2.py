
'''
使用训练好的模型对图片进行预测，将input  predict mask  图片绘制出来，放在同一张图上对比
'''
import os
from PIL import Image
from datetime import datetime
import argparse
import logging
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
import csv
import cv2
import joblib
# 绘图
import matplotlib.pyplot as plt

# 自己写的函数引用
from dataset.dataset_lits import MyDataset
from utils.metrics import get_dice,get_hd,get_metrics    # 评价指标
# 从train.py中引用
from train import getArgs,getModel

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:3")


def getLog(args,predict_result_path):
    filename = predict_result_path + '/predict.log'
    logging.basicConfig(
        filename=filename,
        level=logging.INFO,
        format='%(message)s'
    )
    return logging

def getCsv(csv_path,csv_name):
    frame = pd.DataFrame(columns=['step', 'dice', 'precision','recall','iou','MIoU'])  # 列名
    frame.to_csv(os.path.join(csv_path,csv_name), index=False)  # 路径可以根据需要更改

def getDataset(args):

    #test_dataloader = None LITS&3DIRCADB2
    ds=MyDataset(data_path=r'./data/3DIRCADB',
                 list_name='test_path_list.txt',aug=False)

    test_dataloader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=1, drop_last=True)  #

    return test_dataloader   # 346  一共692张图片346个数据

def predict(test_dataloader,predict_result_path,save_predict:bool):

    logging.info('........Start predict!........')
    # 保存到csv中
    results_file = "predict_results.csv"
    getCsv(predict_result_path, results_file)
    # 预测图片的保存路径
    predict_plot=os.path.join(predict_result_path,'plot')
    if not os.path.exists(predict_plot):
        os.makedirs(predict_plot)
    # 加载模型
    tmp_path = "./save_result/{}/train_result".format(args_path)
    checkpoint_path = os.path.join(tmp_path, 'checkpoint')
    best_model_path = os.path.join(tmp_path, 'best_model')
    # 从最优模型处加载
    path_checkpoint =os.path.join(best_model_path,'model-{}-{}.pth'.format(257,0.8414))  #
    # 从断点处加载
    #path_checkpoint = os.path.join(checkpoint_path, 'model-{}.pth'.format(4))
    best_model_checkpoint = torch.load(path_checkpoint, map_location=device)
    model.load_state_dict(best_model_checkpoint['model'])  # 载入训练好的模型
    model.eval()  # **********

    #plt.ion() #开启动态模式
    with torch.no_grad():
        # 验证指标
        dice_total = 0
        precision_total = 0
        recall_total = 0
        iou_total = 0
        mIou_total = 0
        num = len(test_dataloader)  #验证集图片的总数
        for i, (pic,mask,pic_path,mask_path) in enumerate(test_dataloader):
            i+=1
            pic = pic.to(device)
            mask=mask.to(device)

            predict = model(pic) # (1,1,512,512)
            array_predict = torch.squeeze(predict).cpu().numpy() # (512,512) #输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batch_size

            # 使用y和predict来计算指标
            dice = get_dice(predict, mask).item()
            precision, recall, iou, mIou = get_metrics(mask, predict)
            
            dice_total += dice
            precision_total += precision
            recall_total += recall
            iou_total += iou
            mIou_total += mIou

            print('\n[{}/{}]:  dice={:.4f},  precision={:.4f},  recall={:.4f}'.format(i,num,dice,precision,recall))     # dice和precision.recall
            logging.info('[{}/{}]: dice={:.4f}, precision={:.4f}, recall={:.4f}, iou={:.4f}, mIou={:.4f}'.format(i,num,dice,precision,recall,iou,mIou))
            
            # 将结果写入csv列表中
            with open(os.path.join(predict_result_path,results_file), 'a', encoding='utf-8', newline='') as f:
                wr = csv.writer(f)
                wr.writerow([i,dice,precision,recall,iou,mIou])
            
            
            # 保存图片   
            if save_predict is True:
                dice_str=str(dice)
                dice_str=dice_str[0:6]
                # 3DDIRCADB  ct-0.png  tumor-0.png
                pic_name = dice_str + '-' + str(pic_path[0].split('/')[-1]).replace('.png','')

                # LITS  0-46_ct.png  0-46_tumor-0.png
                #pic_name = dice_str + '-ct-'+str(pic_path[0].split('/')[-1]).replace('_ct.png', '')

                plot_path = os.path.join(predict_plot,pic_name)
                if not os.path.exists(plot_path):
                    os.makedirs(plot_path)

                array_predict[array_predict >= 0.5] = 1
                array_predict[array_predict < 0.5] = 0

                image_ = cv2.imread(pic_path[0])
                plt.imsave(plot_path + '/' + 'image.png', image_)  # 保存图

                mask_ = cv2.imread(mask_path[0])
                plt.imsave(plot_path + '/' + 'mask.png' ,mask_)  # 保存图

                plt.imsave(plot_path + '/' + 'predict.png', array_predict*255, cmap='gray')  # 保存
            

        aver_dice = dice_total/num
        aver_precision = precision_total/num
        aver_recall = recall_total/num
        aver_iou = iou_total/num
        aver_mIou = mIou_total/num
        print("aver_dice:{:.4f}, aver_precision:{:.4f}, aver_recall:{:.4f}".format(aver_dice,aver_precision,aver_recall))
        logging.info(
            "aver_dice:{:.4f}, aver_precision:{:.4f}, aver_recall:{:.4f}, aver_IoU:{:.4f}, aver_MIoU:{:.4f}"
            .format( aver_dice, aver_precision, aver_recall, aver_iou, aver_mIou))

if __name__ == "__main__":

    args = joblib.load(r"/home/data/WB_/Test_Compare2/save_result/3DIRCADB_Res_U_Net_CA2_SAB_DGA__300_8/train_result/args.pkl")
    args_path = str(args.dataset) + "_" + str(args.model) + "_" + str(args.epoch) + "_" + str(args.batch_size)

    predict_result_path="./save_result/{}/predict2_result".format(args_path)
    if not os.path.exists(predict_result_path):
        os.makedirs(predict_result_path)
    logging = getLog(args,predict_result_path)      # 日志

    print('===================>')
    print('Parameters of model training:\ndevice:%s,\nmodels:%s,\nepoch:%s,\nbatch size:%s\ndataset:%s' % \
          (device,args.model, args.epoch, args.batch_size, args.dataset))
    logging.info('\n=======\nParameters of model training:\nmodels:%s,\nepoch:%s,\nbatch size:%s\ndataset:%s\n========' % \
                 (args.model, args.epoch, args.batch_size, args.dataset))
    print('===================>')
    model = getModel(args).to(device)  # 选择使用的模型
    test_dataloader = getDataset(args)
    # 测试函数
    print("!---------Start predict!---------!")
    predict(test_dataloader, predict_result_path,save_predict=True)



