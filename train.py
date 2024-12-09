from datetime import datetime
import argparse
import logging
import torch
from collections import OrderedDict
from torch.utils.data import DataLoader, random_split,Dataset
from torchvision import transforms
from torch import optim
from tqdm import tqdm
import csv
import pandas as pd
import os
import joblib
from torch.utils.tensorboard import SummaryWriter
import time
import torch.backends.cudnn as cudnn


# model
from model.UNet import U_Net    # 


# QAUNet
#from QAUNet.QAU_Net import QAU_Net  # 

# UCTransNet
#from UCTransNet.UCTransNet import UCTrans_Net  # 



from dataset.dataset_lits import MyDataset
# 引入自己写的函数
from utils.metrics import get_dice,get_hd,get_metrics    # 评价指标
from utils.plot import loss_plot, metrics_plot,metrics_plots  # 绘制loss图,评价指标图
from utils.loss import BceDiceLoss,FocalTverskyLoss

device = torch.device("cuda:3")   # 2，3

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def getArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument("--name", type=str, default='')
    parse.add_argument("--img_size", type=str, default='448x448')
    
                       
    parse.add_argument('--model',  default='U_Net',help='---U_Net----')  #   SegNet


    parse.add_argument('--dataset', default='LITS3', help='LITS')  # 
    parse.add_argument("--epoch", type=int, default=300)
    parse.add_argument("--batch_size", type=int, default=16)
    parse.add_argument('--lr', default=1e-5, type=float, help='initial learning rate')
    parse.add_argument('--early-stop', default=50, type=int,              # 50
                        metavar='N', help='early stopping (default: 30)')
    parse.add_argument("--resume", type=bool, default="", help="resume the training from checkpoint")  # 从断点处恢复训练
    parse.add_argument("--threshold",type=float,default=None)                # loss阈值
    args = parse.parse_args()
    return args

def getDataset(args):
    
    ds=MyDataset(data_path=r'./data/LITS3',list_name='train_path_list.txt',aug=True)

    train_size = int(0.8 * len(ds))  # 整个训练集中，百分之80为训练集
    val_size = len(ds) - train_size
    train_dataset, val_dataset = random_split(ds, [train_size, val_size])  # 划分训练集和验证集

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)  #
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=True)  #

    return train_dataloader,val_dataloader


def getModel(args):
  
    if args.model == 'U_Net':
        model = U_Net(3,1)
         
    #if args.model == 'QAU_Net':
        #model = QAU_Net(3,1)
        
    #if args.model == 'UCTrans_Net':    
        #model = UCTrans_Net
           
    return model


def getLog(args,path):
   #                              train_result/tumor_U_Net_150_6
    filename = path+'/train.log'
    logging.basicConfig(
            filename=filename,
            level=logging.INFO,
            format = '%(message)s'
        )
    return logging

def getCsv(csv_path,csv_name):
    frame = pd.DataFrame(columns=['epoch', 'dice', 'precision','recall','voe'])  # 列名
    frame.to_csv(os.path.join(csv_path,csv_name), index=False)  # 路径可以根据需要更改

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # print('===========>alpha',alpha,x,warmup_epochs , num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)  # ,last_epoch=229


# best_iou初始为0
def val(epoch,model,val_dataloader,criterion):
    # 验证指标
    loss_total = 0
    dice_total = 0
    precision_total = 0
    recall_total = 0
    voe_total = 0
    # iou_total = 0
    # mIou_total = 0
    val_step = 0
    num = len(val_dataloader)  # 验证集图片的总数
    model= model.eval()    # **********
    with torch.no_grad():
        for pic, mask,_,_ in tqdm(val_dataloader):   # bath_size需要设为1
        #for pic, mask, pic_path, mask_path in val_dataloader:  # bath_size需要设为1
            val_step+=1
            pic = pic.to(device)
            mask=mask.to(device)
            predict = model(pic)
            val_loss = criterion(predict, mask)
  
            img_predict = torch.squeeze(predict).cpu().numpy()  

            # 使用y和predict来计算指标
            dice=get_dice(predict,mask)
            precision, recall,voe=get_metrics(mask,predict)    # ,  iou, MIoU
            # val的度量标准
            loss_total += val_loss.item()
            dice_total += dice.item()
            precision_total += precision
            recall_total += recall
            voe_total += voe
            
            # 使用tensorboard绘制val指标图
            #print('当前步数：{}'.format(val_step + epoch * num))
            #writer.add_scalar('val_loss', val_loss.item(), val_step + epoch * num)
            #writer.add_scalar('val_dice', dice.item(), val_step + epoch*num)
            #writer.add_scalar('val_precision', precision,val_step + epoch*num)
            #writer.add_scalar('val_recall', recall, val_step + epoch*num)

        # 计算平均评价指标
        aver_loss = loss_total / num
        aver_dice = dice_total/num
        aver_precision=precision_total / num
        aver_recall=recall_total / num
        aver_voe=voe_total / num
        
        # 使用tensorboard绘制平均val指标图
        #writer.add_scalar('aver_dice', aver_dice, epoch)
        #writer.add_scalar('aver_precision', aver_precision, epoch)
        #writer.add_scalar('aver_recall', aver_recall, epoch)

        val_log = OrderedDict([
        ('aver_loss', aver_loss),
        ('aver_dice', aver_dice),
        ('aver_precision', aver_precision),
        ('aver_recall', aver_recall),
        ('aver_voe', aver_voe),
        ])

        return val_log

#def train(model, criterion, optimizer, train_dataloader,epoch,lr_scheduler,args):
def train(model, criterion, optimizer, train_dataloader,epoch,args):
    num_epochs = args.epoch
    threshold = args.threshold
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    logging.info('Epoch {}/{}'.format(epoch+1, num_epochs))
    #ds_size = len(train_dataloader.dataset)    
    num=len(train_dataloader)               
    train_step = 0
    epoch_loss = 0
   
    model = model.train()  # ************
    #for pic, mask,pic_path,mask_path in train_dataloader:
    for pic, mask,_,_ in train_dataloader:
        train_step += 1
        inputs = pic.to(device)
        labels = mask.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)  
        
        # 整个train过程的每一步的loss
        #writer.add_scalar('train_loss', loss.item() ,  train_step+epoch*num)

        if threshold!=None:                     
            if loss > threshold:
                loss.backward()
                optimizer.step()
                epoch_loss = epoch_loss + loss.item()
        else:
            loss.backward()
            optimizer.step()
            epoch_loss = epoch_loss + loss.item()    # 整个epoch的所有损失和

        
        lr = optimizer.param_groups[0]["lr"]  
        
        #lr_scheduler.step()
        

        print("Epoch[{}] {}/{} train: loss = {:.5f} lr = {:.6f} " .format(epoch+1,train_step, num , loss.item(),lr))
        #logging.info("%d/%d,train_loss:%0.3f" % (train_step, num, loss.item() ))

    epoch_aver_loss=epoch_loss/num  # 计算一个epoch的平均loss  单个epoch的总loss/计算loss的数量

    #return epoch_aver_loss,lr_scheduler,lr
    return epoch_aver_loss,lr



if __name__ == '__main__':
    args = getArgs()
    start_time =datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    args_path=str(args.dataset) + "_" + str(args.model) + "_" + str(args.epoch) + "_" + str(args.batch_size)
    # 创建需要保存文件的存储路径
    train_result_path = "./save_result/{}/train_result".format(args_path)
    if not os.path.exists(train_result_path):
        os.makedirs(train_result_path)

    # 日志路径
    logging = getLog(args,train_result_path)
    # 最优模型路径
    best_model_path=os.path.join(train_result_path,'best_model')
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    # 检查点路径
    checkpoint_path = os.path.join(train_result_path, 'checkpoint')
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # 保存到csv中
    csv_log = pd.DataFrame(index=[], columns=['epoch', 'dice', 'precision','recall','iou','MIoU'])

    # 训练参数信息
    print('==========>*<==========')
    print('device:%s,\nmodels:%s,\ninitial_lr:%s,\nepoch:%s,\nbatch size:%s\ndataset:%s\nimg_size:%s' % \
          (device,args.model, args.lr,args.epoch, args.batch_size,args.dataset,args.img_size))

    logging.info('\n==============>\nStartTime:%s,\ndevice:%s,\nmodels:%s,\ninitial_lr:%s,\nepoch:%s,\nbatch size:%s\ndataset:%s\nimg_size:%s' % \
          (start_time,device,args.model, args.lr,args.epoch, args.batch_size,args.dataset,args.img_size))

    # 保存参数
    joblib.dump(args, '{}/args.pkl'.format(train_result_path))

    

    model = getModel(args).to(device)  # 选择使用的模型
    
    
    #torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
  
    # model = torch.nn.DataParallel(model).cuda()
    print("Number of model parameters:", count_params(model))
    logging.info('Number of model parameters:%s\n========', count_params(model))
    print('==========>*<==========')
    train_dataloader,val_dataloader = getDataset(args)
    
    # ---------------------------------------------- loss function ---------------------------------------------- #
    criterion=BceDiceLoss()
    #criterion = torch.nn.BCELoss()
    #criterion=FocalTverskyLoss()
    
    # ---------------------------------------------- 优化器 ---------------------------------------------- #
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam( params_to_optimize,lr=args.lr)   # ,weight_decay=1e-8
    #optimizer = optim.Adam([{'params': params_to_optimize, 'initial_lr': 1e-5}],lr = args.lr)
    
    # ---------------------------------------------- 学习策略 ---------------------------------------------- #
    
    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    
    #lr_scheduler = create_lr_scheduler(optimizer, len(train_dataloader), args.epoch, warmup=True)
    
    # CosineAnnealingLR 余弦退火调整学习率
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,eta_min=0, last_epoch=-1)
    
    # 自适应的学习策略
    #lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

    start_epoch = 0  # 原始的开始epoch
    best_metric = 0
    # 断点恢复,从best_model开始加载
    if args.resume:
        # 从最优模型处加载
        #path_checkpoint =os.path.join(best_model_path,'model-{}-{}.pth'.format(8,0.6179))
        # 从断点处加载
        path_checkpoint =os.path.join(checkpoint_path,'model-{}.pth'.format(219))
        checkpoint = torch.load(path_checkpoint, map_location=device)  # 加载断点
        model.load_state_dict(checkpoint['model'])  # 加载模型可学习参数
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] # 设置开始的epoch
        best_metric=checkpoint['metric']
        print("使用断点恢复！")
        
        
    #path_checkpoint =os.path.join(checkpoint_path,'model-{}.pth'.format(219))
    #checkpoint = torch.load(path_checkpoint, map_location=device)  # 加载断点
    #model.load_state_dict(checkpoint['model'])  # 加载模型可学习参数

    epoch_aver_loss=0
   
    trigger = 0   # 早停触发器
    #writer = SummaryWriter(train_result_path+'/logs')   # 使用tensorboaed 
    
    # 绘图列表
    train_loss_list = []
    val_loss_list = []
    dice_list = []
    precision_list = []
    recall_list = []
    
    for epoch in range(start_epoch,args.epoch):
        start_time = time.time()

        #epoch_aver_loss,lr_scheduler,lr = train(model, criterion, optimizer, train_dataloader,epoch,lr_scheduler,args)
        epoch_aver_loss,lr = train(model, criterion, optimizer, train_dataloader,epoch,args)
       
    
        print("train: loss = {:.5f}".format(epoch_aver_loss))
        # 记录一个epoch的训练平均loss
        logging.info("train: loss = {:.5f} lr = {}".format(epoch_aver_loss,lr))

        # 调用验证函数
        val_log = val(epoch, model, val_dataloader,criterion)
        
        print('val:   loss = {:.5f} dice = {:.4f} precision = {:.4f} recall = {:.4f}'
              .format(val_log['aver_loss'],val_log['aver_dice'], val_log['aver_precision'], val_log['aver_recall']))
        # 验证集的验证指标保存到日志文件
        # logging.info('=====>Validate metrics!')
        logging.info('val:   loss = {:.5f} dice = {:.4f} precision = {:.4f} recall = {:.4f}'
                     .format(val_log['aver_loss'],val_log['aver_dice'], val_log['aver_precision'], val_log['aver_recall'])) # , aver_iou, aver_MIoU
        
        #lr_scheduler.step(val_log['aver_loss'])  # 通过监控val_loss来调整学习率
        
        # 绘图列表
        train_loss_list.append(epoch_aver_loss)
        val_loss_list.append(val_log['aver_loss'])
        dice_list.append(val_log['aver_dice'])
        precision_list.append(val_log['aver_precision'])
        recall_list.append(val_log['aver_recall'])
        
        # 模型的保存文件
        save_file = {
            "model": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch,
            "metric":val_log['aver_dice'],
            #"lr_scheduler": lr_scheduler.state_dict(),
        }

        trigger += 1
        # 保存模型，选择最佳模型
        if val_log['aver_dice'] > best_metric:
            print('val_dice = {:.4f} > best_metric = {:.4f}'.format(val_log['aver_dice'], best_metric))
            logging.info('val_dice = {:.4f} > best_metric = {:.4f}'.format(val_log['aver_dice'], best_metric))
            logging.info('===========>save best model!')
            best_metric = val_log['aver_dice']
            print('===========>save best model!')
            # 保存最优模型  model.state_dict()
            torch.save(save_file,os.path.join(best_model_path,'model-{}-{:.4f}.pth'.format(epoch,val_log['aver_dice'])))
            trigger = 0

        if (epoch+1) % 10 == 0:
            # 检查点路径
            torch.save(save_file, '{}/model-{}.pth'.format(checkpoint_path,epoch))
            print('===========>Saving checkpoint: model-{}.pth'.format(epoch))

        # 使用pandas.concat,将结果写入csv列表中
        csv_tmp = pd.DataFrame({'epoch': [epoch],
                            'dice': [val_log['aver_dice']],
                            'precision': [val_log['aver_precision']],
                            'recall': [val_log['aver_recall']],
                            'voe': [val_log['aver_voe']],
                            # 'MIoU': [val_log['aver_mIou']]
                            })

        csv_log = pd.concat([csv_log, csv_tmp], ignore_index=True)
        csv_log.to_csv(train_result_path+'/train_result.csv', index=False)

        total_time = time.time() - start_time
        print("A full epoch training and val time: {:.2f}".format(total_time))
        
        # early stopping
        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print("==========> early stopping")
                logging.info("==========> early stopping")
                break
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

      
        #writer.flush()  # 用于刷新此Writer流
        
    # 绘制loss和metrics图并保存
    plot_save_path = train_result_path + '/plot/'
    # loss_plot(args=args, loss_list=train_loss_list, plot_save_path=plot_save_path,
    #           name='train loss')  # loss图:epoch和epoch_loss
    metrics_plots(args, 'train loss&val loss', plot_save_path, train_loss_list, val_loss_list)  # train loss和val loss图
    metrics_plot(args=args, metrics_list=dice_list, plot_save_path=plot_save_path, name='dice')  # dice图
    metrics_plot(args=args, metrics_list=precision_list, plot_save_path=plot_save_path, name='precision')  # precision图
    metrics_plot(args=args, metrics_list=recall_list, plot_save_path=plot_save_path, name='recall')  # recall图
    # metrics_plots(args, 'iou&MIoU', iou_list, MIoU_list)  # iou和MIoU 图
    
    print("===============> end!!!")
    logging.info("===============> end!!!")
    #writer.close()


# tensorboard --logdir=E:\PycharmProjects\CodeTemplate\RALUNet\save_result\tumor_U_Net_20_2\train_result\logs


