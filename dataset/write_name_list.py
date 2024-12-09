
import numpy as np
import os
import random
from os.path import join


valid_rate=0.2

lits_path = r'/home/data/WB_/dataset/LITS_PNG/Liver'

def write_train_val_name_list():
    lits_ct_list=[]
    ircadb_ct_list=[]
    for lits_ct in os.listdir(join(lits_path,'image')):
        lits_ct_path=os.path.join(lits_path,'image',lits_ct)
        lits_ct_list.append(lits_ct_path)

    #for ircadb_ct in os.listdir(join(ircadb_path,'image')):
        #ircadb_ct_path=os.path.join(ircadb_path,'image',ircadb_ct)
        #ircadb_ct_list.append(ircadb_ct_path)

    data_ct_list=lits_ct_list
    #data_ct_list=ircadb_ct_list
    #data_ct_list=lits_ct_list+ircadb_ct_list

    print('The CT dataset lits_list numbers of samples is:',len(lits_ct_list))
    # print('The CT dataset ircadb_list numbers of samples is:', len(ircadb_ct_list))
    print('The CT dataset total numbers of samples is :',len(data_ct_list) )

    data_num=len(data_ct_list)

    random.shuffle(data_ct_list)

    assert valid_rate < 1.0
    train_name_list = data_ct_list[0:int(data_num * (1 - valid_rate))]
    val_name_list = data_ct_list[
                    int(data_num * (1 - valid_rate)):int(data_num * ((1 - valid_rate) + valid_rate))]

    write_name_list(train_name_list, "train_path_list.txt")
    write_name_list(val_name_list, "test_path_list.txt")

'''
lits:
0-46_ct.png
0-46_tumor.png

3dircadb:
ct-0.png
tumor-0.png
'''
def write_name_list(name_list, file_name):
    data_path=r'/home/data/WB_/LiverSegProject/data/LITS5'
    f = open(join(data_path, file_name), 'w')  # 在此路径下创建train_path_list.txt文件
    for name in name_list:
        ct_path =  name   #
        seg_path = name.replace('image','mask').replace('ct','liver')
        f.write(ct_path + ' ' + seg_path + "\n")
    f.close()

write_train_val_name_list()