#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:37:59 2019

@author: aaa
"""
import torch
from dataset import IrisDataset
from torch.utils.data import DataLoader 
import numpy as np
import matplotlib.pyplot as plt
from dataset import transform
import os
from opt import parse_args
from models import model_dict
from tqdm import tqdm
from utils import get_predictions
#%%

if __name__ == '__main__':
    
    args = parse_args()
   
    if args.model not in model_dict:
        print ("Model not found !!!")
        print ("valid models are:",list(model_dict.keys()))
        exit(1)

    #if args.useGPU:
        #device=torch.device("cuda")
    #else:
        #device=torch.device("cpu")
    device = torch.device("cpu")
        
    model = model_dict[args.model]
    model  = model.to(device)
    filename = args.load
    if not os.path.exists(filename):
        print("model path not found !!!")
        exit(1)
        
    model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
    model = model.to(device)
    model.eval()

    test_set = IrisDataset(filepath = 'Semantic_Segmentation_Dataset/',\
                                 split = 'test',transform = transform)
    
    testloader = DataLoader(test_set, batch_size = args.bs,
                             shuffle=False, num_workers=2)
    counter=0
    
    os.makedirs('test/labels/',exist_ok=True)
    os.makedirs('test/output/',exist_ok=True)
    os.makedirs('test/mask/',exist_ok=True)
    
    with torch.no_grad():
        for i, batchdata in tqdm(enumerate(testloader),total=len(testloader)):
            img,labels,index,x,y= batchdata
            data = img.to(device)       
            output = model(data)            
            predict = get_predictions(output)
            for j in range (len(index)):       
                np.save('test/labels/{}.npy'.format(index[j]),predict[j].cpu().numpy())
                try:
                    label_img = labels[j].cpu().numpy()  # 确保这一步返回一个有形状的数组
                    if len(label_img.shape) == 2:  # 如果是二维数组，表示灰度图像
                        plt.imsave(f'test/output/{index[j]}.jpg', label_img * 255, cmap='gray')
                    elif len(label_img.shape) == 3:  # 如果是三维数组，表示彩色图像
                        plt.imsave(f'test/output/{index[j]}.jpg', label_img * 255)
                except Exception as e:
                    print(f"保存文件{index[j]}时发生错误: {e}")

                pred_img = predict[j].cpu().numpy()/3.0
                inp = img[j].squeeze() * 0.5 + 0.5
                img_orig = np.clip(inp,0,1)
                img_orig = np.array(img_orig)
                combine = np.hstack([img_orig,pred_img])
                plt.imsave('test/mask/{}.jpg'.format(index[j]),combine)

    os.rename('test',args.save)
