#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 15:29:22 2019

@author: xiankai
"""
import numpy as np
import json
import cv2
import math
from PIL import Image
import matplotlib.pyplot as plt
#from pycocotools.mask import encode, iou, area, decode, toBbox, merge
from my_CRF import  my_NMS, my_NMS_all
import os
import glob
from utils import *

files_path = 'New_first_raw/'#'/home/xiankai/shuabang/davis_res'
attention_path = '/home/xiankai/work/Co-attention/result/test/davis_iteration_conf_davis_un/Results'#'/media/xiankai/Data/segmentation/AGS/test/reults/davis'#'/home/xiankai/shuabang/Results-35'#'/home/xiankai/work/DAVIS-2016/Results/Segmentations/480p/COS_78.3/' #'/home/xiankai/PycharmProjects/PSP/Davis-2017'
rgb_path = '/media/xiankai/Data/segmentation/DAVIS-un/DAVIS-un-test/JPEGImages/480p'
rgb_single_path = '/home/xiankai/shuabang/images-first-cha-cvpr2020' # this is new for each method
json_path = '/home/xiankai/shuabang/json-first-cha-cvpr2020' # this is new for each method
flow_path = '/home/xiankai/shuabang/davis-flow'
if not os.path.exists(json_path):
    os.mkdir(json_path)
if not os.path.exists(rgb_single_path):
    os.mkdir(rgb_single_path)
#ann_fn = '/media/xiankai/Data/segmentation/DAVIS/Annotations_unsupervised/480p/bike-packing/00000.png'
#ann = np.array(Image.open(ann_fn))
#ids = np.unique(ann)
#ids = [id for id in ids if id != 0]    
files = np.sort(os.listdir(attention_path))
img_files=[]
for i in range(0,len(files)):
    
    img_files = img_files+ glob.glob(files_path + '/*.jpg')
    
for i in range(0,len(files)): #
    file_name = files[i]#'dog-competition'#'hurdles'#'jet-ski'#'cat'#''mantaray'#basketball-game'#'horse-race'#'butterfly'# #'car-competition'#'kids-robot'#
    sub_path = os.path.join(files_path,file_name)
    sub_rgb_path = os.path.join(rgb_path,file_name)
    sub_flow_path = os.path.join(flow_path,file_name)
    f = open(os.path.join(sub_path,"00000.json"))
    bboxs = json.load(f)
    ###bbox = my_NMS(bboxs)
    #print(file_name)
    attention_1 = os.path.join(attention_path,file_name)
    #
    heat_map = cv2.imread(os.path.join(attention_1,"00000.png"),cv2.IMREAD_GRAYSCALE)
    heat_map = 1/(1+np.exp(-0.1*((heat_map).astype(float)-0.02)))
    
    #attention_map[attention_map<10]=0
    rgb_im = cv2.imread(os.path.join(sub_rgb_path,"00000.jpg"),  cv2.IMREAD_COLOR)
    if not os.path.exists(os.path.join(rgb_single_path,file_name)):
        os.mkdir(os.path.join(rgb_single_path,file_name))
    cv2.imwrite(os.path.join(rgb_single_path,file_name+"/00000.jpg"),rgb_im)
    print('path:',os.path.join(sub_rgb_path,"00000.jpg"))
    attention_map =  my_CRF(heat_map,rgb_im)
    plt.figure(0)
    plt.imshow(attention_map)
    my_zeros = np.zeros_like(attention_map)
    new_bboxs=[]
    bboxs = my_NMS_all(bboxs,0.35)
    for item in bboxs:
        bbox = np.floor(item["bbox"])
        if bbox[2]*bbox[3]<200:
            continue
        score = item["score"]
        my_zeros1 = my_zeros.copy()
        my_zeros1[int(bbox[1]):int(bbox[1])+int(bbox[3]),int(bbox[0]):int(bbox[0])+int(bbox[2])]=1
        my_iou = db_eval_iou(my_zeros1.astype('uint8'),attention_map) 
        #visualize_tracking_result(rgb_im, bbox, 1),print(file_name,my_iou,score, bbox[2]*bbox[3])
        #iou(encode(np.asfortranarray(my_zeros1.astype('uint8'))),encode(np.asfortranarray(attention_map.astype('uint8'))),np.array([0], np.uint8))
        if my_iou>0.09 and score>0.56 or score>0.89: #or bbox[2]*bbox[3]< 2000 and my_iou>0 0.89
            ## NMS
            #print(my_iou,score)
            empty_prop = dict()
            empty_prop["bbox"] = item["bbox"]
            empty_prop["score"] = item["score"]
            new_bboxs.append(empty_prop)
        elif file_name=='cat' and  score>0.84:
            empty_prop = dict()
            empty_prop["bbox"] = item["bbox"]
            empty_prop["score"] = item["score"]
            new_bboxs.append(empty_prop)  
            #visualize_tracking_result(rgb_im, item["bbox"], 1),print(file_name,my_iou,score)
            
    if len(new_bboxs)==0:
        for item in bboxs:
            bbox = np.floor(item["bbox"])
            score = item["score"]
            my_zeros1 = my_zeros.copy()
            my_zeros1[int(bbox[1]):int(bbox[1])+int(bbox[3]),int(bbox[0]):int(bbox[0])+int(bbox[2])]=1
            my_iou = db_eval_iou(my_zeros1.astype('uint8'), heat_map.astype('uint8')) 
            
            #iou(encode(np.asfortranarray(my_zeros1.astype('uint8'))),encode(np.asfortranarray(attention_map.astype('uint8'))),np.array([0], np.uint8))
            if my_iou>0.08 and score>0.56 or score>0.7 : #0.56 for small target
                ## NMS
                print(my_iou,score)
                empty_prop = dict()
                empty_prop["bbox"] = item["bbox"]
                empty_prop["score"] = item["score"]
                new_bboxs.append(empty_prop)
                #visualize_tracking_result(rgb_im, item["bbox"], 1),print(file_name,my_iou,score)
    new_bboxs = my_NMS(new_bboxs)
    #new_bboxs = my_NMS_all(new_bboxs,0.3)
    new_new_bboxs = []
    cal = 1 # omit the background 
    for kk in range(0,len(new_bboxs)):
        item = new_bboxs[kk]
        if kk>19:
            continue
        bbox = item["bbox"]
        temp={ "bbox" : bbox,
              "score" : item["score"],
              "id" : cal
              }
        cal = cal+1
        new_new_bboxs.append(temp)
        visualize_tracking_result(rgb_im, bbox, 1),print('score:',item["score"])
    
    final_name = str(file_name)+"/00000.json"
    if not os.path.exists(os.path.join(json_path,file_name)):
        os.mkdir(os.path.join(json_path,file_name))
    final_path = os.path.join(json_path,final_name)
    final_data=open(final_path,"w")
    json.dump(new_new_bboxs,final_data,sort_keys=True, indent=4)
    final_data.close()
    #with open(final_path,'r') as f:
    #    result = json.load(f)
#attention_map = cv2.imread('/home/xiankai/PycharmProjects/PSP/Davis-2017/bike-packing/00000.png',cv2.IMREAD_GRAYSCALE)
#attention_map[attention_map>10]=255
#attention_map[attention_map<=10]=0
#attention_map = attention_map/255
#my_zeros = np.zeros_like(attention_map)
#for item in bboxs:
#    bbox = np.floor(item['bbox'])
#    my_zeros1 = my_zeros.copy()
#    my_zeros1[int(bbox[1]):int(bbox[1])+int(bbox[3]),int(bbox[0]):int(bbox[0])+int(bbox[2])]=1
#    my_iou = db_eval_iou(my_zeros1.astype('uint8'),attention_map.astype('uint8')) 
#    #iou(encode(np.asfortranarray(my_zeros1.astype('uint8'))),encode(np.asfortranarray(attention_map.astype('uint8'))),np.array([0], np.uint8))
#    if my_iou>0:
#        visualize_tracking_result(my_zeros1, bbox, 1)
