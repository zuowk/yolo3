import time
import torch 
import torch.nn as nn
import numpy as np
import cv2 
from utils import *
import argparse
import os 
from model import Darknet
import pickle as pkl
import pandas as pd
import random

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det2", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 4)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    
    return parser.parse_args()
    
args = arg_parse()

img_name=args.images
batch_size=args.bs
det=args.det 
threshold1=args.confidence
threshold2=args.nms_thresh
height=int(args.reso )

classes=load_classes()
num_classes=len(classes)

try:
    img_names=[os.path.join(os.path.realpath('.'),img_name,x) for x in os.listdir(img_name)]
except NotADirectoryError:
    img_names=[os.path.join(os.path.realpath('.'),img_name)]
except FileNotFoundError:
    print('error,no such a file')

loaded_imgs=[cv2.imread(x) for x in img_names]
det_imgs=list(map(pre_img,loaded_imgs,[height for i in range(len(loaded_imgs))]))

left_over=0
if (len(loaded_imgs)%batch_size):
    left_over=1

num_batches=(len(loaded_imgs)//batch_size)+left_over

batches=[det_imgs[i*batch_size:min((i+1)*batch_size,len(loaded_imgs))] for i in range(num_batches)]

batches=[torch.cat(batch,0) for batch in batches]

net=Darknet()
net=net.cuda()
net.load_weights()
net.eval()

write=0

def put_rectangle(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2,[255,0,0], 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,[0,255,0], -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img

for i,batch in enumerate(batches,0):
    with torch.no_grad():
        output=net(batch)

    output=nms(output,num_classes,threshold1,threshold2)
    output[:,0]+=i*batch_size
    for ii,jj in enumerate(img_names[i*batch_size:min((i+1)*batch_size,len(img_names))]):
        im_id=ii+i*batch_size
        classes_det=[int(x[-1]) for x in output if int(x[0])==im_id]
        objs=[classes[x] for x in classes_det]
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
    if write:
    	outputs=torch.cat((outputs,output),0)
    else:
    	outputs=output
    	write=1


orgin_shape=[(x.shape[1],x.shape[0]) for x in loaded_imgs]
orgin_shape=torch.FloatTensor(orgin_shape).view(-1,2).cuda()
orgin_shape=orgin_shape.repeat(1,2)

orgin_shape=torch.index_select(orgin_shape,0,outputs[:,0].long())

scalers=torch.min(height/orgin_shape,1)[0].view(-1,1)


outputs[:,1:5]=outputs[:,1:5]-(height-orgin_shape[:,:]*scalers)/2 
outputs[:,1:5]=outputs[:,1:5]/scalers


list(map(lambda y:put_rectangle(y,loaded_imgs),outputs))



if not os.path.exists(det):
	os.makedirs(det)
if len(img_names)==1:
    det_names = pd.Series(img_names).apply(lambda x: "{}/det_{}".format(args.det,x.split("/")[-1]))
else:
    det_names = pd.Series(img_names).apply(lambda x: "{}/det_{}".format(args.det,x.split("\\")[-1]))


print(det_names)
list(map(cv2.imwrite, det_names, loaded_imgs))


torch.cuda.empty_cache()
    
