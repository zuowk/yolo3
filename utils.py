import torch 
import cv2
import numpy as np 
from torch import nn 

def transform_detection(x,input_dim,anchors,num_classes):
	box_lenght=5+num_classes
	num_anchors=len(anchors)
	h=x.shape[2]
	w=x.shape[3]

	x=x.view(-1,num_anchors*box_lenght,h*w)
	x=x.transpose(1,2).contiguous()
	x=x.view(-1,h*w*num_anchors,box_lenght)
	xx,yy=np.meshgrid(np.arange(w),np.arange(h))
	xx,yy=torch.FloatTensor(xx).view(-1,1),torch.FloatTensor(yy).view(-1,1)
	sacle_factor=int(input_dim/h)
	anchors=[(a[0]/sacle_factor,a[1]/sacle_factor) for a in anchors]
	anchors=torch.FloatTensor(anchors).cuda()

	anchors=anchors.repeat(h*w,1).unsqueeze(0)
	x[:,:,2:4]=torch.exp(x[:,:,2:4])
	x[:,:,2:4]*=anchors
	x[:,:,0:2]=torch.sigmoid(x[:,:,0:2])
	x[:,:,4]=torch.sigmoid(x[:,:,4])
	xy=torch.cat((xx,yy),1)
	xy=xy.repeat(1,3)
	xy=xy.view(-1,2).unsqueeze(0)
	xy=xy.cuda()
	x[:,:,0:2]+=xy
	x[:,:,:4]*=sacle_factor
	x[:,:,5:5+num_classes]=torch.sigmoid(x[:,:,5:5+num_classes])


	return x
def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):

    
    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    
    #Add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
    
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    prediction[:,:,:4] *= stride
    
    return prediction



def iou(box1,box2):
	x1=box1[:,0]
	x2=box1[:,1]
	x3=box1[:,2]
	x4=box1[:,3]
	y1=box2[:,0]
	y2=box2[:,1]
	y3=box2[:,2]
	y4=box2[:,3]
	s1=(x3-x1+1)*(x4-x2+1)
	s2=(y3-y1+1)*(y4-y2+1)
	s=(torch.clamp((torch.min(x3,y3)-torch.max(x1,y1)+1),min=0)*torch.clamp((torch.min(x4,y4)-torch.max(x2,y2)+1),min=0))
	return s/(s1+s2-s)

def unique(x):
	y=x.cpu()
	y=y.numpy()
	y=np.unique(y)
	y=torch.FloatTensor(y)
	y=y.cuda()
	return y 


def nms(detections,num_classes,threshold1,threshold2):
	write=0
	x1=detections[:,:,0]-detections[:,:,2]/2
	y1=detections[:,:,1]-detections[:,:,3]/2
	x2=detections[:,:,0]+detections[:,:,2]/2
	y2=detections[:,:,1]+detections[:,:,3]/2
	detections[:,:,0:4]=torch.cat((x1.unsqueeze(2),y1.unsqueeze(2),x2.unsqueeze(2),y2.unsqueeze(2)),2)
	print(detections.shape)
	c1,c2=torch.max(detections[:,:,5:5+num_classes],2)
	c1,c2=c1.float().unsqueeze(2),c2.float().unsqueeze(2)
	detections=torch.cat((detections[:,:,:5],c1,c2),2)

	detections*=((detections[:,:,4]>threshold1).float().unsqueeze(2))
	for index in range(detections.shape[0]):
		detection=detections[index,:,:]
		nonzeros=torch.nonzero(detection[:,4]).squeeze()
		detection=detection[nonzeros,:].view(-1,7)
		classes=unique(detection[:,-1])
		for j in classes:
			jj=(detection[:,-1]==j).float().squeeze()
			jj=torch.nonzero(jj).squeeze()
			detection_j=detection[jj,:].view(-1,7)
			jjj=torch.sort(detection_j[:,4],descending=True)[1].squeeze()
			detection_j=detection_j[jjj,:].view(-1,7)

			for k in range(detection_j.shape[0]):
				try:
					ious=iou(detection_j[k,0:4].unsqueeze(0),detection_j[k+1:,0:4])
				except IndexError:
					break
				except ValueError:
					break
				kk=(ious<threshold2).float().unsqueeze(1)
				detection_j[k+1:,:]*=kk
				nonzero_ind=torch.nonzero(detection_j[:,4]).squeeze()
				detection_j=detection_j[nonzero_ind,:].view(-1,7)

			ind=detection_j.new(detection_j.shape[0],1).fill_(index)
			ind=ind.float()
			x=torch.cat((ind,detection_j),1)



			if write:
				output=torch.cat((output,x))
			else:
				output=x 
				write=1
	return output


def pre_img(img,height):
	img1=np.full((height,height,3),128)
	sacle_factor=min(height/img.shape[0],height/img.shape[1])
	h1=int(sacle_factor*img.shape[0])
	w1=int(sacle_factor*img.shape[1])
	img2=cv2.resize(img,(w1,h1),interpolation = cv2.INTER_CUBIC)
	img1[(height-h1)//2:(height-h1)//2+h1,(height-w1)//2:(height-w1)//2+w1,:]=img2
	img1=img1/255.0
	img1=img1[:,:,::-1].transpose((2,0,1)).copy()
	img1=torch.FloatTensor(img1).cuda()
	img1=img1.unsqueeze(0)
	return img1

# #def load_classes():
# 	fp=open('D:\\torch\\YOLO_v3_tutorial_from_scratch\\data\\coco.names', "r")
# 	names = fp.read().split("\n")[-1]
# 	print(names)
# 	return names

def load_classes():
    fp = open('D:\\torch\\YOLO_v3_tutorial_from_scratch\\data\\coco.names', "r")
    names = fp.read().split("\n")[:-1]
    return names
