import torch 
from torch import nn 
from utils import *

def read_cfg():
	with open('cfg/yolov3.cfg') as fo:
		lines=fo.read().split('\n')
	lines=[x for x in lines if len(x)>0]
	lines=[x for x in lines if x[0]!='#']
	blocks=[]
	block={}
	for x in lines:
		if x[0]=='[':
			if block:
				blocks.append(block)
				block={}
			block['type']=x[1:-1]
		else:
			i,j=x.split('=')
			block[i.rstrip()]=j.lstrip()
	blocks.append(block)
	return  blocks

class Emptylayer(nn.Module):
	def __init__(self):
		super().__init__()

class Yololayer(nn.Module):
	def __init__(self,anchors):
		super().__init__()
		self.anchors=anchors


def create_net(blocks):
	net_info=blocks[0]
	module_list=nn.ModuleList()
	prefilters=3
	filters_list=[]
	for index,block in enumerate(blocks[1:],0):
		module=nn.Sequential()
		if block['type']=='convolutional':
			try:
				batch_norm=int(block['batch_normalize'])
				bias=False
			except:
				batch_norm=0
				bias=True 
			activation=block['activation']
			kernel_size=int(block['size'])
			filters=int(block['filters'])
			stride=int(block['stride'])
			try:
				padding=int(block['pad'])
			except:
				padding=0
			if padding:
				pad_size=int((kernel_size-1)/2)
			else:
				pad_size=1
			conv=nn.Conv2d(prefilters,filters,kernel_size=kernel_size,stride=stride,padding=pad_size,bias=bias)
			module.add_module('conv_{}'.format(index),conv)
			if batch_norm:
				module.add_module('batch_norm_{}'.format(index),nn.BatchNorm2d(filters))
			if activation=='leaky':
				module.add_module('leaky_{}'.format(index),nn.LeakyReLU(0.1,inplace=True))
		elif block['type']=='upsample':
			scale_factor=int(block['stride'])
			module.add_module('upsample_{}'.format(index),nn.Upsample(scale_factor=scale_factor))
		elif block['type']=='maxpool':
			stride=int(block['stride'])
			kernel_size=int(block['size'])
			module.add_module('maxpool_{}'.format(index),nn.MaxPool2d(stride=stride,kernel_size=kernel_size))
		elif block['type']=='shortcut':
			from_index=int(block['from'])
			module.add_module('shortcut_{}'.format(index),Emptylayer())
		elif block['type']=='route':
			module.add_module('route_{}'.format(index),Emptylayer())
			block['layers']=block['layers'].split(',')
			layers=block['layers']

			layers=[int(x) for x in layers]
			if len(layers)==1:
				l1=layers[0]
				filters=filters_list[index+l1]
			else:
				l1,l2=layers
				filters=filters_list[index+l1]+filters_list[l2]
		elif block['type']=='yolo':
			mask=block['mask']
			mask=mask.split(',')
			mask=[int(x) for x in mask]
			anchors=block['anchors']
			anchors=anchors.split(',')
			anchors=[int(x) for x in anchors]
			anchors=[(anchors[i],anchors[i+1]) for i in range(0,len(anchors),2)]
			anchors=[anchors[i] for i in mask]
			module.add_module('yolo_{}'.format(index),Yololayer(anchors))
		else:
			print('error,no such a layer')
		module_list.append(module)
		prefilters=filters
		filters_list.append(filters)
	return net_info,module_list


class Darknet(nn.Module):
	def __init__(self):
		super().__init__()
		self.blocks=read_cfg()
		self.net_info,self.module_list=create_net(self.blocks)
	def forward(self,x):
		write=0
		outputs={}
		for index,block in enumerate(self.blocks[1:],0):
			if block['type']=='convolutional' or block['type']=='upsample' or block['type']=='maxpool':

				x=self.module_list[index](x)

			elif block['type']=='shortcut':
				from_index=int(block['from'])
				x=outputs[index-1]+outputs[index+from_index]
			elif block['type']=='route':
				layers=block['layers']
				if len(layers)==1:
					l1=int(layers[0])
					x=outputs[index+l1]
				else:
					l1,l2=layers
					l1=int(l1)
					l2=int(l2)
					x=torch.cat((outputs[index+l1],outputs[l2]),1)
			elif block['type']=='yolo':
				input_dim=int(self.net_info['width'])
				num_classes=int(block['classes'])
				anchors=self.module_list[index][0].anchors

				x=transform_detection(x,input_dim,anchors,num_classes)

				if write:
					detections=torch.cat((detections,x),1)
				else:
					detections=x
					write=1

			else:
				print('error,no such a layer')
			outputs[index]=x
		return detections 
	def load_weights(self):
        #Open the weights file
		fp = open('D:\\yolo\\YOLO_v3_tutorial_from_scratch\\yolov3.weights', "rb")
    
        #The first 5 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4,5. Images seen by the network (during training)
		header = np.fromfile(fp, dtype = np.int32, count = 5)
		self.header = torch.from_numpy(header)
		self.seen = self.header[3]   
        
		weights = np.fromfile(fp, dtype = np.float32)
        
		ptr = 0
		for i in range(len(self.module_list)):
			module_type = self.blocks[i + 1]["type"]
    
            #If module_type is convolutional load weights
            #Otherwise ignore.
            
			if module_type == "convolutional":
				model = self.module_list[i]
				try:
					batch_normalize = int(self.blocks[i+1]["batch_normalize"])
				except:
					batch_normalize = 0
            
				conv = model[0]
                
                
				if (batch_normalize):
					bn = model[1]
        
                    #Get the number of weights of Batch Norm Layer
					num_bn_biases = bn.bias.numel()
        
                    #Load the weights
					bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
					ptr += num_bn_biases
        
					bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
					ptr  += num_bn_biases
        
					bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
					ptr  += num_bn_biases
					bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
					ptr  += num_bn_biases
        
                    #Cast the loaded weights into dims of model weights. 
					bn_biases = bn_biases.view_as(bn.bias.data)
					bn_weights = bn_weights.view_as(bn.weight.data)
					bn_running_mean = bn_running_mean.view_as(bn.running_mean)
					bn_running_var = bn_running_var.view_as(bn.running_var)
        
                    #Copy the data to model
					bn.bias.data.copy_(bn_biases)
					bn.weight.data.copy_(bn_weights)
					bn.running_mean.copy_(bn_running_mean)
					bn.running_var.copy_(bn_running_var)
                
				else:
                    #Number of biases
					num_biases = conv.bias.numel()
                
                    #Load the weights
					conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
					ptr = ptr + num_biases
                
                    #reshape the loaded weights according to the dims of the model weights
					conv_biases = conv_biases.view_as(conv.bias.data)
                
                    #Finally copy the data
					conv.bias.data.copy_(conv_biases)
                    
                #Let us load the weights for the Convolutional layers
				num_weights = conv.weight.numel()
                
                #Do the same as above for weights
				conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
				ptr = ptr + num_weights
                
				conv_weights = conv_weights.view_as(conv.weight.data)
				conv.weight.data.copy_(conv_weights)


if __name__=='__main__':
	x=torch.randn(1,3,416,416).cuda()
	net=Darknet()
	net=net.cuda()
	print(net(x).shape)

