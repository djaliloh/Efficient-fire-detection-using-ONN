# PyTorch
import torch
from torchvision import models
from torch import cuda 
import torch.nn as nn
import torch.nn.functional as F
# warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
# Data science tools
import numpy as np
from selfonnqu import SelfONNLayer
from SelfONN import SelfONN2d

def Selfire(input_ch, class_num, q_order): 
    model = torch.nn.Sequential( 
        # Feature extraction layers  
        # 1st layer (conv) 
        SelfONN2d(in_channels=input_ch,out_channels=32,kernel_size=3,stride=1,q=q_order,mode='fast'),
        # torch.nn.ReLU(),
        torch.nn.Tanh(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        # torch.nn.Tanh(),
        # 2nd layer (conv)
        SelfONN2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,q=q_order,mode='fast'),
        # torch.nn.ReLU(),
        torch.nn.Tanh(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),  
        # torch.nn.Tanh(),
        # 3rd 
        SelfONN2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,q=q_order,mode='fast'),
        # torch.nn.ReLU(),
        torch.nn.Tanh(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),   
        # torch.nn.Tanh(),
        #  
        # 4th layer (conv)
        SelfONN2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,q=q_order,mode='fast'),
        torch.nn.Tanh(),
        torch.nn.MaxPool2d(2), 

        # 5th layer (conv)
        SelfONN2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,q=q_order,mode='fast'),
        torch.nn.Tanh(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2), 

        # 6th layer (conv)
        SelfONN2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,q=q_order,mode='fast'),
        torch.nn.Tanh(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),  
        

        # Classification layers       
        # flatten 
        torch.nn.Flatten(),  
        # Output layer (MLP)  
        torch.nn.Linear(in_features=256, out_features=256),
        torch.nn.Tanh(),
        # torch.nn.LogSoftmax(dim=1),
        # torch.nn.Linear(in_features=40, out_features=40, bias=True),
        torch.nn.Linear(in_features=256, out_features=class_num), 
        # 2nd linear layer
        # torch.nn.Linear(in_features=8450, out_features=class_num, bias=True),
        # # 3rd linear layer
        # torch.nn.Linear(in_features=8450, out_features=class_num, bias=True),  
        torch.nn.LogSoftmax(dim=1)
    ) 

    return model 




############# Average [ Loss:0.213 | Accuracy:0.947] the best net so far #############
def Selfire(input_ch, class_num, q_order): 
    model = torch.nn.Sequential( 
        # Feature extraction layers  
        # 1st layer (conv) 
        SelfONN2d(in_channels=input_ch,out_channels=32,kernel_size=3,stride=1,q=q_order,mode='fast'),
        torch.nn.Tanh(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        
        # 2nd layer (conv)
        SelfONN2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,q=q_order,mode='fast'),
        torch.nn.Tanh(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),  

        # 3rd 
        SelfONN2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,q=q_order,mode='fast'),
        torch.nn.Tanh(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),   
      
        # 4th layer (conv)
        SelfONN2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,q=q_order,mode='fast'),
        torch.nn.Tanh(),
        torch.nn.MaxPool2d(2), 

        # 5th layer (conv)
        SelfONN2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,q=q_order,mode='fast'),
        torch.nn.Tanh(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2), 

        # 6th layer (conv)
        SelfONN2d(in_channels=512,out_channels=1024,kernel_size=3,stride=1,q=q_order,mode='fast'),
        torch.nn.Tanh(),
        # torch.nn.MaxPool2d(kernel_size=2, stride=2),  

        # 7th layer (conv)
        SelfONN2d(in_channels=1024,out_channels=2048,kernel_size=3,stride=1,q=q_order,mode='fast'),
        torch.nn.Tanh(),
        

        # Classification layers       
        # flatten 
        torch.nn.Flatten(),  
        # Output layer (MLP) 
        # 1st linear layer 
        torch.nn.Linear(in_features=2048, out_features=2048),
        torch.nn.Tanh(),
        # 2nd linear layer
        torch.nn.Linear(in_features=2048, out_features=class_num), 
        torch.nn.LogSoftmax(dim=1)
    ) 

    return model 