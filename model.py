# PyTorch
import torch
from torchvision import models
from torch import cuda 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
# warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from fastONN.SelfONN import SelfONN2d

# channel attention module
class SELFCAM(nn.Module):
    def __init__(self, inchannels, q_order):
        super(SELFCAM, self).__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = SelfONN2d(in_channels=inchannels, out_channels=2, kernel_size=1, q=q_order,mode='fast')
        # self.relu = nn.ReLU() #nn.Tanh()
        self.tanh = nn.Tanh()
        self.fc2 = SelfONN2d(in_channels=2, out_channels=inchannels, kernel_size=1, q=q_order,mode='fast')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.avg_pooling(x)
        # w = self.relu(self.fc1(w))
        w = self.tanh(self.fc1(w))
        w = self.sigmoid(self.fc2(w))
        x = w * x

        return x



def Selfire(input_ch, class_num, q_order): 
    model = torch.nn.Sequential( 
        # Feature extraction layers  
        # 1st layer (conv) 
        SelfONN2d(in_channels=input_ch,out_channels=32,kernel_size=5,stride=1,q=q_order,mode='fast'),
        torch.nn.Tanh(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        
        # 2nd layer (conv)
        SelfONN2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,q=q_order,mode='fast'),
        torch.nn.Tanh(),
        # torch.nn.MaxPool2d(kernel_size=2, stride=2),  

        # 3rd layer (conv)
        SelfONN2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,q=q_order,mode='fast'),
        torch.nn.Tanh(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),

        # 4th 
        SelfONN2d(in_channels=64,out_channels=64,kernel_size=5,stride=1,q=q_order,mode='fast'),
        torch.nn.Tanh(),
        # torch.nn.MaxPool2d(kernel_size=2, stride=2),   
      
        # 5th layer (conv)
        SelfONN2d(in_channels=64,out_channels=128,kernel_size=5,stride=1,q=q_order,mode='fast'),
        torch.nn.Tanh(),
        torch.nn.MaxPool2d(2, stride=2),

        # 6th layer (conv)
        SelfONN2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,q=q_order,mode='fast'),
        torch.nn.Tanh(),
        # torch.nn.MaxPool2d(kernel_size=2, stride=2), 

        # 7th layer (conv)
        SelfONN2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,q=q_order,mode='fast'),
        torch.nn.Tanh(),
        torch.nn.MaxPool2d(kernel_size=2), 

        # 6th layer (conv)
        SelfONN2d(in_channels=128,out_channels=512,kernel_size=3,stride=1,q=q_order,mode='fast'),
        torch.nn.Tanh(),
        # torch.nn.MaxPool2d(kernel_size=2, stride=2),  

        # 7th layer (conv)
        SelfONN2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,q=q_order,mode='fast'),
        torch.nn.Tanh(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2), 

        # channel attention 
        SELFCAM(inchannels=512, q_order=q_order),
    

        # Classification layers       
        # flatten 
        torch.nn.Flatten(),  
        # Output layer (MLP)  
        torch.nn.Linear(in_features=512, out_features=512),
        torch.nn.Tanh(),
        # torch.nn.LogSoftmax(dim=1),
        # torch.nn.Linear(in_features=40, out_features=40, bias=True),
        torch.nn.Linear(in_features=512, out_features=class_num), 
        # 2nd linear layer
        # torch.nn.Linear(in_features=8450, out_features=class_num, bias=True),  
        torch.nn.LogSoftmax(dim=1)
    ) 

    return model 