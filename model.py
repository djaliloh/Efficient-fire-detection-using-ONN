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
# Data science tools
import numpy as np
from selfonnqu import SelfONNLayer
from SelfONN import SelfONN2d



import torch
import torch.nn as nn

class FireDetectionCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FireDetectionCNN, self).__init__()

        # Feature extraction layers
        self.conv1 = nn.Conv2d(in_channels=input_size[0], out_channels=32, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Classification layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=115200, out_features=128)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Feature extraction
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # Classification
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x

# Create the model
# model = FireDetectionCNN(input_size=(3, 256, 256), num_classes=2)



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

        # 8th layer (conv)
        SelfONN2d(in_channels=512,out_channels=512,kernel_size=2,stride=1,q=q_order,mode='fast'),
        torch.nn.Tanh(),
        # torch.nn.MaxPool2d(kernel_size=2, stride=2),

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



##################################### EFDNet #####################################
# Basic_layer
# "basic_layer" of this module takes the feature output from all the previous layers as the input
def basic_layer(in_planes, out_planes, kernel, stride=1, padding=0):
    layer = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel, stride, padding),
        nn.BatchNorm2d(out_planes, eps=1e-3),
        nn.ReLU(inplace=True)
    )
    
    return layer


# MFE MODULE    
class inception(nn.Module):
    def __init__(self, in_planes, out_planes1, out_planes2_1, out_planes2_2, out_planes3_1, out_planes3_2, out_planes4_1):
        super(inception, self).__init__()
        self.planes = out_planes1 + out_planes2_2 + out_planes3_2 + out_planes4_1
        self.mp = self.planes // 16

        self.branch1 = basic_layer(in_planes, out_planes1, 1)
        
        self.branch2 = nn.Sequential(
            basic_layer(in_planes, out_planes2_1, 1),
            basic_layer(out_planes2_1, out_planes2_2, 3, padding=1)
        )
        
        self.branch3 = nn.Sequential(
            basic_layer(in_planes, out_planes3_1, 1),
            basic_layer(out_planes3_1, out_planes3_2, 5, padding=2)
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            basic_layer(in_planes, out_planes4_1, 1)
        )

        
    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        x = torch.cat((out1, out2, out3, out4), 1)

        return x

# channel attention module
class CAM(nn.Module):
    def __init__(self, inchannels):
        super(CAM, self).__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Conv2d(in_channels=inchannels, out_channels=2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels=2, out_channels=inchannels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.avg_pooling(x)
        w = self.relu(self.fc1(w))
        w = self.sigmoid(self.fc2(w))
        x = w * x

        return x


class _denselayer(nn.Module):
    
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate): 
        super(_denselayer, self).__init__()
        self.drop_rate = drop_rate
        self.planes = num_input_features + growth_rate
        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, bn_size * growth_rate,
                      kernel_size=1, stride=1, bias=False)
        )
        self.layer2 = nn.Sequential(
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3,
                      stride=1, padding=1, bias=False)
        )
        
    def forward(self, x):
        new_features = self.layer1(x)
        new_features = self.layer2(new_features)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        out = torch.cat([x, new_features], 1)
        
        return out
        

def _DenseBlock(num_layers, num_input_features, growth_rate, bn_size, drop_rate):
    
    layer = [ ]
    for i in range(num_layers):
        layer.append(_denselayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate))
        
    return nn.Sequential(*layer)
        

class _Transition(nn.Module):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.trans = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1,
                      stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        
    def forward(self, x):
        x = self.trans(x)
        return x
        
        
class densenet(nn.Module):
    
    def __init__(self, growth_rate=32, block_config=(6, 8, 12, 24),
                 num_init_features=480, bn_size=4, drop_rate=0, num_classes=2):
        super(densenet, self).__init__()
        self.dense, num_features = self._make_layer(growth_rate, block_config, num_init_features, bn_size, drop_rate)
        
        # The last bn layer
        self.bn = nn.BatchNorm2d(num_features)
        
        # Fully connected layer
        self.classifier = nn.Linear(num_features, num_classes)
        
        self.gloabalpool = nn.AvgPool2d(7, stride=1)
        
    def _make_layer(self, growth_rate, block_config, num_init_features, bn_size, drop_rate):
        layer = [ ]
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            layer.append(_DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                     bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate))
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                layer.append(CAM(inchannels=num_features))
                layer.append(_Transition(num_input_features=num_features, num_output_features=num_features // 2))
                num_features = num_features // 2
        return nn.Sequential(*layer), num_features
        
        
    def forward(self, x):
        x = self.dense(x)
        x = self.bn(x)
        x = self.gloabalpool(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        
        return out
        

class FDnet(nn.Module):
    
    def __init__(self, in_planes, num_class, block_config=(4, 8, 10, 12)):
        super(FDnet, self).__init__()
        
        #level one
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_planes, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        ##level two
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        #Three-layer inception structure
        self.inception1 = inception(64, 16, 8, 32, 8, 32, 16)
        self.inception2 = inception(96, 16, 16, 48, 16, 48, 16)
        self.inception3 = inception(128, 32, 32, 96, 32, 96, 32) #Output feature map 480
        
        # densenet module
        self.dense = densenet(num_init_features=256, growth_rate=12, block_config=block_config, num_classes=num_class)


  

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        out = self.dense(x)
   
        return out
        
        
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out') 
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, mean=0, std=0.01)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)  

# def count_neurons(model):
#     return sum([])
# print(len(classes))
# net = FDnet(3, 2, block_config=(4, 6, 8, 10)).cuda() # the original
# net.apply(weights_init)







def SelfO(input_ch, class_num, q_order): 
    model = torch.nn.Sequential(   
        # 1st layer (conv) 
        SelfONN2d(in_channels=input_ch,out_channels=90,kernel_size=3,stride=1,padding=2,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        # torch.nn.Tanh(),
        # 2nd layer (conv)
        SelfONN2d(in_channels=90,out_channels=100,kernel_size=3,stride=2,padding=2,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),  
        # torch.nn.Tanh(),
        # 3rd 
        SelfONN2d(in_channels=100,out_channels=50,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),  
        # torch.nn.Tanh(), 
        # 4th layer (conv)
        SelfONN2d(in_channels=50,out_channels=50,kernel_size=3,stride=2,padding=5,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        # torch.nn.MaxPool2d(2),  
        # torch.nn.Tanh(), 
        # # 5th layer (conv) 
        SelfONN2d(in_channels=50,out_channels=60,kernel_size=3,stride=2,padding=2,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),  
        # torch.nn.Tanh(), 
        # 6th layer (conv) 
        SelfONN2d(in_channels=60,out_channels=10,kernel_size=3,stride=4,padding=2,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        # torch.nn.MaxPool2d(2),  
        # torch.nn.Tanh(), 
        # flatten 
        torch.nn.Flatten(),  
        # Output layer (MLP)  
        torch.nn.Linear(in_features=40, out_features=40, bias=True),
        # torch.nn.LogSoftmax(dim=1),
        # torch.nn.Linear(in_features=40, out_features=40, bias=True),
        torch.nn.Linear(in_features=40, out_features=class_num, bias=True), 
        # 2nd linear layer
        # torch.nn.Linear(in_features=8450, out_features=class_num, bias=True),
        # # 3rd linear layer
        # torch.nn.Linear(in_features=8450, out_features=class_num, bias=True),  
        torch.nn.LogSoftmax(dim=1)
    ) 

    return model 




class SimpleCNN(nn.Module):
    def __init__(self):
        # ancestor constructor call
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=90, kernel_size=3, padding=2)
        self.conv2 = nn.Conv2d(in_channels=90, out_channels=100, kernel_size=3, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=100, out_channels=50, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=3, stride=2, padding=5)
        self.conv5 = nn.Conv2d(in_channels=50, out_channels=60, kernel_size=3, stride=2, padding=2)
        self.conv6 = nn.Conv2d(in_channels=60, out_channels=10, kernel_size=3, stride=4, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=8, padding=-2, stride=2)
        self.fc_out = nn.Linear(in_features= 40, out_features=2)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(in_features= 40, out_features=40) # !!!
    def forward(self, x):
        x = self.pool(self.conv1(x)) 
        x = self.pool1(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.conv4(x)
        x = self.pool(self.conv5(x))
        x = self.conv6(x)
        x = x.view(-1, 40) # !!!
        x = self.fc(x)
        x = self.dropout(self.fc(x))
        x = self.fc(x)
        x = self.fc_out(x)
        return x






def reset_function_generic(m):
    if hasattr(m,'reset_parameters') or hasattr(m,'reset_parameters_like_torch'): 
        # print(m) 
        if isinstance(m, SelfONNLayer):
            m.reset_parameters_like_torch() 
        else:
            m.reset_parameters()

# class SqueezeLayer(nn.Module):
    
#     def forward(self,x):
#         x = x.squeeze(2)
#         x = x.squeeze(2)
#         return x 



class FireDetectionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        # x = x.view(-1, 3, 8, 8)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 8 * 8)
        print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    


def SelfONN_1(input_ch, class_num, q_order): 
    model = torch.nn.Sequential(   
        # 1st layer (conv) 
        SelfONNLayer(in_channels=input_ch,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 2nd layer (conv)
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        #torch.nn.MaxPool2d(2),  
        torch.nn.Tanh(), 
        # flatten 
        torch.nn.Flatten(),  
        # Output layer (MLP)  
        torch.nn.Linear(in_features=896, out_features=class_num, bias=True),  
        torch.nn.LogSoftmax(dim=1)
    ) 
    #
    # reset_fn = reset_function_generic 
    # model.apply(reset_fn) 
    return model 


def SelfONN_4(input_ch, class_num, q_order): 
    model = torch.nn.Sequential(   
        # 1st layer (conv) 
        SelfONNLayer(in_channels=input_ch,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 2nd layer (conv)
        SelfONNLayer(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # # 3rd layer (conv) 
        # SelfONNLayer(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        # torch.nn.MaxPool2d(2),
        # torch.nn.Tanh(),
        # 4th layer (conv)
        # SelfONNLayer(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        # # torch.nn.MaxPool2d(2),
        # torch.nn.Tanh(), 
        # # 5th layer (conv) 
        # SelfONNLayer(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        # # torch.nn.MaxPool2d(2),
        # torch.nn.Tanh(),
        # # 6th layer (conv)
        # SelfONNLayer(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        # # torch.nn.MaxPool2d(3),  
        # torch.nn.Tanh(), 
        # flatten
        # 
        torch.nn.Flatten(),  
        # Output layer (MLP)  
        torch.nn.Linear(in_features=1792, out_features=class_num, bias=True),  
        torch.nn.LogSoftmax(dim=1)
    ) 
    #
    # reset_fn = reset_function_generic 
    # model.apply(reset_fn) 
    return model 


# def SelfONN_5(input_ch, class_num, q_order): 
#     model = torch.nn.Sequential(   
#         # 1st layer (conv) 
#         SelfONNLayer(in_channels=input_ch,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
#         torch.nn.MaxPool2d(2),
#         torch.nn.Tanh(),
#         # 2nd layer (conv)
#         SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
#         torch.nn.Tanh(), 
#         # 3rd layer (conv) 
#         SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
#         torch.nn.MaxPool2d(2),
#         torch.nn.Tanh(),
#         # 4th layer (conv)
#         SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
#         torch.nn.Tanh(), 
#         # 5th layer (conv) 
#         SelfONNLayer(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
#         torch.nn.MaxPool2d(2),
#         torch.nn.Tanh(),
#         # 6th layer (conv)
#         SelfONNLayer(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
#         torch.nn.Tanh(), 
#         # 7th layer (conv) 
#         SelfONNLayer(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
#         torch.nn.MaxPool2d(2),
#         torch.nn.Tanh(),
#         # 8th layer (conv)
#         SelfONNLayer(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
#         torch.nn.MaxPool2d(3),  
#         torch.nn.Tanh(), 
#         # flatten 
#         torch.nn.Flatten(),  
#         # Output layer (MLP)  
#         torch.nn.Linear(in_features=512, out_features=class_num, bias=True),  
#         torch.nn.LogSoftmax(dim=1)
#     ) 
#     #
#     reset_fn = reset_function_generic 
#     model.apply(reset_fn) 
#     return model 