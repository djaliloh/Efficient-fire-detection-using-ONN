import os
import torch.optim as optim
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.transforms as T
from model import SelfONN_1, FireDetectionCNN, SelfONN_4, Selfire
from loader import *
from tqdm import tqdm
from torchsummary import summary
from copy import deepcopy

dataworkdir = "./data"
k = 5 # CV fold number
inputchanel = 3
classnum = 2
qorder = 3 
optim_fc = "Adam"
lr = 0.001
n_epochs = 20
train_batch = 32
test_batch = 32


# verify if CUDA is available 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# def collate_fn(batch):
#     images = []
#     labels = []
#     #a, b = batch[0][0].shape[-2:]
#     resize_func = T.Resize(size=(224,224))
#     #resize_func = T.Resize(size=(int(b * 0.8), int(a * 0.8)))
#     to_tensor = T.ToTensor()
#     to_pil = T.ToPILImage()

#     for img, lab in batch:
#         labels.append(lab)
#         images.append(to_tensor(resize_func(to_pil(img))))
#     images = torch.stack(images)
#     labels = torch.stack(labels)
#     return images, labels
# transformer
transform = transforms.Compose([transforms.Resize((224, 224)),  transforms.ToTensor()])



# modelcnn = FireDetectionCNN(input_size=(3, 256, 256), num_classes=2)
# summary(modelcnn.to(device), (3, 256, 256))
onnmodel = Selfire(inputchanel, classnum, qorder)
summary(Selfire(inputchanel, classnum, qorder).to(device), (3, 224, 224))

# from torchsummary import summary
# summary(onnmodel, input_size=(3, 224, 224), device=device.type)
# exit()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(onnmodel.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False) 
# if optim_fc == 'Adam':  
#     optimizer = optim.Adam(onnmodel.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False) 
# elif optim_fc == 'SGD': 
#     optimizer = optim.SGD(onnmodel.parameters(), lr=lr, momentum=0.9, dampening=0, weight_decay=0.0001, nesterov=False)  

# output = torch.empty((2,3))
# output = torch.zeros((224, 224, 3))   # Create 3D tensor 
# output = output.unsqueeze(0)

# for m in onnmodel.children():
#     # print(m)
#     # exit()
#     output = m(output)
#     print("Heeerrrreee", m, output.shape)

# see where error occure in the network
# def print_sizes(model, input_tensor=[224]):
#     output = input_tensor
#     for m in model.children():
#         output = m(output)
#         print(m, output.shape)
#     return output

# print_sizes(onnmodel)




class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def run_epoch(model, train_dataloader, val_dataloader, criterion, optimizer):

    model.to(device)
    eval_metrics = AverageMeter()
    train_metrics = AverageMeter()

    # Set the model to train mode
    model.train()

    # Loop over the training data
    for X_batch, y_batch in train_dataloader:
        # X_batch = X_batch.permute(0,3,1,2) # for CNN
        # print(X_batch.shape)

        # Forward pass
        X_batch = X_batch.to(device) 
        y_batch = y_batch.to(device)
        # print(X_batch.shape)

        # zero the parameter gradients
        optimizer.zero_grad()
        y_pred = model(X_batch)

        # Compute the loss
        loss = criterion(y_pred, y_batch).to(device)

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

        acc = (y_batch == y_pred.argmax(dim=1)).float().sum() / len(y_batch)
        #print(acc)
        train_metrics.update(acc)

    # Set the model to eval mode
    with torch.no_grad():
        model.eval()
        # Loop over the validation data
        for X_batch, y_batch in val_dataloader:

            X_batch = X_batch.to(device) 
            y_batch = y_batch.to(device)
            # Forward pass
            y_pred = model(X_batch)

            # Compute the loss
            loss = criterion(y_pred, y_batch).to(device)

            # Update the evaluation metrics
            correct_val = (y_batch == y_pred.argmax(dim=1)).float().sum()
            eval_metrics.update(correct_val / len(y_batch))
    # Return the evaluation metrics 
    return model, train_metrics ,eval_metrics

models = [deepcopy(onnmodel)  for _ in range(k)]
optimizers = [optim.Adam(m.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False) for m in models]

for fold in tqdm(range(k)):
    #model = Selfire(inputchanel, classnum, qorder)
    #model = model.to(device)
    ep_train= AverageMeter()
    ep_val = AverageMeter()
    for epoch in range(n_epochs):
        # print(os.path.join(dataworkdir, str(fold), "train"))
        # Create the dataset
        train_dataset = FireDataset(os.path.join(dataworkdir, str(fold), "train"), transform=transform)
        # load the data
        train_dataloader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True)

        test_dataset = FireDataset(os.path.join(dataworkdir, str(fold), "test"), transform=transform)
        # load the data
        test_dataloader = DataLoader(test_dataset, batch_size=test_batch, shuffle=False)

        model , train_acc, val_acc = run_epoch(model=models[fold],train_dataloader=train_dataloader, val_dataloader=test_dataloader, criterion=criterion, optimizer=optimizers[fold])
        models[fold] = model
        ep_train.update(train_acc.avg)
        ep_val.update(val_acc.avg)
        print(f"Epoch {epoch} [ train_acc:{train_acc.avg.item():.3f} | val_acc:{val_acc.avg.item():.3f}]")
    print(f"Fold {fold} [ train_acc:{ep_train.avg.item():.3f} | val_acc:{ep_val.avg.item():.3f}]")
