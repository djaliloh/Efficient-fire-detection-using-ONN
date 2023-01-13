import os
import torch.optim as optim
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.transforms as T
from model import Selfire
from loader import *
from tqdm import tqdm
from torchsummary import summary
from model_utils import average

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


dataworkdir = "./data"
datainit = "./data_init"
k = 5 # CV fold number
inputchanel = 3
classnum = 2
qorder = 3 
optim_fc = "Adam"
lr = 0.0001
n_epochs = 10


# verify if CUDA is available 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

transform = transforms.Compose([transforms.Resize((224, 224)),  transforms.ToTensor()])


################################## print model summary ################################## 

onnmodel = Selfire(inputchanel, classnum, qorder)
summary(Selfire(inputchanel, classnum, qorder).to(device), (3, 224, 224))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(onnmodel.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False) 


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
    # Set the model to train mode
    model.to(device)
    model.train()
    eval_metrics = AverageMeter()
    train_metrics = AverageMeter()

    # Loop over the training data
    for X_batch, y_batch in train_dataloader:
        # X_batch = X_batch.permute(0,3,1,2) 
        # X_batch = X_batch.permute(0,2,3,1) 
    
        # Forward pass
        X_batch = X_batch.to(device) 
        y_batch = y_batch.to(device)
        # print(X_batch.shape)

        # zero the parameter gradients
        optimizer.zero_grad()
        y_pred = model(X_batch)

        # print(y_batch.shape)
        # Compute the loss
        loss = criterion(y_pred, y_batch).to(device)
        loss.backward()
        optimizer.step()
        
        # print(y_batch == y_pred.argmax(dim=1))
        acc = (y_batch == y_pred.argmax(dim=1)).float().sum() / len(y_batch)
        train_metrics.update(acc)
        # Backward pass
        # loss.backward()

        # Update the weights
        # optimizer.step()

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
    return train_metrics ,eval_metrics


def test_loop(model, dataloader):
    model.to(device)
    loss_metric = AverageMeter()
    acc_metric = AverageMeter()
    with torch.no_grad():
        model.eval()
        # Loop over the validation data
        for X_batch, y_batch in dataloader:

            X_batch = X_batch.to(device) 
            y_batch = y_batch.to(device)
            # Forward pass
            y_pred = model(X_batch)

            # Compute the loss
            loss = criterion(y_pred, y_batch).to(device)
            loss_metric.update(loss)

            # Update the evaluation metrics
            correct_val = (y_batch == y_pred.argmax(dim=1)).float().sum()
            acc_metric.update(correct_val / len(y_batch))
        # Return the evaluation metrics
    return loss_metric, acc_metric


models = []

test_dataset = FireDataset(os.path.join(datainit, "test"), transform=transform)
# load the test data
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

for fold in range(k):
    ep_train= AverageMeter()
    ep_val = AverageMeter()
    model = Selfire(inputchanel, classnum, qorder)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False) 

    for epoch in tqdm(range(n_epochs)):
        
        # Create train dataset
        train_dataset = FireDataset(os.path.join(dataworkdir, str(fold), "train"), transform=transform)
        # load train data
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        val_dataset = FireDataset(os.path.join(dataworkdir, str(fold), "test"), transform=transform)
        # load val data
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        train_acc, val_acc = run_epoch(model,train_dataloader=train_dataloader, val_dataloader=val_dataloader, criterion=criterion, optimizer=optimizer)
        
        ep_train.update(train_acc.avg)
        ep_val.update(val_acc.avg)
        print(f"Epoch {epoch} [ train_acc:{train_acc.avg.item():.3f} | val_acc:{val_acc.avg.item():.3f}]")
        if (1 - val_acc.avg)<=1e-6:
            print("early stop, best model found at epoch ", epoch)
            break
    print(f"Fold {fold} [ train_acc:{ep_train.avg.item():.3f} | val_acc:{ep_val.avg.item():.3f}]")
    models.append(model)

test_metrics = AverageMeter()
test_loss = AverageMeter()

for fold, model in enumerate(models):
    loss, acc = test_loop(model, test_dataloader)
    print(f"Model k={fold} [ Loss:{loss.avg.item():.3f} | Accuracy:{acc.avg.item():.3f}]")
    test_metrics.update(acc.avg)
    test_loss.update(loss.avg)

    torch.save(model.state_dict(), f"model_with_loss_{loss.avg:.03f}_acc_{acc.avg:.03f}.pth")

print(f"Average [ Loss:{test_loss.avg.item():.3f} | Accuracy:{test_metrics.avg.item():.3f}]")






    
