import torch
from torch.utils.data import Dataset
from cv import CrossValidation
import pandas as pd
from PIL import Image
import os
import shutil
import glob
import numpy as np


etiquette = {"fire":0, "nonfire":1}

def create_folder(fold):
    if not os.path.exists(fold):
        os.makedirs(fold)

 

def write_out(df, pth):
    [create_folder(os.path.join(pth, i)) for i in etiquette] 
    for v in df["Image"].values:
        shutil.copy2(v, os.path.join(pth, v.split("\\")[1]))



######## Cross validation ########
def crossval(datasourced, dataworkd, k=5, first_split=False):

    images=glob.glob(datasourced +"/*/*")
    labels=[]

    for file in images:
        clss = file.split("\\")[1]
        labels.append(clss)

    print('image number',len(images))
    print('label number',len(labels))
    

    data = {'Image':images, 'Label':labels} 

    dfdata = pd.DataFrame(data) 


    train_out_pth = "train"
    test_out_pth = "test"
    cv = CrossValidation(dfdata, shuffle=True, target_cols=["Label"], num_folds=k) 
    splitdf = cv.split()
    print(splitdf)

    # grouped kfold into train and test folders respectively
    # split them into 80-20 respectively if does not want cross val
    if first_split:
        cv = CrossValidation(dfdata, shuffle=True, target_cols=["Label"], num_folds=5)
        splitdf = cv.split()
        fold = 0
        train, test = splitdf[splitdf['kfold'] != fold], splitdf[splitdf['kfold'] == fold]
        
        print(test, "\n\n", train)
        train_path = os.path.join(dataworkd, train_out_pth)
        test_path = os.path.join(dataworkd, test_out_pth)

        #[create_folder(os.path.join(train_path, i)) for i in etiquette]
    
        write_out(train, train_path)
        write_out(test,  test_path)

    # then split the train fold in kfold you want
    else:
        for fold in range(k):
            train, test = splitdf[splitdf['kfold'] != fold], splitdf[splitdf['kfold'] == fold]
        
            write_out(train,  os.path.join(dataworkd,  str(fold), train_out_pth))
            write_out(test,   os.path.join(dataworkd, str(fold), test_out_pth))


class FireDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        # self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
        self.image_paths = [f for f in glob.glob(data_dir+"/*/*.jpg")]

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        # Calculate the new height of the image
        #w, h = image.size
        #new_h = int(h * (224 / w))
        # Resize the image
        #image = image.resize((224, new_h))

        if self.transform:
            image = self.transform(image)

        getlabel = self.image_paths[idx].split("\\")[-2]
        # Read the image file to get the image data and label
        # image = torch.from_numpy(np.asarray(image)).float()
        # label = int('nonfire' not in self.image_paths[idx])
        label = etiquette[getlabel]

        return image, label
