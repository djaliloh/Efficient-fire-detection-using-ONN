import argparse
import glob
# import numpy as np
import os
import pandas as pd
import shutil
from cv import CrossValidation


def create_folder(fold):
    if not os.path.exists(fold):
        os.makedirs(fold)
   

classes = {True:"fall", False: "nofall"}

def is_fall(pth, criteria="11"):

    with open(pth, "r", encoding="utf-8") as f:
        for line in f.readlines():
            if line.split(",")[0] == str(criteria):
                return True
    return False

def grouper(path): 
    # :: https://stackoverflow.com/questions/59148265/using-python-to-group-files-with-similar-filenames
    dictionary = {} 
    dataset = {"prefix": [] , "label": []} 

    for x in glob.glob(path+"/label/*"): 
        # key = x.split(".")[0].split("-")[0]  # works with os.listdir
        keyy = os.path.basename(x).split(".")[0].split("-")[0] # The key is the first 4 characters of the file name
        if keyy not in dataset["prefix"]:
            dataset["prefix"].append(keyy)
            dataset["label"].append(classes[is_fall(x)])
        group = dictionary.get(keyy,[])
        group.append(x)  
        dictionary[keyy] = group

    return dictionary, dataset

def write_out(df, dict, pth):

    create_folder(os.path.join(pth, "label"))
    create_folder(os.path.join(pth, "jsvideo"))

    for v in df["prefix"].values:
        for filepth in dict[v]:
            shutil.copy2(filepth, os.path.join(pth, "label"))
            shutil.copy2(filepth.replace("label","jsvideo").replace("txt","json"), os.path.join(pth, "jsvideo"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stratified cross validation')
    parser.add_argument('--data_path', default='./data') 
    parser.add_argument('--out_path_train', default='./data/pku/train')
    parser.add_argument('--out_path_test', default='./data/pku/test')
    parser.add_argument('--k_value', type=int, default=5) 
    parser.add_argument('--first_split', action='store_true')
    arg = parser.parse_args()

    # create train and test dir to save k folds
    k = arg.k_value
    print(f"START {k} FOLD CROSS VALIDATIION SPLIT...")
    print(f"train fold has {k} created folds")
    create_folder(arg.out_path_train)
    print(f"test fold has {k} created folds")
    create_folder(arg.out_path_test)

    grouped, dataset = grouper(arg.data_path) 
    df = pd.DataFrame(dataset)

    
    cv = CrossValidation(df, shuffle=True, target_cols=["label"], num_folds=k) #problem_type="binary_classification",
    splitdf = cv.split()
    print(splitdf)

    # grouped kfold into train and test folders respectively
    # split them into 80-10 respectively
    if arg.first_split:
        cv = CrossValidation(df, shuffle=True, target_cols=["label"], num_folds=5)
        splitdf = cv.split()
        fold = 0
        train, test = splitdf[splitdf['kfold'] != fold], splitdf[splitdf['kfold'] == fold]
        

        print(test, "\n\n", train)
        write_out(train, grouped, os.path.join(arg.out_path_train))
        write_out(test, grouped, os.path.join(arg.out_path_test))

    # then split the train fold in kfold you want
    else:
        for fold in range(k):
            train, test = splitdf[splitdf['kfold'] != fold], splitdf[splitdf['kfold'] == fold]
        
            write_out(train, grouped, os.path.join(arg.out_path_train, str(fold)))
            write_out(test, grouped, os.path.join(arg.out_path_test, str(fold)))
   