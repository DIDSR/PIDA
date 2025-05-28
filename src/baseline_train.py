import os
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from glob import glob
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchsummary import summary
import torch.nn.functional as F
import torch.utils.data as utils
from tqdm import tqdm
from os.path import isfile, join
from os import listdir
import datetime
import pandas as pd
import time, copy
from models2_pytorch import CNNT_3D
from vgg_pytorch import vgg11
from dataloaders_pytorch import  LUNA_Dataset_3D
from train_tools_fl import train_model,write_csv,write_submission_file
from noduleCADEvaluationLUNA16 import collect,evaluateCAD
from test_tools import predict,predict2,predict3
from torchvision import transforms, datasets
import dicaugment as dca
import json
import random
import ast


params = {
    "model": "baseline_weights",
    "lr": 0.0001,
    "batch_size": 64,
    "epochs": 40,
    "optimizer" :'Adam',
    "criterion": 'FocalLoss',
    "Dropout":'yes',
    "dropout_conv" : 0.3,
    "dropout_fc" :0.9,
    "gamma" : 2,
    "alpha" :16,
    "p":1

}

#log file create
LOGS_dir_prefix = "../model_checkpoints"
LOGS_path = LOGS_dir_prefix +'/'+ params["model"]

if not os.path.exists(LOGS_path):
    os.makedirs(LOGS_path)

# save hyperparameters in a json file
with open(LOGS_path + '/'+ 'hyperparameters.json','w') as json_file:
    json.dump(params,json_file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def normalizePlanes(npzarray):
    # Clip HUs to [-1000,400], and normalize it to [0,1]
    maxHU = 400.
    minHU = -1000.
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray > 1] = 1.
    npzarray[npzarray < 0] = 0.
    return npzarray

class LUNA_3D_baseline(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __getitem__(self, index):
        image = self.data.iloc[index]
        y_class = int(image['class'])
        lung_img = np.load(image['filename'], allow_pickle=True)        
        lung_img = normalizePlanes(lung_img)
        X = lung_img.reshape(49, 49, 17)
        return  X.reshape((1, 49, 49, 17)), y_class

    def __len__(self):
        return self.data.shape[0] 

print('saved train configuration!')
shuffle_dataset = True

# load data
root_dir = "../data/49x49x17"
train_data = pd.read_csv(root_dir + "/train_csv_files/HU_data/high_dose_unnorm_train.csv")

# IF 3-D DATASET
train_dataset = LUNA_3D_baseline(train_data)
random.seed(42)

train_loader = utils.DataLoader(train_dataset, batch_size=params["batch_size"],num_workers=32, shuffle=True, pin_memory=True)

# set GPU
model = CNNT_3D(Dropout=params["Dropout"],dropout_conv=params["dropout_conv"],dropout_fc=params["dropout_fc"]).to(device)

optimizer = torch.optim.Adam(model.parameters(), params["lr"])
criterion = nn.CrossEntropyLoss()

dataloaders_dict = {'train': train_loader}

# Train and evaluate
model, df_logs = train_model(model, dataloaders_dict, criterion, optimizer, gamma=params["gamma"], alpha=params["alpha"], num_epochs=params["epochs"])

# Saving results and model

df_logs.to_csv(
    LOGS_path + '/' + 'CNNT_3D.csv',
    index=False)
name = 'CNNT_3D'

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
},
    LOGS_path +'/' + '{}.pt'.format(
        name))
print('model saved')

