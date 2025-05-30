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
from dataloaders_pytorch import  LUNA_Dataset_3D
from train_tools_fl import train_model,write_csv,write_submission_file
from noduleCADEvaluationLUNA16 import collect,evaluateCAD
from test_tools import predict,predict2,predict3
from torchvision import transforms, datasets
import dicaugment as dca
import json
import random
import ast

# create a dictionary to store hyperparameters
params = {
    "model": "PIDA_weights",
    "lr": 0.0001,
    "batch_size": 64,
    "epochs": 40,
    "STANDARD" : ['STANDARD'],
    "LUNG" : ["LUNG"],
    "B45f": ["B45f"],
    "B70f": ["B70f"],
    "noise_std_range": (1,5),
    "optimizer" :'Adam',
    "criterion": 'FocalLoss',
    "Dropout":'yes',
    "dropout_conv" : 0.3,
    "dropout_fc" :0.9,
    "gamma" : 2,
    "alpha" :16,
    "p":1

}

# #log file create
LOGS_dir_prefix = "../model_checkpoints"
LOGS_path = LOGS_dir_prefix +'/' + params["model"]
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

class LUNA_3D_DICaugmentation(Dataset):
    def __init__(self, data,transform=None):
        self.data = data
        self.transform = transform
        
    def __getitem__(self, index):
        image = self.data.iloc[index]
        y_class = int(image['class'])
        lung_img = np.load(image['filename'], allow_pickle=True)
        lung_img = np.float64(lung_img)
        string_value = image['PixelSpacing']
        values_as_strings = ast.literal_eval(string_value)

        # Convert each string element to a float
        float_values = [float(val) for val in values_as_strings]

        # If you want to access individual values
        f1 = float_values[0]
        f2 = float_values[1]

        dicom = {
                "PixelSpacing" :  (f1,f2),
                "RescaleIntercept" : 0,
                "RescaleSlope" : 1.0,
                "ConvolutionKernel" : image['ConvolutionKernel'],
                "XRayTubeCurrent" : image['XRayTubeCurrent']
                }

        if self.transform is not None:
            kernel = dicom["ConvolutionKernel"]
            if kernel == 'STANDARD':
                dicom["ConvolutionKernel"] = random.choice(params["STANDARD"])
            if kernel == 'LUNG':
                dicom["ConvolutionKernel"] = random.choice(params["LUNG"])
            if kernel == 'B45f':
                dicom["ConvolutionKernel"] = random.choice(params["B45f"])
            if kernel == 'B70f':
                dicom["ConvolutionKernel"] = random.choice(params["B70f"])
     
            lung_img = self.transform(image=lung_img, dicom=dicom)["image"]
            lung_img = normalizePlanes(lung_img)

        X = lung_img.reshape(49, 49, 17)
        return  X.reshape((1, 49, 49, 17)), y_class

    def __len__(self):
        return self.data.shape[0] 

train_transform = dca.Compose(
                            [
                                dca.NPSNoise(magnitude=params['noise_std_range'],sample_tube_current=False,p=1.0)
                            ]
)

print('saved train configuration!')
shuffle_dataset = True

# load data
#Preprocessing
# The data_preprocessing/preprocessing.py script was used to extract lung nodule patches 
# of size 49×49×17 from the original CT scans. 
# The extracted patches were saved as .npy files.

# The full directory paths for these patches were stored in a CSV file named high_dose_train.csv.
root_dir = "../data/49x49x17"
train_data = pd.read_csv(root_dir + "/train_csv_files/high_dose_train.csv")

# IF 3-D DATASET
train_dataset = LUNA_3D_DICaugmentation(train_data,transform=train_transform)
random.seed(42)

train_loader = utils.DataLoader(train_dataset, batch_size=params["batch_size"],num_workers=64, shuffle=True, pin_memory=True)

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




