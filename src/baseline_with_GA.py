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
from dataloaders_pytorch import  LUNA_Dataset_3D,LUNA_Dataset_3D_scaled
from train_tools_fl import train_model,write_csv,write_submission_file
from noduleCADEvaluationLUNA16 import collect,evaluateCAD
from test_tools import predict,predict2,predict3
from torchvision import transforms, datasets
# import dicaugment as dca
import json
import random
import ast
from albumentations import (
    Compose,
    HorizontalFlip,
    VerticalFlip,
    RandomRotate90,
    Transpose,
    ShiftScaleRotate,
    Rotate
)


params = {
    "model": "baseline_with_GA_weights",
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
    "p":0.6

}

#log file create
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

class NPYdataset_3d_augmented(Dataset):
    def __init__(self, data):
        self.data = data
        self.augmentator = Compose([
            # Non destructive transformations
            VerticalFlip(p=0.6),
            HorizontalFlip(p=0.6),
            RandomRotate90(p=0.6),
            Rotate(p=0.6),
            ShiftScaleRotate(p=0.2, scale_limit=(0.1, 0.3)),
            Transpose(p=0.6)
        ])

    def __getitem__(self, index):
        image = self.data.iloc[index]
        y_class = int(image['class'])
        lung_img = np.load(image['filename'])
        augmented = self.augmentator(image=lung_img)
        aug_lung_img = augmented['image']
        X = normalizePlanes(aug_lung_img)
        X = X.reshape(49, 49, 17)
        return X.reshape((1, 49, 49, 17)), y_class
    
    def __len__(self):
        return self.data.shape[0] 

print('saved train configuration!')
shuffle_dataset = True

# load data
root_dir = "../data/49x49x17"
train_data = pd.read_csv(root_dir + "/train_csv_files/HU_data/train_fold0_8_unnorm.csv")

# IF 3-D DATASET
train_dataset = NPYdataset_3d_augmented(train_data)
random.seed(42)

train_loader = utils.DataLoader(train_dataset, batch_size=params["batch_size"],num_workers=32, shuffle=True, pin_memory=True)

# # set GPU
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


# MODELS

model = CNNT_3D(Dropout=params["Dropout"],dropout_conv=params["dropout_conv"],dropout_fc=params["dropout_fc"])
model=model.eval()
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

path_weights = LOGS_path + '/'+ 'CNNT_3D.pt'
checkpoint = torch.load(path_weights)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)



## write submission csv file 
CandidateFile = "../annotations/candidates_V2_kernel_excluded.csv"

##### Standard Evaluation #####
annotations_filename          = "../annotations/annotations_kernel_excluded.csv"
annotations_excluded_filename = "../annotations/annotations_excluded.csv"
seriesuids_filename           = "../annotations/seriesuids.csv"


#Evaluation Settings
bPerformBootstrapping = True
bNumberOfBootstrapSamples = 1000
bOtherNodulesAsIrrelevant = True
bConfidence = 0.95

# Test the trained model on Low Dose Patients
ld_data = pd.read_csv('../test_fold9_low_dose.csv')

# ELIF 3-D DATASET
ld_test_dataset = LUNA_Dataset_3D_scaled(ld_data)
ld_test_loader = utils.DataLoader(ld_test_dataset, batch_size=1, shuffle=False)

ld_dataloaders_dict = {'test': ld_test_loader}

ld_dataloaders_length = len(ld_test_loader)
recall_test, acc_test, conf_matrix,predict_labels,probability_vector = predict(model, ld_dataloaders_dict,ld_dataloaders_length)


## write test_csv2 with probability scores
test_csv = "../test_fold9_low_dose.csv"
score_path= LOGS_path + '/' + "ld_test_csv_files"
if not os.path.exists(score_path):
    os.makedirs(score_path)
test_score = score_path + '/' + "test_score.csv"
write_csv(probability_vector,test_csv,test_score)

## write submission csv file 
submission_path = LOGS_path + '/' + "ld_Evaluation" 
if not os.path.exists(submission_path):
    os.makedirs(submission_path)    
submission_csv= submission_path + '/' + 'submission.csv'
write_submission_file(CandidateFile,submission_csv,test_score)

##### Standard Evaluation for low_dose_patients#####
results_filename              = submission_csv
outputDir                     = LOGS_path + '/' + "ld_Evaluation"

subset_path="../low_dose_fold9/subset9"
seriesUIDs_LD = []
for id in os.listdir(subset_path):
    if id.endswith('.mhd'):
        id = id[:-4]
        seriesUIDs_LD.append(id)
(allNodules, seriesUIDs) = collect(annotations_filename, annotations_excluded_filename, seriesuids_filename)
allNodules_ld = {key:value for key, value in allNodules.items() if key in seriesUIDs_LD}

#CAD evaluation
evaluateCAD(seriesUIDs_LD, results_filename, outputDir, allNodules_ld,
            os.path.splitext(os.path.basename(results_filename))[0],
            maxNumberOfCADMarks=100, performBootstrapping=bPerformBootstrapping,
            numberOfBootstrapSamples=bNumberOfBootstrapSamples, confidence=bConfidence)

print("Finished testing on low dose patients!")


# Test the trained model on Standard Dose Patients
sd_data = pd.read_csv('../test_fold9_standard_dose.csv')

# ELIF 3-D DATASET
sd_test_dataset = LUNA_Dataset_3D_scaled(sd_data)
sd_test_loader = utils.DataLoader(sd_test_dataset, batch_size=1, shuffle=False)

sd_dataloaders_dict = {'test2': sd_test_loader}

sd_dataloaders_length = len(sd_test_loader)
recall_test2, acc_test2, conf_matrix2,predict_labels2,probability_vector2 = predict2(model, sd_dataloaders_dict,sd_dataloaders_length)

## write test_csv2 with probability scores
test_csv2 = "../test_fold9_standard_dose.csv"
score_path= LOGS_path + '/' + "sd_test_csv_files"
if not os.path.exists(score_path):
    os.makedirs(score_path)
test_score2 = score_path + '/' + "test_score.csv"
write_csv(probability_vector2,test_csv2,test_score2)

## write submission csv file 
submission_path2 = LOGS_path + '/' + "sd_Evaluation" 
if not os.path.exists(submission_path2):
    os.makedirs(submission_path2)    
submission_csv2= submission_path2 + '/' + 'submission.csv'
write_submission_file(CandidateFile,submission_csv2,test_score2)

##### Standard Evaluation for high_dose_patients#####
results_filename2              = submission_csv2
outputDir2                     = LOGS_path + '/' + "sd_Evaluation"

subset_path="../standard_dose_fold9/subset9"
seriesUIDs_SD = []
for id in os.listdir(subset_path):
    if id.endswith('.mhd'):
        id = id[:-4]
        seriesUIDs_SD.append(id)
(allNodules2, seriesUIDs) = collect(annotations_filename, annotations_excluded_filename, seriesuids_filename)
allNodules_sd = {key:value for key, value in allNodules2.items() if key in seriesUIDs_SD}

#CAD evaluation
evaluateCAD(seriesUIDs_SD, results_filename2, outputDir2, allNodules_sd,
            os.path.splitext(os.path.basename(results_filename2))[0],
            maxNumberOfCADMarks=100, performBootstrapping=bPerformBootstrapping,
            numberOfBootstrapSamples=bNumberOfBootstrapSamples, confidence=bConfidence)

print("Finished testing on standard dose patients!")


# Test the trained model on High Dose Patients
hd_data = pd.read_csv('../test_fold9_high_dose.csv')

# ELIF 3-D DATASET
hd_test_dataset = LUNA_Dataset_3D_scaled(hd_data)
hd_test_loader = utils.DataLoader(hd_test_dataset, batch_size=1, shuffle=False)

hd_dataloaders_dict = {'test3': hd_test_loader}

hd_dataloaders_length = len(hd_test_loader)
recall_test3, acc_test3, conf_matrix3,predict_labels3,probability_vector3 = predict3(model, hd_dataloaders_dict,hd_dataloaders_length)


## write test_csv2 with probability scores
test_csv3 = "../test_fold9_high_dose.csv"
score_path= LOGS_path + '/' + "hd_test_csv_files"
if not os.path.exists(score_path):
    os.makedirs(score_path)
test_score3 = score_path + '/' + "test_score.csv"
write_csv(probability_vector3,test_csv3,test_score3)

## write submission csv file 
submission_path3 = LOGS_path + '/' + "hd_Evaluation" 
if not os.path.exists(submission_path3):
    os.makedirs(submission_path3)    
submission_csv3= submission_path3 + '/' + 'submission.csv'
write_submission_file(CandidateFile,submission_csv3,test_score3)

##### Standard Evaluation for high_dose_patients#####
results_filename3              = submission_csv3
outputDir3                     = LOGS_path + '/' + "hd_Evaluation"

subset_path="../high_dose_fold9/subset9"
seriesUIDs_HD = []
for id in os.listdir(subset_path):
    if id.endswith('.mhd'):
        id = id[:-4]
        seriesUIDs_HD.append(id)
(allNodules3, seriesUIDs) = collect(annotations_filename, annotations_excluded_filename, seriesuids_filename)
allNodules_hd = {key:value for key, value in allNodules3.items() if key in seriesUIDs_HD}

#CAD evaluation
evaluateCAD(seriesUIDs_HD, results_filename3, outputDir3, allNodules_hd,
            os.path.splitext(os.path.basename(results_filename3))[0],
            maxNumberOfCADMarks=100, performBootstrapping=bPerformBootstrapping,
            numberOfBootstrapSamples=bNumberOfBootstrapSamples, confidence=bConfidence)

print("Finished testing on high dose patients!")


# ##### CPM Calculation #####
FROC_bootstrap = outputDir + '/' + 'froc_' + results_filename.split('/')[-1].split('.csv')[0] + '_bootstrapping.csv'
FROC = pd.read_csv(FROC_bootstrap, skipinitialspace=True)
FPrate_values = FROC['FPrate'].values
sensitivity = FROC['Sensivity[Mean]'].values
fps_points = np.array([0.125,0.25,0.5,1,2,4,8])
cpm_array = np.interp(fps_points, FPrate_values, sensitivity)
CPM = np.mean(cpm_array)
print(f'CPM for test fold9 Patients :{CPM}')


###### CPM Calculation #####
FROC_bootstrap = outputDir2 + '/' + 'froc_' + results_filename2.split('/')[-1].split('.csv')[0] + '_bootstrapping.csv'
FROC = pd.read_csv(FROC_bootstrap, skipinitialspace=True)
FPrate_values = FROC['FPrate'].values
sensitivity = FROC['Sensivity[Mean]'].values
fps_points = np.array([0.125,0.25,0.5,1,2,4,8])
cpm_array = np.interp(fps_points, FPrate_values, sensitivity)
CPM = np.mean(cpm_array)
print(f'CPM for Standard Dose Patients :{CPM}')


##### CPM Calculation #####
FROC_bootstrap = outputDir3 + '/' + 'froc_' + results_filename3.split('/')[-1].split('.csv')[0] + '_bootstrapping.csv'
FROC = pd.read_csv(FROC_bootstrap, skipinitialspace=True)
FPrate_values = FROC['FPrate'].values
sensitivity = FROC['Sensivity[Mean]'].values
fps_points = np.array([0.125,0.25,0.5,1,2,4,8])
cpm_array = np.interp(fps_points, FPrate_values, sensitivity)
CPM = np.mean(cpm_array)
print(f'CPM for high Dose Patients :{CPM}')





