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
from dataloaders_pytorch import  LUNA_Dataset_3D, LUNA_Dataset_3D_scaled
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
    "model": "baseline_weights",
    "Dropout":'yes',
    "dropout_conv" : 0.3,
    "dropout_fc" :0.9,

}

#log file 
LOGS_dir_prefix = "../model_checkpoints"
LOGS_path = LOGS_dir_prefix +'/'+ params["model"]

output_path = '../output'
LOGS_new_path = output_path + '/' + 'baseline_results' + '/' + 'results_on_final_test_set'
if not os.path.exists(LOGS_new_path):
    os.makedirs(LOGS_new_path)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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

# Test the trained model on low dose patients
data = pd.read_csv('../low_dose_test.csv')

# ELIF 3-D DATASET
test_dataset = LUNA_Dataset_3D_scaled(data)
test_loader = utils.DataLoader(test_dataset, batch_size=1, shuffle=False)

TEST = 'test'
dataloaders_dict = {'test': test_loader}

dataloaders_length = len(test_loader)
recall_test, acc_test, conf_matrix,predict_labels,probability_vector = predict(model, dataloaders_dict,dataloaders_length)

# write test_fold9 with probability scores
test_fold9 = "../low_dose_test.csv"
score_path= LOGS_new_path + '/' + "ld_test_csv_files"
if not os.path.exists(score_path):
    os.makedirs(score_path)
test_fold9_score = score_path + '/' + "test_score.csv"
write_csv(probability_vector,test_fold9,test_fold9_score)

## write submission csv file 
CandidateFile = "../annotations/candidatesFile.csv"
submission_path = LOGS_new_path + '/' + "ld_Evaluation" 
if not os.path.exists(submission_path):
    os.makedirs(submission_path)    
submission_csv= submission_path + '/' + 'submission.csv'
write_submission_file(CandidateFile,submission_csv,test_fold9_score)

##### Standard Evaluation #####
annotations_filename          = "../annotations/annotations.csv"
annotations_excluded_filename = "../annotations/annotations_excluded.csv"
seriesuids_filename           = "../annotations/seriesuids.csv"
results_filename              = submission_csv
outputDir                     = LOGS_new_path + '/' + "ld_Evaluation"

low_dose_test_patient_dir="../scans/LD_valid_test_scans"
seriesUIDs_LD = []
for subset in os.listdir(low_dose_test_patient_dir):
    subset_path = os.path.join(low_dose_test_patient_dir,subset)
    for id in os.listdir(subset_path):
        if id.endswith('.mhd'):
            id = id[:-4]
            seriesUIDs_LD.append(id)
(allNodules, seriesUIDs) = collect(annotations_filename, annotations_excluded_filename, seriesuids_filename)
allNodules = {key:value for key, value in allNodules.items() if key in seriesUIDs_LD}

#Evaluation Settings
bPerformBootstrapping = True
bNumberOfBootstrapSamples = 1000
bOtherNodulesAsIrrelevant = True
bConfidence = 0.95

# CAD evaluation
evaluateCAD(seriesUIDs_LD, results_filename, outputDir, allNodules,
            os.path.splitext(os.path.basename(results_filename))[0],
            maxNumberOfCADMarks=100, performBootstrapping=bPerformBootstrapping,
            numberOfBootstrapSamples=bNumberOfBootstrapSamples, confidence=bConfidence)

print("Finished testing low dose patients!")


# Test the trained model on standard Dose Patients
sd_data = pd.read_csv('../standard_dose_test.csv')

# ELIF 3-D DATASET
sd_test_dataset = LUNA_Dataset_3D_scaled(sd_data)
sd_test_loader = utils.DataLoader(sd_test_dataset, batch_size=1, shuffle=False)

sd_dataloaders_dict = {'test2': sd_test_loader}

sd_dataloaders_length = len(sd_test_loader)
recall_test2, acc_test2, conf_matrix2,predict_labels2,probability_vector2 = predict2(model, sd_dataloaders_dict,sd_dataloaders_length)


# write test_csv2 with probability scores
test_csv2 = "../standard_dose_test.csv"
score_path= LOGS_new_path + '/' + "sd_test_csv_files"
if not os.path.exists(score_path):
    os.makedirs(score_path)
test_score2 = score_path + '/' + "test_score.csv"
write_csv(probability_vector2,test_csv2,test_score2)

## write submission csv file 
submission_path2 = LOGS_new_path + '/' + "sd_Evaluation" 
if not os.path.exists(submission_path2):
    os.makedirs(submission_path2)    
submission_csv2= submission_path2 + '/' + 'submission.csv'
write_submission_file(CandidateFile,submission_csv2,test_score2)

#### Standard Evaluation for high_dose_patients#####
results_filename2              = submission_csv2
outputDir2                     = LOGS_new_path + '/' + "sd_Evaluation"

standard_dose_test_patient_dir="../scans/SD_valid_test_scans"
seriesUIDs_SD = []
for subset in os.listdir(standard_dose_test_patient_dir):
    subset_path = os.path.join(standard_dose_test_patient_dir,subset)
    for id in os.listdir(subset_path):
        if id.endswith('.mhd'):
            id = id[:-4]
            seriesUIDs_SD.append(id)
(allNodules2, seriesUIDs) = collect(annotations_filename, annotations_excluded_filename, seriesuids_filename)
allNodules_ld = {key:value for key, value in allNodules2.items() if key in seriesUIDs_SD}

#CAD evaluation
evaluateCAD(seriesUIDs_SD, results_filename2, outputDir2, allNodules_ld,
            os.path.splitext(os.path.basename(results_filename2))[0],
            maxNumberOfCADMarks=100, performBootstrapping=bPerformBootstrapping,
            numberOfBootstrapSamples=bNumberOfBootstrapSamples, confidence=bConfidence)

print("Finished testing on standard dose patients!")


# Test the trained model on high Dose Patients
hd_data = pd.read_csv('../high_dose_unnorm_test.csv')

# ELIF 3-D DATASET
hd_test_dataset = LUNA_Dataset_3D(hd_data)
hd_test_loader = utils.DataLoader(hd_test_dataset, batch_size=1, shuffle=False)

hd_dataloaders_dict = {'test3': hd_test_loader}

hd_dataloaders_length = len(hd_test_loader)
recall_test3, acc_test3, conf_matrix3,predict_labels3,probability_vector3 = predict3(model, hd_dataloaders_dict,hd_dataloaders_length)


## write test_csv2 with probability scores
test_csv3 = "../high_dose_unnorm_test.csv"
score_path= LOGS_new_path + '/' + "hd_test_csv_files"
if not os.path.exists(score_path):
    os.makedirs(score_path)
test_score3 = score_path + '/' + "test_score.csv"
write_csv(probability_vector3,test_csv3,test_score3)

## write submission csv file 
submission_path3 = LOGS_new_path + '/' + "hd_Evaluation" 
if not os.path.exists(submission_path3):
    os.makedirs(submission_path3)    
submission_csv3= submission_path3 + '/' + 'submission.csv'
write_submission_file(CandidateFile,submission_csv3,test_score3)

##### Standard Evaluation for high_dose_patients#####
results_filename3              = submission_csv3
outputDir3                     = LOGS_new_path + '/' + "hd_Evaluation"

hd_dose_test_patient_dir="../scans/high_dose_test_scans"
seriesUIDs_HD = []
for subset in os.listdir(hd_dose_test_patient_dir):
    subset_path = os.path.join(hd_dose_test_patient_dir,subset)
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

print("Finished testing on High dose patients!")

##### CPM Calculation #####
FROC_bootstrap = outputDir + '/' + 'froc_' + results_filename.split('/')[-1].split('.csv')[0] + '_bootstrapping.csv'
FROC = pd.read_csv(FROC_bootstrap, skipinitialspace=True)
FPrate_values = FROC['FPrate'].values
sensitivity = FROC['Sensivity[Mean]'].values
fps_points = np.array([0.125,0.25,0.5,1,2,4,8])
cpm_array = np.interp(fps_points, FPrate_values, sensitivity)
CPM = np.mean(cpm_array)
print(f'CPM for Low Dose Patients :{CPM}')

##### CPM Calculation #####
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