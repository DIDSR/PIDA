import os
import torch
import torch.nn as nn
import torchvision
from torchvision import models
import numpy as np
from torchsummary import summary
import torch.nn.functional as F
import torch.utils.data as utils
from tqdm import tqdm
from os.path import isfile, join
from os import listdir
import pandas as pd
import time, copy
from torch.autograd import Variable
from sklearn.metrics import recall_score, confusion_matrix

test_batches = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def replace_column(evaluation_dir, probability):
    data = pd.read_csv(evaluation_dir)
    # for i in range(len(data['probability'])):
    data['probability']= [float(x) for x in probability]
    data.to_csv(evaluation_dir,index=False)

def predict(mod, dataloaders, dataloader_len):
    since = time.time()
    acc = 0
    recall = 0
    # batches = len(dataloaders)
    mod.train(False)
    mod.eval()
    true_lables = []
    predict_labels = []
    probability_vector = []

    for inputs, labels in dataloaders['test']:
        inputs, labels = Variable(inputs.to(device, dtype=torch.float), volatile=True), Variable(labels.to(device, dtype=torch.long), volatile=True)
        true_label= labels.cpu().data.numpy()
        true_lables.append(true_label)
        outputs = mod(inputs)
        m = nn.Softmax()
        soft_out = m(outputs)
        probability_score = soft_out[:,1]
        values, preds = torch.max(outputs.data, 1)
        predict_labels.append(preds.cpu().data.numpy())
        # probability_vector.append(probability.cpu().data.numpy())
        probability_vector.append(probability_score.cpu().data.numpy())
        acc += torch.sum(preds == labels.data)
        recall += recall_score(torch.Tensor.cpu(labels.data), torch.Tensor.cpu(preds))
        del inputs, labels, outputs, preds,values
        torch.cuda.empty_cache()

    acc = float(acc / dataloader_len)
    print(len(dataloaders))
    print(f"acc:{acc}")
    recall = float(recall / dataloader_len)
    print(f"recall:{recall}")
    conf_matrix = confusion_matrix(true_lables, predict_labels)
    print(f"conf_matrix:{conf_matrix}")
 
    return recall, acc, conf_matrix, predict_labels, probability_vector

def predict_F(mod, dataloaders, dataloader_len):
    since = time.time()
    acc = 0
    recall = 0
    # batches = len(dataloaders)
    mod.train(False)
    mod.eval()
    true_lables = []
    predict_labels = []
    probability_vector = []

    for inputs, labels in dataloaders['test']:
        inputs, labels = Variable(inputs.to(device, dtype=torch.float), volatile=True), Variable(labels.to(device, dtype=torch.long), volatile=True)
        true_label= labels.cpu().data.numpy()
        true_lables.append(true_label)
        outputs = mod(inputs)
        outputs = mod(inputs)
        m = nn.Softmax()
        soft_out = m(outputs)
        probability_score = soft_out[:,1]
        values, preds = torch.max(outputs.data, 1)
        predict_labels.append(preds.cpu().data.numpy())
        # probability_vector.append(probability.cpu().data.numpy())
        probability_vector.append(probability_score.cpu().data.numpy())
        acc += torch.sum(preds == labels.data)
        recall += recall_score(torch.Tensor.cpu(labels.data), torch.Tensor.cpu(preds))
        del inputs, labels, outputs, preds,values
        torch.cuda.empty_cache()

    acc = float(acc / dataloader_len)
    print(len(dataloaders))
    print(f"acc:{acc}")
    recall = float(recall / dataloader_len)
    print(f"recall:{recall}")
    conf_matrix = confusion_matrix(true_lables, predict_labels)
    print(f"conf_matrix:{conf_matrix}")
 
    return recall, acc, conf_matrix, predict_labels, probability_vector

def predict2(mod, dataloaders, dataloader_len):
    since = time.time()
    acc = 0
    recall = 0
    # batches = len(dataloaders)
    mod.train(False)
    mod.eval()
    true_lables = []
    predict_labels = []
    probability_vector = []

    for inputs, labels in dataloaders['test2']:
        inputs, labels = Variable(inputs.to(device, dtype=torch.float), volatile=True), Variable(labels.to(device, dtype=torch.long), volatile=True)
        true_label= labels.cpu().data.numpy()
        true_lables.append(true_label)
        outputs = mod(inputs)
        m = nn.Softmax()
        soft_out = m(outputs)
        probability_score = soft_out[:,1]
        values, preds = torch.max(outputs.data, 1)
        predict_labels.append(preds.cpu().data.numpy())
        # probability_vector.append(probability.cpu().data.numpy())
        probability_vector.append(probability_score.cpu().data.numpy())
        acc += torch.sum(preds == labels.data)
        recall += recall_score(torch.Tensor.cpu(labels.data), torch.Tensor.cpu(preds))
        del inputs, labels, outputs, preds,values
        torch.cuda.empty_cache()

    acc = float(acc / dataloader_len)
    print(len(dataloaders))
    print(f"acc:{acc}")
    recall = float(recall / dataloader_len)
    print(f"recall:{recall}")
    conf_matrix = confusion_matrix(true_lables, predict_labels)
    print(f"conf_matrix:{conf_matrix}")
 
    return recall, acc, conf_matrix, predict_labels, probability_vector


def predict3(mod, dataloaders, dataloader_len):
    since = time.time()
    acc = 0
    recall = 0
    # batches = len(dataloaders)
    mod.train(False)
    mod.eval()
    true_lables = []
    predict_labels = []
    probability_vector = []

    for inputs, labels in dataloaders['test3']:
        inputs, labels = Variable(inputs.to(device, dtype=torch.float), volatile=True), Variable(labels.to(device, dtype=torch.long), volatile=True)
        true_label= labels.cpu().data.numpy()
        true_lables.append(true_label)
        outputs = mod(inputs)
        m = nn.Softmax()
        soft_out = m(outputs)
        probability_score = soft_out[:,1]
        values, preds = torch.max(outputs.data, 1)
        predict_labels.append(preds.cpu().data.numpy())
        # probability_vector.append(probability.cpu().data.numpy())
        probability_vector.append(probability_score.cpu().data.numpy())
        acc += torch.sum(preds == labels.data)
        recall += recall_score(torch.Tensor.cpu(labels.data), torch.Tensor.cpu(preds))
        del inputs, labels, outputs, preds,values
        torch.cuda.empty_cache()

    acc = float(acc / dataloader_len)
    print(len(dataloaders))
    print(f"acc:{acc}")
    recall = float(recall / dataloader_len)
    print(f"recall:{recall}")
    conf_matrix = confusion_matrix(true_lables, predict_labels)
    print(f"conf_matrix:{conf_matrix}")
 
    return recall, acc, conf_matrix, predict_labels, probability_vector