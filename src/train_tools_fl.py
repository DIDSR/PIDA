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
import csv
from torch import optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def write_csv(predict,input_csv,output_csv):

    score = []
    for i in range(len(predict)):
        score.append(float(predict[i]))
    with open(input_csv,'r') as csvinput:
        with open(output_csv, 'w') as csvoutput:
            writer = csv.writer(csvoutput)
            reader = csv.reader(csvinput)
            all = []
            row = next(reader)
            row.append('probability')
            all.append(row)
            for i, row in enumerate(reader):
                row.append(score[i])
                all.append(row)
            writer.writerows(all)


def write_submission_file(CandidateFile,submission_csv,test_fold9_score):
    
    with open(test_fold9_score,'r') as csvinput:

        reader1 = csv.reader(csvinput)
        data = pd.read_csv(CandidateFile)
        row = next(reader1)
        for row in reader1:
            serial_id = int(row[0].split('/')[-1].strip('.npy'))
            score = row[2]
            print(type(score))
            data.loc[serial_id-2,'probability'] = score
        data.to_csv(submission_csv,sep=',')


def train_model_train(model, criterion, optimizer, gamma, alpha, dataloaders):
    model.train()
    running_loss, running_corrects = 0.0, 0
    phase = 'train/iteration'
    for batch_idx, (inputs, labels) in enumerate(dataloaders['train']):
        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(inputs)
        ce_loss = criterion(outputs, labels)

        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss

        _, preds = torch.max(outputs, 1)       
        focal_loss.backward()
        optimizer.step()

        # batch_loss = focal_loss.item()* inputs.size(0)
        # batch_acc = torch.sum(preds == labels.data)
        # print('{} {} Loss: {:.4f}  Acc: {:.4f} '.format(phase, batch_idx, batch_loss, batch_acc))

        running_loss += focal_loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    return running_loss, running_corrects


def train_model_evaluate(model, criterion,gamma,alpha, dataloaders):
    model.eval()
    running_loss, running_corrects = 0.0, 0
    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(inputs)
            ce_loss = criterion(outputs, labels) 
            pt = torch.exp(-ce_loss)
            focal_loss = alpha * (1 - pt) ** gamma * ce_loss

            _, preds = torch.max(outputs, 1)

            running_loss += focal_loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
    return running_loss, running_corrects

#without validation train model

def train_model(model, dataloaders, criterion, optimizer, gamma, alpha, num_epochs):
    since = time.time()
    logs = pd.DataFrame(columns=['train_loss', 'train_acc'])
    train_acc_history, train_loss_history = [], []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc, best_loss = 0.0, 100.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                running_loss, running_corrects = train_model_train(model, criterion, optimizer, gamma, alpha,  dataloaders)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f}  Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
       
            if phase == 'train':
                epoch_acc=epoch_acc.detach().cpu().numpy()
                train_acc_history.append(epoch_acc), train_loss_history.append(epoch_loss)

        print("Epoch finished! Next Epoch starts....")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))
    train_accuracy = pd.Series(train_acc_history)
    train_loss = pd.Series(train_loss_history)
    logs['train_acc'], logs['train_loss'] = train_accuracy.values, train_loss.values
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, logs


def train_model_lr_scheduler(model, dataloaders, criterion, optimizer, scheduler, gamma, alpha, num_epochs):
    since = time.time()
    logs = pd.DataFrame(columns=['train_loss', 'train_acc'])
    train_acc_history, train_loss_history = [], []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc, best_loss = 0.0, 100.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                running_loss, running_corrects = train_model_train(model, criterion, optimizer, gamma, alpha,  dataloaders)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f}  Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
       
            if phase == 'train':
                epoch_acc=epoch_acc.detach().cpu().numpy()
                train_acc_history.append(epoch_acc), train_loss_history.append(epoch_loss)

        print("Epoch finished! Next Epoch starts....")
        scheduler.step()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))
    train_accuracy = pd.Series(train_acc_history)
    train_loss = pd.Series(train_loss_history)
    logs['train_acc'], logs['train_loss'] = train_accuracy.values, train_loss.values
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, logs

def train_model_prev(model, dataloaders, criterion, optimizer, gamma, alpha, num_epochs):
    since = time.time()
    logs = pd.DataFrame(columns=['val_acc', 'val_loss', 'train_loss', 'train_acc'])
    val_acc_history, val_loss_history = [], []
    train_acc_history, train_loss_history = [], []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc, best_loss = 0.0, 100.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                running_loss, running_corrects = train_model_train(model, criterion, optimizer, gamma, alpha,  dataloaders)
            else:
                running_loss, running_corrects = train_model_evaluate(model, criterion, gamma, alpha, dataloaders)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f}  Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                epoch_acc=epoch_acc.detach().cpu().numpy()
                val_acc_history.append(epoch_acc), val_loss_history.append(epoch_loss)
       
            if phase == 'train':
                epoch_acc=epoch_acc.detach().cpu().numpy()
                train_acc_history.append(epoch_acc), train_loss_history.append(epoch_loss)

        print("Epoch finished! Next Epoch starts....")
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))
    val_accuracy = pd.Series(val_acc_history)
    val_loss = pd.Series(val_loss_history)
    logs['val_acc'],logs['val_loss'] = val_accuracy.values, val_loss.values
    train_accuracy = pd.Series(train_acc_history)
    train_loss = pd.Series(train_loss_history)
    logs['train_acc'], logs['train_loss'] = train_accuracy.values, train_loss.values
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, logs


# def train_model(model, dataloaders, criterion, optimizer, gamma, alpha, num_epochs):
#     since = time.time()
#     logs = pd.DataFrame(columns=['val_acc', 'val_loss', 'train_loss', 'train_acc'])
#     val_acc_history, val_loss_history = [], []
#     train_acc_history, train_loss_history = [], []
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc, best_loss = 0.0, 100.0

#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#         print('-' * 10)

#         # Each epoch has a training and validation phase
#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 running_loss, running_corrects = train_model_train(model, criterion, optimizer, gamma, alpha,  dataloaders)
#             else:
#                 running_loss, running_corrects = train_model_evaluate(model, criterion, gamma, alpha, dataloaders)

#             epoch_loss = running_loss / len(dataloaders[phase].dataset)
#             epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

#             print('{} Loss: {:.4f}  Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
#             if phase == 'val' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())
#             if phase == 'val':
#                 epoch_acc=epoch_acc.detach().cpu().numpy()
#                 val_acc_history.append(epoch_acc), val_loss_history.append(epoch_loss)
       
#             if phase == 'train':
#                 epoch_acc=epoch_acc.detach().cpu().numpy()
#                 train_acc_history.append(epoch_acc), train_loss_history.append(epoch_loss)

#         print("Epoch finished! Next Epoch starts....")
#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
#     print('Best val Loss: {:4f}'.format(best_loss))
#     val_accuracy = pd.Series(val_acc_history)
#     val_loss = pd.Series(val_loss_history)
#     logs['val_acc'],logs['val_loss'] = val_accuracy.values, val_loss.values
#     train_accuracy = pd.Series(train_acc_history)
#     train_loss = pd.Series(train_loss_history)
#     logs['train_acc'], logs['train_loss'] = train_accuracy.values, train_loss.values
    
#     # load best model weights
#     model.load_state_dict(best_model_wts)
#     return model, val_acc_history, logs
