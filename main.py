# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 15:30:33 2024

@author: Tuan
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import DigitClassifierCNN
from tqdm import tqdm
from data_loader import ImgData
import os
import numpy as np
import pandas as pd


class EarlyStopping:
    def __init__(self, patience=10, sign=-1, verbose=False, criteria='loss', epsilon=0.0):
        """
        Args:
            patience (int): How long to wait after last time improved.
                            Default: 7
            verbose (bool): If True, prints a message for each improvement.
                            Default: False
            epsilon (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0.0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.epsilon = epsilon
        self.sign = sign

        if self.verbose:
            self.sign_str = '>' if sign < 0 else '<'
            self.sign_str2 = 'decreased' if sign < 0 else 'increased'
            self.criteria = criteria

    def __call__(self, score, model):

        score = score * self.sign

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)

        elif score < self.best_score * (1 + self.epsilon * self.sign):
            self.counter += 1

            if self.verbose:
                print(f'\nCurrent {self.criteria} {score:.4f} {self.sign_str} {self.best_score:.4f} * {1 + self.epsilon * self.sign} = {self.best_score * (1 + self.epsilon * self.sign):.4f}')
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(score, model)
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, score, model):

        if self.verbose:
            print(f'\n{self.criteria} {self.sign_str2} ({self.best_score:.4f} --> {score:.4f}).  Saving model ...\n')
        # Note: Here you should define how you want to save your model. For example:
        torch.save(model.state_dict(), os.path.join(model_path, f'{DS_NAME[ds_idx]}_classifier.pkl'))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 100
BATCH_SIZE = 64

# MNIST_path = './dataset/MNIST_train.pt'
# MNISTM_path = './dataset/MNISTM_train.pt'
# SYN_path = './dataset/SYN_train.pt'
# USPS_path = './dataset/USPS_train.pt'

# ds_path = [MNIST_path, MNISTM_path, SYN_path, USPS_path]
# DS_NAME = ["MNIST", "MNISTM", "SYN", "USPS"]
DS_NAME = ["USPS"]

# ds_idx = 0

for ds_idx in range(len(DS_NAME)):

    train_ds = ImgData(f'./dataset/{DS_NAME[ds_idx]}_train.pt', 32, 32)
    test_ds = ImgData(f'./dataset/{DS_NAME[ds_idx]}_test.pt', 32, 32)

    train_loader = torch.utils.data.DataLoader(train_ds,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False)

    model_path = './model'
    os.makedirs(model_path, exist_ok=True)

    current_setting = f'{DS_NAME[ds_idx]}_batch{BATCH_SIZE}'
    print(f'Current train setting: {current_setting}')

    model_path = os.path.join(model_path, current_setting)
    os.makedirs(model_path, exist_ok=True)

    classifier = DigitClassifierCNN().to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters())

    early_stopping = EarlyStopping(patience=30, sign=1, verbose=True, criteria='test accuracy')

    train_info = []
    test_info = []

    for epoch in range(EPOCHS):

        classifier.train()

        ep_train_loss = 0.0
        ep_train_correct = 0

        for _, (train_data, train_target) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1} training progress'):

            train_data, train_target = train_data.to(device), train_target.to(device)

            optimizer.zero_grad()

            train_output = classifier(train_data)

            train_loss = criterion(train_output, train_target)

            train_loss.backward()
            optimizer.step()

            ep_train_loss += train_loss.item()

            train_pred = train_output.argmax(dim=1, keepdim=True)
            ep_train_correct += train_pred.eq(train_target.view_as(train_pred)).sum().item()

        ep_train_loss /= len(train_ds)
        ep_train_accuracy = ep_train_correct / len(train_ds) * 100

        train_info.append([ep_train_loss, ep_train_accuracy])

        print()
        print('Train set:')
        print(f'Average loss: {ep_train_loss:.4f}, Accuracy: {ep_train_correct}/{len(train_ds)} ({ep_train_accuracy:.4f}%)')
        print("------------------------------")


        classifier.eval()

        ep_test_loss = 0.0
        ep_test_correct = 0

        with torch.no_grad():
            # for _, (data, target) in tqdm(enumerate(test_loader), total=len(test_loader), desc=f'Epoch {epoch+1} testing progress'):
            for _, (test_data, test_target) in enumerate(test_loader):

                test_data, test_target = test_data.to(device), test_target.to(device)

                test_output = classifier(test_data)

                ep_test_loss += criterion(test_output, test_target).item()

                test_pred = test_output.argmax(dim=1, keepdim=True)

                ep_test_correct += test_pred.eq(test_target.view_as(test_pred)).sum().item()

        ep_test_loss /= len(test_ds)
        ep_test_accuracy = ep_test_correct / len(test_ds) * 100

        test_info.append([ep_test_loss, ep_test_accuracy])


        print('Test set:')
        print(f'Average loss: {ep_test_loss:.4f}, Accuracy: {ep_test_correct}/{len(test_ds)} ({ep_test_accuracy:.4f}%)')
        print("==============================")

        early_stopping(ep_test_accuracy, classifier)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}.\n")
            break


    df = pd.DataFrame(train_info, columns=['Train Loss', 'Train Accuracy'])
    train_info_path = os.path.join('.', 'train_info', current_setting)
    os.makedirs(train_info_path, exist_ok=True)
    train_info_path = os.path.join(train_info_path, 'train_info.csv')
    df.to_csv(train_info_path, index=True)

    print(f'Saving train info to {train_info_path}')

    df = pd.DataFrame(test_info, columns=['Test Loss', 'Test Accuracy'])
    test_info_path = os.path.join('.', 'test_info', current_setting)
    os.makedirs(test_info_path, exist_ok=True)
    test_info_path = os.path.join(test_info_path, 'test_info.csv')
    df.to_csv(test_info_path, index=True)

    print(f'Saving test info to {train_info_path}')
    print()

print('All done')


