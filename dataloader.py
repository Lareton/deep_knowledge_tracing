# -*- coding: utf-8 -*-
# @Author: jarvis.zhang
# @Date:   2020-05-08 16:21:21
# @Last Modified by:   jarvis.zhang
# @Last Modified time: 2020-05-10 11:47:28
import torch
import torch.utils.data as Data
from readdata import DataReader


def getDataLoader(batch_size, num_of_questions, max_step):
    handle = DataReader('text_dataset_train.txt',
                        'text_dataset_valid.txt',
                        max_step,
                        num_of_questions)
    # dtrain = torch.tensor(handle.getTrainData().astype(float).tolist(),
    #                       dtype=torch.float32)
    # dtest = torch.tensor(handle.getTestData().astype(float).tolist(),
    #                      dtype=torch.float32)

    print("initing dataset ok")
    trainLoader = Data.DataLoader(handle.getTrainData(), batch_size=batch_size, shuffle=True)
    testLoader = Data.DataLoader(handle.getTestData(), batch_size=batch_size, shuffle=False)
    return trainLoader, testLoader
