
import sys
import gc
import numpy as np
import itertools
from tqdm import tqdm
from pympler import muppy, summary
import tracemalloc
tracemalloc.start()

import torch
from torch.utils.data import Dataset
from torchvision import datasets


class CustomDataset(Dataset):
    def __init__(self, data, numofques, maxstep):
        self.data = data
        self.numofques = numofques
        self.maxstep = maxstep

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        len_seq, ques, ans = self.data[idx]

        # если размер сильно больше можно добавить рандом для сдвига
        temp = np.zeros(shape=[self.maxstep, 2 * self.numofques])
        # temp = np.zeros(shape=[self.maxstep, self.numofques + 1])
        for i in range(self.maxstep):
            # temp[i][ques[i]] = 1
            # temp[i][-1] = ans[i]
            #
            if ans[i] == 1:
                temp[i][ques[i]] = 1
            else:
                temp[i][ques[i] + self.numofques] = 1

        return torch.FloatTensor(temp)


class DataReader():
    def __init__(self, train_path, test_path, maxstep, numofques):
        self.train_path = train_path
        self.test_path = test_path
        self.maxstep = maxstep
        self.numofques = numofques

    def getData(self, file_path):
        data = []
        with open(file_path, 'r') as file:
            for len_seq, ques, ans in itertools.zip_longest(*[file] * 3):
                len_seq = int(len_seq.strip().strip(','))
                ques = [int(q) for q in ques.strip().strip(',').split(',')]
                ans = [int(a) for a in ans.strip().strip(',').split(',')]

                if len(ques) < self.maxstep:
                    # ques = ques + [0] * (self.maxstep - len(ques))
                    ques = [0] * (self.maxstep - len(ques)) + ques
                    # ans = ans + [0] * (self.maxstep - len(ans))
                    ans = [0] * (self.maxstep - len(ans)) + ans

                data.append((len_seq, ques, ans))

        dataset = CustomDataset(data, self.numofques, self.maxstep)

        # slices = len//self.maxstep + (1 if len % self.maxstep > 0 else 0)
        # # print(slices)
        # for i in range(slices):
        #     temp = np.zeros(shape=[self.maxstep, 2 * self.numofques])
        #     print(".", end="")
        #
        #     if len > 0:
        #         if len >= self.maxstep:
        #             steps = self.maxstep
        #         else:
        #             steps = len
        #
        #         for j in range(steps):
        #             if ans[i*self.maxstep + j] == 1:
        #                 temp[j][ques[i*self.maxstep + j]] = 1
        #             else:
        #                 temp[j][ques[i*self.maxstep + j] + self.numofques] = 1
        #         len = len - self.maxstep
        #     data.append(temp)
        #     print(sys.getsizeof(data))
        #     del temp

        return dataset

    def getTrainData(self):
        print('loading train data...')
        trainData = self.getData(self.train_path)
        return trainData

    def getTestData(self):
        print('loading test data...')
        testData = self.getData(self.test_path)
        return testData
