
import torch
import torch.utils.data as Data
from readdata import DataReader


def getDataLoader(batch_size, num_of_questions, max_step):
    handle = DataReader('text_dataset_train_new.txt',
                        'text_dataset_valid_new.txt',
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
