import os
import random
import logging
import torch

import torch.optim as optim
import numpy as np

from datetime import datetime
from docopt import docopt
from dataloader import getDataLoader
import eval

from RNNModel import RNNModel
from SAKT.model import SAKTModel


def setup_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    length = 200
    questions = 2829
    lr = 0.001
    bs = 48
    seed = 42
    epochs = 5
    use_cuda = 1
    hidden = 256
    layers = 1
    heads = 8
    dropout = 0.1
    model_type = 'RNN'  # SAKT / RNN

    logger = logging.getLogger('main')
    logger.setLevel(level=logging.DEBUG)
    date = datetime.now()
    handler = logging.FileHandler(
        f'log/{date.year}_{date.month}_{date.day}_{model_type}_result.log')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # logger.info(list(args.items()))

    setup_seed(seed)

    if torch.cuda.is_available() and use_cuda:
        # os.environ["CUDA_VISIBLE_DEVICES"] = cuda
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("device: ", device)

    trainLoader, testLoade = getDataLoader(bs, questions, length)
    
    if model_type == 'RNN':
        model = RNNModel(questions * 2, hidden, layers, questions, device)
    elif model_type == 'SAKT':
        model = SAKTModel(heads, length, hidden, questions, dropout)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = eval.lossFunc(questions, length, device)

    for epoch in range(epochs):
        print('epoch: ' + str(epoch))
        model, optimizer = eval.train_epoch(model, trainLoader, optimizer,
                                          loss_func, device, epoch_number=epoch)
        logger.info(f'epoch {epoch}')
        eval.test_epoch(model, testLoade, loss_func, device)


if __name__ == '__main__':
    main()
