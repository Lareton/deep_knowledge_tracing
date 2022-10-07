import torch
import torch.nn as nn

from SAKT.attention import Encoder
from SAKT.embedding import Embedding

class SAKTModel(nn.Module):
    def __init__(self, h, length, d_model, n_question, dropout):
        super(SAKTModel, self).__init__()
        self.embedding = Embedding(n_question, length, d_model).to("cuda")
        self.encoder = Encoder(h, length, d_model, dropout).to("cuda")
        # self.w = nn.Linear(d_model, n_question).to("cuda")
        self.w = nn.Linear(d_model, n_question).to("cuda")
        self.sig = nn.Sigmoid()

    def forward(self, y):  # shape of input: [batch_size, length, questions * 2]
        y = y.to("cuda")
        x, y = self.embedding(y)  # shape: [batch_size, length, d_model]
        encode = self.encoder(x, y)  # shape: [batch_size, length, d_model]
        res = self.sig(self.w(encode))
        return res

