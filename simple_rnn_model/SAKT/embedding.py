import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, n_questions, length, embedding_dim):
        super().__init__()
        self.n_questions = n_questions
        self.x_emb = nn.Linear(n_questions, embedding_dim, bias=False).to("cuda")
        self.y_emb = nn.Linear(n_questions * 2, embedding_dim, bias=False).to("cuda")
        self.pos_emb = nn.Embedding(length, embedding_dim).to("cuda")
        self.length = length

    def forward(self, y):  # [batch_size, length, questions * 2]
        n_batch = y.shape[0]
        x = y[:, :, 0:self.n_questions] + y[:, :, self.n_questions:]
        p = torch.LongTensor([[i for i in range(self.length)] for j in range(n_batch)]).to("cuda")
        pos = self.pos_emb(p)
        y = self.y_emb(y)  # [batch_size, length, embedding_dim]
        x = self.x_emb(x)  # [batch_size, length, embedding_dim]
        return (x+pos, y)