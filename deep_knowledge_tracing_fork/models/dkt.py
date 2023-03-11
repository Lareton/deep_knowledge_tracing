import os
from tqdm import tqdm
import numpy as np
import torch
import wandb

from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from torch.nn.functional import one_hot, binary_cross_entropy
from sklearn import metrics


class DKT(Module):
    '''
        Args:
            num_q: the total number of the questions(KCs) in the given dataset
            emb_size: the dimension of the embedding vectors in this model
            hidden_size: the dimension of the hidden vectors in this model
    '''
    def __init__(self, num_q, emb_size, hidden_size):
        super().__init__()
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = hidden_size

        self.interaction_emb = Embedding(self.num_q * 2, self.emb_size)
        self.lstm_layer = LSTM(
            self.emb_size, self.hidden_size, batch_first=True
        )
        self.out_layer = Linear(self.hidden_size, self.num_q)
        self.dropout_layer = Dropout()

    def forward(self, q, r):
        '''
            Args:
                q: the question(KC) sequence with the size of [batch_size, n]
                r: the response sequence with the size of [batch_size, n]

            Returns:
                y: the knowledge level about the all questions(KCs)
        '''
        x = q + self.num_q * r

        h, _ = self.lstm_layer(self.interaction_emb(x))
        y = self.out_layer(h)
        y = self.dropout_layer(y)
        y = torch.sigmoid(y)

        return y

    def train_model(
        self, train_loader, test_loader, num_epochs, opt, ckpt_path
    ):
        '''
            Args:
                train_loader: the PyTorch DataLoader instance for training
                test_loader: the PyTorch DataLoader instance for test
                num_epochs: the number of epochs
                opt: the optimization to train this model
                ckpt_path: the path to save this model's parameters
        '''
        aucs = []
        loss_means = []

        max_auc = 0

        for epoch in range(1, num_epochs + 1):
            loss_mean = []
            train_losses = []

            for ind_step, data in enumerate(train_loader):
                q, r, qshft, rshft, m = data

                self.train()

                y = self(q.long(), r.long())
                y = (y * one_hot(qshft.long(), self.num_q)).sum(-1)

                y = torch.masked_select(y, m)
                t = torch.masked_select(rshft, m)

                opt.zero_grad()
                loss = binary_cross_entropy(y, t)
                loss.backward()
                opt.step()

                loss_mean.append(loss.detach().cpu().numpy())

                cur_loss = loss.detach().cpu().numpy()
                train_losses.append(cur_loss)
                wandb.log({
                    "train_loss":  cur_loss,
                    "loss_step": epoch * len(train_loader) + ind_step}
                )

            wandb.log({
                "train_loss_avg_epoch":  np.mean(train_losses),
                "epoch_num": epoch
            })


            with torch.no_grad():
                for data in tqdm(test_loader):
                    q, r, qshft, rshft, m = data

                    self.eval()

                    y = self(q.long(), r.long())
                    y = (y * one_hot(qshft.long(), self.num_q)).sum(-1)

                    y = torch.masked_select(y, m).detach().cpu()
                    t = torch.masked_select(rshft, m).detach().cpu()

                    auc = metrics.roc_auc_score(
                        y_true=t.numpy(), y_score=y.numpy()
                    )

                    accuracy = metrics.accuracy_score(
                        y_true=(t.numpy() > 0.5).astype("int32"),
                        y_pred=(y.numpy() > 0.5).astype("int32")
                    )

                    loss_mean = np.mean(loss_mean)

                    # print(
                    #     "Epoch: {},   AUC: {},   Loss Mean: {}"
                    #     .format(i, auc, loss_mean)
                    # )

                    if auc > max_auc:
                        torch.save(
                            self.state_dict(),
                            os.path.join(
                                ckpt_path, "model.ckpt"
                            )
                        )

                        torch.save(
                            self,
                            os.path.join(
                                ckpt_path, "model_full.ckpt"
                            )
                        )

                        max_auc = auc

                    aucs.append(auc)
                    loss_means.append(loss_mean)

                    wandb.log({
                        "valid_auc": auc,
                        "valid_acc": accuracy,
                        "epoch_valid": epoch
                    })


        return aucs, loss_means
