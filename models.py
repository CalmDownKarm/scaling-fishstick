import torch
import random
import torch.nn as nn
from typing import List
import lightning.pytorch as pl
import torch.utils.data as data
import torch.nn.functional as F







class LSTMAttentionLM(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 embedding_size: int,
                 hidden_size: int,
                 n_layers: int,
                 pad_token: int = 22,
                 hidden_p: float = 0.2,
                 num_heads: int = 2):
        super().__init__()
        # self.bs = 1
        self.embedding_layer = nn.Embedding(vocab_size,
                                            embedding_size,
                                            padding_idx=pad_token)
        self.rnn = nn.LSTM(input_size=embedding_size,
                           hidden_size=hidden_size,
                           batch_first=True,
                           bidirectional=True,
                           dropout=hidden_p,
                           num_layers=n_layers)
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads)
        self.fc = nn.Linear(hidden_size * 4, vocab_size)

    def forward(self, x):
        # x should be batch x bptt x emb_size
        embedding = self.embedding_layer(x)
        rnn_output, _ = self.rnn(embedding)
        context, _ = self.attention(rnn_output.transpose(0, 1),
                                    rnn_output.transpose(0, 1),
                                    rnn_output.transpose(0, 1))
        attended_output = torch.cat(
            [rnn_output, context.transpose(0, 1)], dim=-1)
        output = self.fc(attended_output)
        return output


class LM(pl.LightningModule):

    def __init__(self,
                 embedding_size: int,
                 vocab_size: int,
                 hidden_size: int,
                 n_layers: int,
                 num_heads: int,
                 hidden_p: float = 0.2,
                 learning_rate: float = 1e-5):
        super().__init__()
        self.learning_rate = learning_rate
        self.lm = LSTMAttentionLM(vocab_size=vocab_size,
                                  embedding_size=embedding_size,
                                  hidden_size=hidden_size,
                                  n_layers=n_layers,
                                  num_heads=num_heads,
                                  hidden_p=hidden_p)
        self.save_hyperparameters()

    def do_step(self, batch, batch_idx):
        x, y = batch
        z = self.lm(x)
        loss = F.cross_entropy(z.view(-1, z.size(-1)), y.view(-1))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.do_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        test_loss = self.do_step(batch, batch_idx)
        self.log("test_loss", test_loss)

    def validation_step(self, batch, batch_idx):
        val_loss = self.do_step(batch, batch_idx)
        self.log("val_loss", val_loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

class Regression(nn.Module):

    def __init__(self, LM: pl.LightningModule, hidden_size, dropout):
        super().__init__()
        self.embedding_layer = LM.embedding_layer
        self.rnn = LM.rnn
        self.attention = LM.attention
        self.fc = nn.Sequential(nn.BatchNorm1d(hidden_size * 4),
                                nn.Dropout(dropout),
                                nn.Linear(hidden_size * 4, hidden_size * 2),
                                nn.ReLU(), nn.BatchNorm1d(hidden_size * 2),
                                nn.Dropout(dropout),
                                nn.Linear(hidden_size * 2, hidden_size),
                                nn.ReLU(), nn.BatchNorm1d(hidden_size),
                                nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, x):
        embedding = self.embedding_layer(x)
        rnn_output, _ = self.rnn(embedding)
        context, _ = self.attention(rnn_output.transpose(0, 1),
                                    rnn_output.transpose(0, 1),
                                    rnn_output.transpose(0, 1))

        attended_output = torch.cat(
            [rnn_output, context.transpose(0, 1)], dim=-1)
        encoder_output = attended_output[:, -1, :]
        output = self.fc(encoder_output)
        return output


class ConditionalLM(pl.LightningModule):

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 hidden_size,
                 num_layers,
                 num_heads,
                 embedding_layer=None,
                 dropout=0.5,
                 learning_rate=1e-5):
        super().__init__()
        self.learning_rate = learning_rate
        if embedding_layer:
            self.embedding = embedding_layer
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(input_size=embedding_size + 1,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=dropout)
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads)
        self.fc = nn.Sequential(nn.Linear(hidden_size * 4, hidden_size * 2),
                                nn.ReLU(),
                                nn.Linear(hidden_size * 2, hidden_size),
                                nn.ReLU(), nn.Linear(hidden_size, vocab_size))
        self.save_hyperparameters(ignore=["embedding_layer"])

    def forward(self, x, activity):

        # expand the real number to have the same length as the input sequence

        # apply the embedding layer to the input sequence and concatenate with the real number
        embedded = self.embedding(x)

        batch_size, sequence_length, emb_dim = embedded.size()
        activity = activity.repeat(sequence_length,
                                   1).view(batch_size, sequence_length, -1)
        embedded = torch.cat([embedded, activity], dim=-1)
        # apply the RNN layer
        rnn_output, _ = self.rnn(embedded)

        # apply multi-headed attention to the RNN output
        context, _ = self.attention(rnn_output.transpose(0, 1),
                                    rnn_output.transpose(0, 1),
                                    rnn_output.transpose(0, 1))
        attended_output = torch.cat(
            [rnn_output, context.transpose(0, 1)], dim=-1)
        # print(attended_output.shape)
        # apply the fully connected layers
        output = self.fc(attended_output)
        output = nn.functional.softmax(output, dim=-1)

        return output

    def do_step(self, batch):
        input_seq, activity, target_seq = batch
        output_seq = self.forward(input_seq, activity)
        # z = self.lm(x)
        loss = F.cross_entropy(output_seq.view(-1, output_seq.size(-1)),
                               target_seq.view(-1))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.do_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        test_loss = self.do_step(batch)
        self.log("test_loss", test_loss)

    def validation_step(self, batch, batch_idx):
        val_loss = self.do_step(batch)
        self.log("val_loss", val_loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

class Regressor(pl.LightningModule):
    def __init__(self, lm, learning_rate=1e-4, hidden_size=300, dropout=0.5):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = Regression(lm, hidden_size=hidden_size, dropout=dropout)
        self.save_hyperparameters(ignore=['lm'])
        # self.save_hyperparameters()
    def do_step(self, batch):
        x = batch["text"]
        y = batch["label"]
        z = self.model(x).flatten()
        loss = F.mse_loss(z, y)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.do_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    def test_step(self, batch, batch_idx):
        test_loss = self.do_step(batch)
        self.log("test_loss", test_loss)
    
    def validation_step(self, batch, batch_idx):
        val_loss = self.do_step(batch)
        self.log("val_loss", val_loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        
