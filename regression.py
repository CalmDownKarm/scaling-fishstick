import torch
import random
import numpy as np
import pandas as pd
from typing import List
from pathlib import Path
from tqdm.auto import tqdm
import lightning.pytorch as pl
from collections import Counter
import torch.utils.data as data
from models import LM, Regressor
from utils import preprocess_text, train_cycle
from lightning.pytorch.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class LMDataset(data.Dataset):

    def __init__(self,
                 sequences: List,
                 bs: int = 1,
                 bptt: int = 14,
                 shuffle: bool = False):
        self.sequences = sequences
        self.bs = bs
        self.bptt = bptt
        self.shuffle = shuffle
        total_len = sum([len(t) for t in sequences])
        self.n_batches = total_len // bs
        self.shuffle_create_batches()

    def __len__(self):
        return ((self.n_batches - 1) // self.bptt) * self.bs

    def __getitem__(self, index: int):
        subset_batched_data = self.batched_data[index % self.bs]
        sequence_index = (index // self.bs) * self.bptt
        return subset_batched_data[sequence_index:sequence_index +
                                   self.bptt], subset_batched_data[
                                       sequence_index + 1:sequence_index +
                                       self.bptt + 1]

    def shuffle_create_batches(self):
        if self.shuffle:
            random.shuffle(self.sequences)
        all_text = torch.cat([torch.tensor(t) for t in self.sequences])
        self.batched_data = all_text[:self.n_batches * self.bs].view(
            self.bs, self.n_batches)


class RegDataset(data.Dataset):

    def __init__(self, df, train_tok_count, is_test=False):
        self.sequences, _ = preprocess_text(df["seq"], train_tok_count)
        self.is_test = is_test
        if not is_test:
            self.labels = df["activity"].tolist()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        if not self.is_test:
            label = self.labels[idx]
            return {
                "text": torch.tensor(sequence),
                "label": torch.tensor(label)
            }
        return {"text": torch.tensor(sequence)}


def run_inference_calculate_preds():
    preds = []
    targets = []
    reg_model.eval()
    with torch.no_grad():
        for (idx, batch) in enumerate(test_loader):
            x = batch["text"]
            y = batch["label"]
            z = reg_model.model(x).flatten()
            preds.append(z)
            targets.append(y)
    preds = torch.cat(preds).cpu().numpy()
    targets = torch.cat(targets).cpu().numpy()
    print(f"MAE : {mean_absolute_error(targets, preds)}")
    print(f"RMSE: {mean_squared_error(targets, preds, squared=False)}")
    print(f"MSE: {mean_squared_error(targets, preds, squared=True)}")
    print(f"R2: {r2_score(targets, preds)}")
    print(np.corrcoef(targets, preds))


def pred_write_csv(model, test_set):
    final_test_set = RegDataset(test_set, train_tok_count, is_test=True)
    model.eval()
    test_preds = []
    with torch.no_grad():
        for sample in tqdm(final_test_set):
            test_preds.append(
                reg_model.model(sample["text"].view(1, -1)).flatten())
    test_preds = torch.cat(test_preds).tolist()
    test_set["pred_activity"] = test_preds


if __name__ == "__main__":
    BPTT = 6
    EMBEDDING_SIZE = 300
    HIDDEN_SIZE = 300
    N_LAYERS = 2
    NUM_HEADS = 4
    HIDDEN_P = 0.5
    BATCHSIZE = 256
    DATALOADER_NUM_WORKERS = 10
    # Load Data
    data_path = Path() / "data"
    train_set = pd.read_csv(data_path / "train.csv")
    test_set = pd.read_csv(data_path / "test.csv")
    train_tok_count = Counter()
    train_set["seq"].apply(lambda x: train_tok_count.update(list(x)))
    sequences, tok_to_index_train = preprocess_text(train_set.seq,
                                                    train_tok_count)
    # Prep data for Language Modelling
    random.shuffle(sequences)
    train_sequences = sequences[:90000]
    test_sequences = sequences[90000:]
    train_dataset = LMDataset(train_sequences, bptt=BPTT, shuffle=True)
    test_dataset = LMDataset(test_sequences, shuffle=True)
    lm = LM(embedding_size=EMBEDDING_SIZE,
            vocab_size=len(tok_to_index_train.keys()),
            hidden_size=HIDDEN_SIZE,
            n_layers=N_LAYERS,
            num_heads=NUM_HEADS,
            hidden_p=HIDDEN_P)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=BATCHSIZE,
                                   num_workers=DATALOADER_NUM_WORKERS,
                                   shuffle=True)
    test_loader = data.DataLoader(test_dataset,
                                  batch_size=BATCHSIZE,
                                  num_workers=DATALOADER_NUM_WORKERS)
    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", patience=5),
        ModelCheckpoint(save_top_k=1,
                        save_on_train_epoch_end=True,
                        monitor="val_loss",
                        mode="min",
                        dirpath=".")
    ]
    wb_logger = WandbLogger(project="Protein Sequence Activation LM",
                            log_model="all")
    train_cycle_kwargs = {
        "model": lm,
        "train_dataloader": train_loader,
        "val_dataloader": test_loader,
        "max_epochs": 5,
        "callbacks": callbacks,
        "do_tune": True,
        "reload_dataloaders_every_n_epochs": 1,
        "logger": wb_logger,
        "gradient_clip_val": 0.7,
        "save_checkpoint_path": None,
        "learning_rate": 1e-4
    }
    lm = train_cycle(**train_cycle_kwargs)

    regtrain, regvalid = train_test_split(train_set,
                                          train_size=0.8,
                                          random_state=42)
    reg_model = Regressor(lm=lm.lm, hidden_size=HIDDEN_SIZE, dropout=HIDDEN_P)
    for layer in [
            reg_model.model.embedding_layer, reg_model.model.rnn,
            reg_model.model.attention
    ]:
        for param in layer.parameters():
            param.requires_grad = False

    reg_train_ds = RegDataset(regtrain, train_tok_count)
    reg_valid_ds = RegDataset(regvalid, train_tok_count)
    train_loader = torch.utils.data.DataLoader(reg_train_ds,
                                               batch_size=BATCHSIZE,
                                               num_workers=10,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(reg_valid_ds,
                                              batch_size=BATCHSIZE,
                                              num_workers=10)
    wb_logger = WandbLogger(project="Protein Sequence Activation Regression",
                            log_model="all")
    train_cycle_kwargs = {
        "model": reg_model,
        "train_dataloader": train_loader,
        "val_dataloader": test_loader,
        "max_epochs": 5,
        "callbacks": callbacks,
        "do_tune": False,
        "reload_dataloaders_every_n_epochs": 0,
        "logger": wb_logger,
        "gradient_clip_val": 0.7,
        "save_checkpoint_path": None,
        "learning_rate": 1e-4
    }

    reg_model = train_cycle(**train_cycle_kwargs)
    for layer in [reg_model.model.attention]:
        for param in layer.parameters():
            param.requires_grad = True
    train_cycle_kwargs["max_epochs"] = 10,
    train_cycle_kwargs["do_tune"] = True
    reg_model = train_cycle(**train_cycle_kwargs)

    for layer in [
            reg_model.model.embedding_layer, reg_model.model.rnn,
            reg_model.model.attention
    ]:
        for param in layer.parameters():
            param.requires_grad = True
    train_cycle_kwargs["max_epochs"] = 40
    train_cycle_kwargs["save_checkpoint_path"] = "best_reg_model.ckpt"
    reg_model = train_cycle(**train_cycle_kwargs)
    run_inference_calculate_preds()
    pred_write_csv(reg_model, test_set=test_set)