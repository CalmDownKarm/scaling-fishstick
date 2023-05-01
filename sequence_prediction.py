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
from lightning.pytorch.loggers import WandbLogger
from models import LM, ConditionalLM
from sklearn.model_selection import train_test_split
from utils import preprocess_text, train_cycle, get_exact_accuracies, get_loose_accuracies
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint


class CLMDataset(data.Dataset):

    def __init__(self, sequences, activities, subsequence_length=6):
        self.sequences = sequences
        self.activities = activities
        self.max_length = 14
        self.subsequence_length = subsequence_length
        # Max index position can be 12 - subsequence_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        start_indx = random.randint(
            0, self.max_length - (self.subsequence_length + 2))
        sequence = self.sequences[index]
        activity = torch.tensor(self.activities[index])
        input = torch.LongTensor(sequence[start_indx:start_indx +
                                          self.subsequence_length])
        target = torch.LongTensor(sequence[start_indx + 1:start_indx +
                                           self.subsequence_length + 1])

        return input, activity, target


def predict_one(model, activity, tok_to_index_train):
    # Top K Sampling
    START_TOKEN = "^"
    EPSILON = 0.0001
    model.eval()
    with torch.no_grad():
        start_token_index = tok_to_index_train[START_TOKEN]
        generated_tokens = [start_token_index]
        activity = torch.tensor(activity).view(1, -1)
        for _ in range(12):
            activity += EPSILON
            x = torch.LongTensor(generated_tokens).view(1, -1)
            outputs = model.forward(x, activity)[:, -1, :].flatten()
            top_k_probs, top_k_indices = torch.topk(outputs, 5)
            top_k_probs /= torch.sum(top_k_probs)  # Normalize
            sampled_index = torch.multinomial(top_k_probs, 1).item()
            character = top_k_indices[sampled_index].item()
            generated_tokens.append(character)
        generated_tokens.pop(0)
        return generated_tokens


def decode_preds(preds, tok_to_index):
    index_to_tok = {v: k for k, v in tok_to_index.items()}
    decoded_preds = [
        "".join([index_to_tok[tok] for tok in pred]) for pred in preds
    ]
    return decoded_preds


if __name__ == "__main__":
    EMBEDDING_SIZE = 300
    HIDDEN_SIZE = 128
    N_LAYERS = 2
    NUM_HEADS = 4
    HIDDEN_P = 0.5
    BATCHSIZE = 256
    NUM_WORKERS = 10

    data_path = Path() / "data"
    train_set = pd.read_csv(data_path / "train.csv")
    train_tok_count = Counter()
    train_set["seq"].apply(lambda x: train_tok_count.update(list(x)))
    _, tok_to_index_train = preprocess_text(train_set["seq"], train_tok_count)

    # activities = train_set["activity"].tolist()
    train_subset, validation_subset = train_test_split(train_set,
                                                       train_size=0.8,
                                                       shuffle=True,
                                                       random_state=42)
    train_sequences, _ = preprocess_text(train_subset.seq, train_tok_count)
    test_sequences, _ = preprocess_text(validation_subset.seq, train_tok_count)
    train_activities = train_subset.activity.tolist()
    test_activities = validation_subset.activity.tolist()

    print(len(train_sequences), len(train_activities), len(test_sequences),
          len(test_activities))
    train_ds = CLMDataset(train_sequences, train_activities)
    test_ds = CLMDataset(test_sequences, test_activities)

    train_loader = torch.utils.data.DataLoader(train_ds,
                                               batch_size=BATCHSIZE,
                                               num_workers=NUM_WORKERS,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds,
                                              batch_size=BATCHSIZE,
                                              num_workers=NUM_WORKERS,
                                              shuffle=True)

    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", patience=5),
        ModelCheckpoint(save_top_k=1,
                        save_on_train_epoch_end=True,
                        monitor="val_loss",
                        mode="min",
                        dirpath=".")
    ]

    wb_logger = WandbLogger(project="Protein Sequence Prediction",
                            log_model="all")
    #Regression is the LM we trained for Task 1, parameters are not enumerated because they're independent of the current model parameters
    vocab_size = len(tok_to_index_train.keys())
    regression_lm = LM.load_from_checkpoint("best_lm_vocabfixed.ckpt",
                                            embedding_size=300,
                                            vocab_size=vocab_size,
                                            hidden_size=300,
                                            n_layers=2,
                                            num_heads=3,
                                            hidden_p=0.5)

    lm = ConditionalLM(embedding_size=EMBEDDING_SIZE,
                       vocab_size=vocab_size,
                       hidden_size=HIDDEN_SIZE,
                       num_layers=N_LAYERS,
                       num_heads=NUM_HEADS,
                       embedding_layer=regression_lm.lm.embedding_layer)

    # Freeze Embedding layer
    for parameter in lm.embedding.parameters():
        parameter.requires_grad = False
    # A note about repeated train_cycle loops
    # pytorch lightning requires that you set a max_epochs when you intialize a trainer object
    # once the max epochs has been hit, I prefer to create a new trainer object and track iterations.
    # Since the majority changes are to learning rate, these could definitely be reduced with a custom
    # learning rate scheduler.
    trainer_kwargs = {
        "model": lm,
        "max_epochs": 10,
        "callbacks": callbacks,
        "reload_dataloaders_ever_n_epochs": 1,
        "logger": wb_logger,
        "train_dataloader": train_loader,
        "val_dataloader": test_loader,
        "do_tune": True
        # "learning_rate":  1e-4
    }
    lm = train_cycle(**trainer_kwargs)
    trainer_kwargs["max_epochs"] = 50
    trainer_kwargs["gradient_clip_val"] = 5
    trainer_kwargs.pop("do_tune", None)
    trainer_kwargs["learning_rate"] = 1e-4
    lm = train_cycle(**trainer_kwargs)
    trainer_kwargs["max_epochs"] = 100
    trainer_kwargs["do_tune"] = True
    lm = train_cycle(**trainer_kwargs)
    trainer_kwargs.pop("do_tune", None)
    trainer_kwargs["save_checkpoint_path"] = "SEQUENCE_PREDICTION_LM_FINAL.ckpt"
    lm = train_cycle(**trainer_kwargs)

    print("-----MAKING PREDICTION------")
    # Generate sequences on the validation set
    preds = [predict_one(lm, activity, tok_to_index_train) for activity in tqdm(test_activities)]
    print(
        f"Exact Token Accuracy: {get_exact_accuracies(test_sequences, preds)}")
    decoded_preds = decode_preds(preds, tok_to_index_train)
    print(f"Loose Token Accuracy: {get_loose_accuracies(validation_subset, decoded_preds, test_activities)}")
    results_df = pd.DataFrame({
        "seq": validation_subset.seq.tolist(),
        "predicted_sequences": decoded_preds,
        "activity": test_activities
    }).to_csv("preds/predicted_sequences_rnn.csv")
