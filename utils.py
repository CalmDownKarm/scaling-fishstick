import pandas as pd
from collections import defaultdict
from typing import List
import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner

def preprocess_text(srs: pd.Series, train_tok_count):
    '''
    1. Sandwich each sequence with ^, $
    2. tokenize the sequences
    3. Numericalize the sequences
      '''
    sequences = pd.DataFrame(srs.apply(lambda x: list(f"^{x}$")).tolist())
    tok_to_index = {
        tok: indx
        for indx, tok in enumerate(train_tok_count.keys())
    }
    tok_to_index["^"] = 20  # Start token
    tok_to_index["$"] = 21  # Stop token
    tok_to_index["*"] = 22  # padding_index
    for col in sequences:
        sequences[col] = sequences[col].map(tok_to_index)
    return sequences.values.tolist(), tok_to_index


def train_cycle(model,
                train_dataloader,
                val_dataloader,
                max_epochs=5,
                callbacks=None,
                do_tune=False,
                reload_dataloaders_every_n_epochs=0,
                logger=None,
                gradient_clip_val=None,
                save_checkpoint_path=None,
                learning_rate=1e-4, **kwargs):
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        enable_progress_bar=True,
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
        logger=logger,
        gradient_clip_val=gradient_clip_val)
    if do_tune:
        tuner = Tuner(trainer)
        tuner.lr_find(model,
                      train_dataloaders=train_dataloader,
                      val_dataloaders=val_dataloader)
    else:
        model.learning_rate = learning_rate
    trainer.fit(model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)
    if save_checkpoint_path:
        trainer.save_checkpoint(save_checkpoint_path)
    return model

def get_token_accuracy(targets: List[str], pred: str) -> float:
    correct_tokens = 0
    for tok_indx, tok in enumerate(pred):
        if tok in [target[tok_indx] for target in targets]:
            correct_tokens += 1
    return correct_tokens / 12


def get_loose_accuracies(validation_set: pd.DataFrame, preds: List[str],
                         corresponding_activities: List[float]) -> float:
    # Corresponding activities are the activities that the predictions were made on
    test_sequences = validation_set.seq.tolist()
    rounded_test_activities = validation_set.activity.round(3).tolist()
    grouped_activity_sequence = defaultdict(list)

    for sequence, act in zip(test_sequences, rounded_test_activities):
        str_act = str(act)
        grouped_activity_sequence[str_act] += [sequence]

    token_accuracies = []
    for target_act, predicted in zip(corresponding_activities, preds):
        if len(predicted) > 1:
            # Chatgpt can predict sequences longer than 12
            predicted = predicted[:12]
            rounded_target_act = str(round(target_act, 3))  
            target_sequences = grouped_activity_sequence[rounded_target_act]
            token_accuracies.append(
                get_token_accuracy(target_sequences, predicted))
    return sum(token_accuracies) / len(token_accuracies)


def get_exact_accuracies(valid_targets: List[str], decoded_preds: List[str]) -> float:
    exact_accuracies = []
    for target, pred in zip(valid_targets, decoded_preds):
        correct_tokens = 0
        if len(pred) > 1:
            predicted = pred[:12]
        for t_tok, pred_tok in zip(target, predicted):
            if t_tok == pred_tok:
                correct_tokens += 1
        exact_accuracies.append(correct_tokens / 12)
    return sum(exact_accuracies) / len(exact_accuracies)