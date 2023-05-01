import os
import time
import openai
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from typing import List, Dict
from collections import defaultdict
from sklearn.model_selection import train_test_split
from utils import get_exact_accuracies, get_loose_accuracies

openai.api_key = os.getenv("OPENAI_API_KEY")


def prompt_design(samples: pd.DataFrame, act_to_predict: float) -> List[Dict]:
    messages = [{
        "role":
        "system",
        "content":
        """You are a protein-sequence designer, you're presented with a sequence of proteins composed of 12 amino acids, these amino acids relate to a specific real valued activity which has mean of 0 and std 1. 
        Given an activity, predict the sequence"""
    }]
    for _, (
            seq,
            activity,
    ) in samples.iterrows():
        messages.append({
            "role": "user",
            "content": f"ACTIVITY: {round(activity, 3)}"
        })
        messages.append({"role": "assistant", "content": f"SEQ: {seq}"})
    messages.append({
        "role": "user",
        "content": f"ACTIVITY: {round(act_to_predict, 3)}"
    })
    return messages



if __name__ == "main":
    NUM_SAMPLES_TO_PREDICT = 1000
    NUM_EXEMPLARS = 10
    data_path = Path() / "data"
    pred_path = Path() / "preds"
    train_set = pd.read_csv(data_path / "train.csv")
    train_set, valid_set = train_test_split(train_set,
                                            train_size=0.8,
                                            shuffle=True,
                                            random_state=42)
    preds = []
    valid_subset = valid_set.sample(NUM_SAMPLES_TO_PREDICT)
    valid_activity = valid_subset.activity.tolist()
    for activity in tqdm(valid_activity):
        try:
            samples = train_set.sample(NUM_EXEMPLARS)
            response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                    messages=prompt_design(
                                                        samples, activity))
            preds.append(response["choices"][0]["message"]["content"].split(
                ": ")[1].strip())
        except IndexError:
            preds.append("")
        except Exception:
            time.sleep(1)
            preds.append("")
        time.sleep(0.1)
    valid_subset["predicted_sequence"] = preds
    valid_subset.to_csv(pred_path / "predicted_sequences_chatgpt.csv")
    print(
        f"Strict Accuracy Metric {get_exact_accuracies(valid_subset.seq.tolist(), preds)}"
    )
    print(
        f"Loose Token Accuracy {get_loose_accuracies(valid_set, preds, valid_activity)}"
    )
