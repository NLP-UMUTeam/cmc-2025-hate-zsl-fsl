import argparse
import json
import torch
import random
import os

import pandas as pd
import numpy as np

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error
from transformers import DataCollatorWithPadding

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SEED = 42

# Set seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

parser = argparse.ArgumentParser()

parser.add_argument('-m', type=int, help='Model to evaluate', required=True)
parser.add_argument('-t', type=int, help='Task to evaluate', required=True)

args = parser.parse_args()

if args.m == 0:
    model_path = "PlanTL-GOB-ES/roberta-base-bne" # MARIA
    local_path = "/data/tomas/models/maria"
    model_name = "maria_ft"
elif args.m == 1:
    model_path = "dccuchile/bert-base-spanish-wwm-uncased" # BETO UNCASED
    local_path = "/data/tomas/models/beto_un"
    model_name = "beto_un_ft"
elif args.m == 2:
    model_path = "dccuchile/bert-base-spanish-wwm-cased" # BETO CASED
    local_path = "/data/tomas/models/beto_ca"
    model_name = "beto_ca_ft"
elif args.m == 3:
    model_path = "bertin-project/bertin-roberta-base-spanish" # BERTIN
    local_path = "/data/tomas/models/bertin"
    model_name = "bertin_ft"
elif args.m == 4:
    model_path = "dccuchile/albert-base-spanish" # ALBETO
    local_path = "/data/tomas/models/albeto"
    model_name = "albeto_ft"
elif args.m == 5:
    model_path = "dccuchile/distilbert-base-spanish-uncased" # DISTILBETO
    local_path = "/data/tomas/models/distilbeto"
    model_name = "distilbeto_ft"

if args.t == 0:
    task_name = "label"

    num_labels = 4

    id2label = {0: "none", 1: "hate", 2: "hope", 3: "offensive"}
    label2id = {"none": 0, "hate": 1, "hope": 2, "offensive": 3}

elif args.t == 1:
    task_name = "target"

    num_labels = 3

    id2label = {0: "none", 1: "individual", 2: "group"}
    label2id = {"none": 0, "individual": 1, "group": 2}

elif args.t == 2:
    task_name = "intensity"

    num_labels = 7

    id2label = {0: "none", 1: "1-disagreement", 2: "2-negative-actions-poor-treatment", 3: "3-negative-character-insults", 4: "4-demonizing", 5: "5-violence", 6: "6-death"}
    label2id = {"none": 0, "1-disagreement": 1, "2-negative-actions-poor-treatment": 2, "3-negative-character-insults": 3, "4-demonizing": 4, "5-violence": 5, "6-death": 6}

elif args.t == 3:
    task_name = "group"

    num_labels = 9

    id2label = {0: "safe", 1: "homophoby-related", 2: "misogyny-related", 3: "racism-related", 4: "fatphobia-related", 5: "transphoby-related", 6: "profession-related", 7: "disablism-related", 8: "aporophobia-related"}
    label2id = {"safe": 0, "homophoby-related": 1, "misogyny-related": 2, "racism-related": 3, "fatphobia-related": 4, "transphoby-related": 5, "profession-related": 6, "disablism-related": 7, "aporophobia-related": 8}

data = pd.read_csv("/data/tomas/hate/dataset.csv")

if args.t != 3:
    data.loc[:, task_name] = data[task_name].map(label2id.get)
else:
    def multilabel_encode(label_str):

        binary_vector = [0] * num_labels

        if pd.isna(label_str) or label_str.strip() == "":
            return binary_vector
        
        labels = label_str.split("; ")

        for label in labels:
            label = label.strip()
            if label in label2id:
                binary_vector[label2id[label]] = 1
        return binary_vector

    data[task_name] = data[task_name].apply(multilabel_encode)

data_train = data[data['__split'] == 'train']
data_eval = data[data['__split'] == 'val']
data_test = data[data['__split'] == 'test']

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if args.t == 3:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        else:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item

# Charging model and tokenizer

def model_init():
    
    if args.t == 3:
        model = AutoModelForSequenceClassification.from_pretrained(local_path, 
                                                                num_labels=num_labels,
                                                                problem_type="multi_label_classification")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(local_path, 
                                                                num_labels=num_labels,
                                                                problem_type="single_label_classification")
        
    model.config.id2label = id2label
    model.config.label2id = label2id
    
    return model


tokenizer = AutoTokenizer.from_pretrained(local_path)

df_train = TextDataset(data_train["tweet_clean_lowercase"].tolist(), data_train[task_name].tolist(), tokenizer)
df_eval = TextDataset(data_eval["tweet_clean_lowercase"].tolist(), data_eval[task_name].tolist(), tokenizer)
df_test = TextDataset(data_test["tweet_clean_lowercase"].tolist(), data_test[task_name].tolist(), tokenizer)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(p):

    if args.t == 3:
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.tensor(p.predictions))
        preds = (probs > 0.5).int().numpy()
        labels = p.label_ids
    else:
        preds = p.predictions.argmax(axis=-1)
        labels = p.label_ids

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "mae": mean_absolute_error(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0)
    }

    f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
    precision_per_class = precision_score(labels, preds, average=None, zero_division=0)
    recall_per_class = recall_score(labels, preds, average=None, zero_division=0)

    for i, (f1, prec, rec) in enumerate(zip(f1_per_class, precision_per_class, recall_per_class)):
        metrics[f"f1_class_{i}"] = float(f1)
        metrics[f"precision_class_{i}"] = float(prec)
        metrics[f"recall_class_{i}"] = float(rec)

    return metrics

epochs = 15
batch_size = 16
lr = 3e-05
wd = 0.01

print(f"Entrenando {model_name}")


training_args = TrainingArguments(
    output_dir=f"./logs/" + model_name + "/" + task_name,    
    
    eval_strategy="epoch",  
    save_strategy="epoch", 
    num_train_epochs=epochs,    
    per_device_train_batch_size=batch_size,  
    per_device_eval_batch_size=batch_size,
    learning_rate=lr,            
    weight_decay=wd,               

    metric_for_best_model="f1_macro",
    save_total_limit=1,

    seed=SEED,
    report_to="none"
)


trainer = Trainer(
    model=model_init(),
    tokenizer=tokenizer,
    args=training_args,              
    train_dataset=df_train,
    eval_dataset=df_eval,
    data_collator=data_collator,
    compute_metrics=compute_metrics 
)

# Fine-tuning
trainer.train()

predictions = trainer.predict(df_test)

if args.t == 3:
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.tensor(predictions.predictions))
    preds = (probs > 0.5).int().numpy()

    df_preds = pd.DataFrame(preds, columns=[id2label[i] for i in range(num_labels)])
else:

    preds = predictions.predictions.argmax(axis=-1)

    df_preds = pd.DataFrame({
        "PredictedLabel": preds.tolist()
    })


df_preds.to_csv(f"/data/tomas/hate/results/ft/{model_name}_{task_name}_predictions.csv", index=False)

metrics = compute_metrics(predictions)

with open("/data/tomas/hate/results/ft/" + model_name + "_" + task_name + "_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("fin ft de " + model_name)