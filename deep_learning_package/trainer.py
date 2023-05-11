import pandas as pd
from tqdm import tqdm
import numpy as np
import json
import torch
import transformers
import random
import os
from torch import nn
from sklearn.model_selection import train_test_split

from datasets import load_dataset, Dataset, load_metric, DatasetDict

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding, EarlyStoppingCallback
from transformers import DistilBertPreTrainedModel, DistilBertConfig
from transformers import DistilBertForSequenceClassification, AutoConfig, AutoModel, AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

def compute_metrics(eval_preds):
    metric = load_metric('mse')
    logits, labels = eval_preds
    d = metric.compute(predictions=logits, references=labels)
    d["eval_Mse"] = d['mse']
    return d

class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss = outputs.get("loss")
        #print(logits, labels)
        return (loss, outputs) if return_outputs else loss