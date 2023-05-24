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

from helper import make_train_test, clean_price, seed_everything
from options import args



def assign_cat(category):
    if category ==  "Beauty & Personal Care":
        return 1,0,0,0
    if category ==  "Baby Products":
        return 0,1,0,0
    if category ==  "Health & Household":
        return 0,0,1,0
    if category ==  "Home & Kitchen":
        return 0,0,0,1

def create_cat(sd):
    cat = sd['train']['category']
    bea, ba, hou, kit = [], [], [], []
    for ca in cat:
        a, b, c, d = assign_cat(ca)
        bea.append(a)
        ba.append(b)
        hou.append(c)
        kit.append(d)
    sd['train'] = sd['train'].add_column("bea", bea)
    sd['train'] = sd['train'].add_column("ba", ba)
    sd['train'] = sd['train'].add_column("hou", hou)
    sd['train'] = sd['train'].add_column("kit", kit)

    cat = sd['test']['category']
    bea, ba, hou, kit = [], [], [], []
    for ca in cat:
        a, b, c, d = assign_cat(ca)
        bea.append(a)
        ba.append(b)
        hou.append(c)
        kit.append(d)
    sd['test'] = sd['test'].add_column("bea", bea)
    sd['test'] = sd['test'].add_column("ba", ba)
    sd['test'] = sd['test'].add_column("hou", hou)
    sd['test'] = sd['test'].add_column("kit", kit)


def preprocess(args):

    def preprocess_function(examples):

        return tokenizer(examples["description"], padding=True, truncation=True)

    def preprocess_function2(examples):

        return ro_tokenizer(examples["des_sentence"], padding='max_length', truncation=True, max_length=256)

    def preprocess_function3(examples):

        return tokenizer(examples["reviews"], padding=True, truncation=True)

    df = pd.read_csv(args.data_path)
    df1 = pd.read_csv(args.test_path)

    #tfidf = np.load(args.tfidf_path)

    df['rating'] = df['rating'].fillna(0)
    df1['rating'] = df1['rating'].fillna(0)

    seed_everything(args.random_seed)

    df['price'] = df['price'].apply(clean_price)
    df1['price'] = df1['price'].apply(clean_price)

    X_train = df[['price', 'rating', 'category', 'reviews', 'description', 'des_sentence']]
    y_train = df['finch score']

    X_test = df1[['price', 'rating', 'category', 'reviews', 'description', 'des_sentence']]
    y_test = df1['finch score']

    X_train['labels'] = y_train
    X_test['labels'] = y_test


    for i, row in X_train.iterrows():
        if type(row['reviews']) != str:
            X_train['reviews'][i] = ""

    for i, row in X_test.iterrows():
        if type(row['reviews']) != str:
            X_test['reviews'][i] = ""

    train_set = Dataset.from_pandas(X_train, preserve_index=False)
    val_set = Dataset.from_pandas(X_test, preserve_index=False)

    split_datasets = DatasetDict({"train": train_set, "test": val_set})

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    ro_tokenizer = AutoTokenizer.from_pretrained(args.ro_model_name)

    tokenized_dataset = split_datasets.map(preprocess_function2, batched=True)
    tokenized_dataset = tokenized_dataset.rename_column("input_ids", "input_ids0")
    tokenized_dataset = tokenized_dataset.rename_column("attention_mask", "attention_mask0")

    tokenized_dataset = tokenized_dataset.map(preprocess_function3, batched=True)
    tokenized_dataset = tokenized_dataset.rename_column("input_ids", "input_ids1")
    tokenized_dataset = tokenized_dataset.rename_column("attention_mask", "attention_mask1")

    tokenized_dataset = tokenized_dataset.map(preprocess_function, batched=True)

    create_cat(tokenized_dataset)  ##################

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    return tokenized_dataset, data_collator, 0, tokenizer, ro_tokenizer



def preprocess_cv(args, fold):

    def preprocess_function(examples):

        return tokenizer(examples["description"], padding=True, truncation=True)

    def preprocess_function2(examples):

        return ro_tokenizer(examples["des_sentence"], padding='max_length', truncation=True, max_length=256)

    def preprocess_function3(examples):

        return tokenizer(examples["reviews"], padding=True, truncation=True)

    df_ = pd.read_csv(args.data_path)
    df = df_[df_["fold"] != fold].reset_index(drop=True)
    df1 = df_[df_["fold"] == fold].reset_index(drop=True)

    #tfidf = np.load(args.tfidf_path)

    df['rating'] = df['rating'].fillna(0)
    df1['rating'] = df1['rating'].fillna(0)

    seed_everything(args.random_seed)

    df['price'] = df['price'].apply(clean_price)
    df1['price'] = df1['price'].apply(clean_price)

    X_train = df[['price', 'rating', 'category', 'reviews', 'description', 'des_sentence']]
    y_train = df['finch score']

    X_test = df1[['price', 'rating', 'category', 'reviews', 'description', 'des_sentence']]
    y_test = df1['finch score']

    X_train['labels'] = y_train
    X_test['labels'] = y_test


    for i, row in X_train.iterrows():
        if type(row['reviews']) != str:
            X_train['reviews'][i] = ""

    for i, row in X_test.iterrows():
        if type(row['reviews']) != str:
            X_test['reviews'][i] = ""

    train_set = Dataset.from_pandas(X_train, preserve_index=False)
    val_set = Dataset.from_pandas(X_test, preserve_index=False)

    split_datasets = DatasetDict({"train": train_set, "test": val_set})

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    ro_tokenizer = AutoTokenizer.from_pretrained(args.ro_model_name)

    tokenized_dataset = split_datasets.map(preprocess_function2, batched=True)
    tokenized_dataset = tokenized_dataset.rename_column("input_ids", "input_ids0")
    tokenized_dataset = tokenized_dataset.rename_column("attention_mask", "attention_mask0")

    tokenized_dataset = tokenized_dataset.map(preprocess_function3, batched=True)
    tokenized_dataset = tokenized_dataset.rename_column("input_ids", "input_ids1")
    tokenized_dataset = tokenized_dataset.rename_column("attention_mask", "attention_mask1")

    tokenized_dataset = tokenized_dataset.map(preprocess_function, batched=True)

    create_cat(tokenized_dataset)  ##################

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    return tokenized_dataset, data_collator, 0, tokenizer, ro_tokenizer
