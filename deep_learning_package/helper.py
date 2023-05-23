import pandas as pd
import numpy as np
import torch
import random
import os

from sklearn.model_selection import train_test_split
def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



def clean_price(price):
    price = str(price)
    price = price.replace(',','')
    price = float(price)
    if pd.isnull(price):
        price = 0
    return price

def make_train_test(data, labels, test_size=0.1, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test