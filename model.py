import koco

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')
import re 
from tqdm import tqdm
import math

import pandas as pd
import numpy as np
import json 

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import torch.nn as nn
from torchmetrics import Accuracy, F1Score

from transformers import BertModel, BertTokenizer, AdamW, get_cosine_schedule_with_warmup

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import matplotlib.pyplot as plt

import preprocess
import data_setup


class HateDataset(Dataset):
  def __init__(self, df, max_token_length = 80, model_name = 'klue/bert-base'):
    self.df = df
    self.max_token_length = max_token_length
    self.model_name = model_name
    self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
    self.data = df
  
  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    row = self.data.iloc[index]
    text = str(row.comments)
    label = row.hate

    output = self.tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=self.max_token_length,
        return_attention_mask = True,
        return_token_type_ids=False
    )
    
    input_ids = output['input_ids'].squeeze(0)
    attention_mask = output['attention_mask'].squeeze(0)

    return {
        'text' : text,
        'input_ids' : input_ids,
        'attention_mask' : attention_mask,
        'label' : label

    }
class HateDataModule(pl.LightningDataModule):
  def __init__(self, max_token_length: int = 90, batch_size: int = 32, model_name = 'klue/bert-base'):
    self.max_token_length = max_token_length
    self.batch_size = batch_size
    self.model_name = model_name
    self.prepare_data()
    self.prepare_data_per_node = True
    self._log_hyperparams = None
    self.allow_zero_length_dataloader_with_multiple_devices = False
  
  def prepare_data(self):
    train, valid, test = data_setup()
    self.train = HateDataset(train)
    self.valid = HateDataset(train)
    self.test = HateDataset(test)
  
  def setup(self, stage):
    if stage in (None, 'fit'):
      self.train
      self.valid
    
    if stage == 'test':
      self.test
  
  def train_dataloader(self):
    return DataLoader(self.train, batch_size = self.batch_size, num_workers=6, shuffle=True)

  def val_dataloader(self):
    return DataLoader(self.valid, batch_size = self.batch_size, num_workers=6, shuffle=False)

  def test_dataloader(self):
    return DataLoader(self.test, batch_size = self.batch_size, num_workers=6, shuffle=False)

class HateClassifier(pl.LightningModule):
  def __init__(self, config: dict, model_name = 'klue/bert-base'):
    super(HateClassifier, self).__init__()
    self.config = config
    self.model_name = model_name
    self.bert = BertModel.from_pretrained(self.model_name)
    self.classifier = nn.Linear(self.bert.config.hidden_size, config['n_classes'])
    self.dropout = nn.Dropout(p = 0.3)
    self.loss_func = nn.CrossEntropyLoss()
    self.f1 = F1Score(task="binary", num_classes= config['n_classes'])
    self.accuracy = Accuracy(task="binary", num_classes= config['n_classes'])
  
  def forward(self, input_ids, attention_mask, labels = None):
    output = self.bert(input_ids = input_ids, attention_mask = attention_mask)
    output = self.dropout(output.pooler_output)
    output = self.classifier(output)

    loss = 0
    if labels is not None:
        loss = self.loss_func(output, labels.long())
    return loss, output

  def training_step(self, batch, batch_idx):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['label']

    loss, outputs = self(input_ids, attention_mask, labels)

    outputs = torch.argmax(outputs, dim=1)
    accuracy = self.accuracy(outputs, labels)
    f1 = self.f1(outputs, labels)

    self.log("train_loss", loss, prog_bar=True, logger=True, batch_size=len(batch))
    self.log(
        "train_accuracy",
        accuracy,
        prog_bar=True,
        logger=True,
        batch_size=len(batch),
    )
    self.log("train_f1", f1, prog_bar=True, logger=True, batch_size=len(batch))

    return {"loss": loss, "predictions": outputs, "labels": labels}

  def validation_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["label"]

    loss, outputs = self(input_ids, attention_mask, labels)
    outputs = torch.argmax(outputs, dim=1)
    accuracy = self.accuracy(outputs, labels)
    f1 = self.f1(outputs, labels)

    self.log("val_loss", loss, prog_bar=True, logger=True, batch_size=len(batch))
    self.log( "val_accuracy", accuracy, prog_bar=True, logger=True, batch_size=len(batch))
    self.log("val_f1", f1, prog_bar=True, logger=True, batch_size=len(batch))

    return {"val_loss": loss, "val_accuracy": accuracy, "val_f1": f1}

  def test_step(self, batch, batch_idx):
      input_ids = batch["input_ids"]
      attention_mask = batch["attention_mask"]
      labels = batch["label"]

      loss, outputs = self(input_ids, attention_mask, labels)
      outputs = torch.argmax(outputs, dim=1)
      accuracy = self.accuracy(outputs, labels)
      f1 = self.f1(outputs, labels)
      
      self.log("test_loss", loss, prog_bar=True, logger=True, batch_size=len(batch))
      self.log(
          "test_accuracy", accuracy, prog_bar=True, logger=True, batch_size=len(batch)
      )
      self.log("test_f1", f1, prog_bar=True, logger=True, batch_size=len(batch))

      return {"test_loss": loss, "test_accuracy": accuracy, "test_f1": f1}
      
  def configure_optimizers(self):
    optimizer = AdamW(self.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
    total_steps = self.config['train_size']/self.config['batch_size']

    warmup_steps = math.floor(total_steps * self.config['warmup'])
    warmup_steps = math.floor(total_steps * self.config['warmup'])
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    return [optimizer],[scheduler]