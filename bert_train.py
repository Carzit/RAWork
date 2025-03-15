import os
import sys
import logging
from collections import OrderedDict
from numbers import Number
from typing import List, Tuple, Dict, Literal, Union, Optional, Callable, Any
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import akshare as ak


from gensim.models import Word2Vec, KeyedVectors
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import BertPreTokenizer
from transformers import PreTrainedTokenizerFast, BertConfig, BertForMaskedLM
from accelerate import Accelerator

import utils
from datasets import AssetBERTMLMDataset, WrappedTokenizer
from preparers import AssetBERT_Preparer, Optimizer_Preparer
from configs import Config, OptimizerConfig, AssetBERTModelConfig, AssetBERTTrainConfig, ConfigDict



class AssetBERTMLMTrainer:
    def __init__(self):
        self.logger:logging.Logger = None
        self.model:BertForMaskedLM = None
        self.tokenizer:WrappedTokenizer = None
        self.optimizer:torch.optim.Optimizer = None
        self.lr_scheduler:torch.optim.lr_scheduler.LRScheduler = None

        self.accelerator:Accelerator = None

        self.config:ConfigDict = ConfigDict({"model":AssetBERTModelConfig(),
                                             "optimizer":OptimizerConfig(),
                                             "train":AssetBERTTrainConfig(),})
        
        self.model_preparer:AssetBERT_Preparer = AssetBERT_Preparer()
        self.optimizer_preparer:Optimizer_Preparer = Optimizer_Preparer()

    def set_logger(self,
                   name: str = 'AssetBERT-MLM Trainer',
                   log_file: str = 'AssetBERT-MLM_Trainer.log'):
        self.logger = utils.LoggerPreparer(name, log_file=log_file).prepare()
        self.optimizer_preparer.set_logger(self.logger)

    def load_configs(self, config_file:str):
        self.logger.info(f"Loading Configs from `{config_file}`")
        self.config.from_file(config_file)
        if self.config.optimizer.lr_scheduler_train_steps is None:
            self.config.optimizer.lr_scheduler_train_steps = self.config.train.max_epoches
        self.model_preparer.set_config(self.config.model)
        self.optimizer_preparer.set_config(self.config.optimizer)

    def load_args(self, args:Dict[str, Any]):
        self.logger.info("Loading Args...")
        self.config.from_dict(args)
        if self.config.optimizer.lr_scheduler_train_steps is None:
            self.config.optimizer.lr_scheduler_train_steps = self.config.train.max_epoches
        self.model_preparer.set_config(self.config.model)
        self.optimizer_preparer.set_config(self.config.optimizer)

    def prepare(self):
        self.logger.info("Preparing Trainer...")

        if self.config.train.mixed_precision is None:
            self.accelerator = Accelerator(gradient_accumulation_steps=self.config.train.gradient_accumulation_steps)
        else:
            self.accelerator = Accelerator(gradient_accumulation_steps=self.config.train.gradient_accumulation_steps, 
                                           mixed_precision=self.config.train.mixed_precision)
        
        self.model = self.model_preparer.prepare()
        self.optimizer, self.lr_scheduler = self.optimizer_preparer.prepare(self.model.bert.parameters())


if __name__ == "__main__":
    trainer = AssetBERTMLMTrainer()
    trainer.set_logger()
    trainer.load_configs(r"bert_stock_saved\train.json")
    trainer.prepare()
   






