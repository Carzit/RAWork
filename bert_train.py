import os
import sys
import datetime
import logging
from collections import OrderedDict
from numbers import Number
from typing import List, Tuple, Dict, Literal, Union, Optional, Callable, Any

from tqdm import tqdm
import akshare as ak
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from transformers import PreTrainedTokenizerFast, BertForMaskedLM
from accelerate import Accelerator

import utils
from datasets import AssetBERTMLMDataset, WrappedTokenizer, DataLoader
from preparers import AssetBERT_Preparer, Optimizer_Preparer, DataLoader_Preparer, Tokenizer_Preparer
from configs import Config, OptimizerConfig, AssetBERTModelConfig, AssetBERTTrainConfig, TokenizerConfig, DataLoaderConfig, ConfigDict



class AssetBERTMLMTrainer:
    def __init__(self):
        self.logger:logging.Logger = None

        self.dataloader:DataLoader = None
        self.model:BertForMaskedLM = None
        self.tokenizer:PreTrainedTokenizerFast = None
        self.optimizer:torch.optim.Optimizer = None
        self.lr_scheduler:torch.optim.lr_scheduler.LRScheduler = None

        self.accelerator:Accelerator = None
        self.summary_writer:SummaryWriter = None

        self.config:ConfigDict = ConfigDict({"model":AssetBERTModelConfig(),
                                             "tokenizer":TokenizerConfig(),
                                             "dataloader":DataLoaderConfig(),
                                             "optimizer":OptimizerConfig(),
                                             "train":AssetBERTTrainConfig()
                                             })
        
        self.tokenizer_preparer:Tokenizer_Preparer = Tokenizer_Preparer()
        self.dataloader_preparer:DataLoader_Preparer = DataLoader_Preparer()
        self.model_preparer:AssetBERT_Preparer = AssetBERT_Preparer()
        self.optimizer_preparer:Optimizer_Preparer = Optimizer_Preparer()

    def set_logger(self,
                   name: str = 'AssetBERT-MLM Trainer',
                   log_file: str = 'AssetBERT-MLM_Trainer.log'):
        self.logger = utils.LoggerPreparer(name, log_file=log_file).prepare()
        self.tokenizer_preparer.set_logger(self.logger)
        self.dataloader_preparer.set_logger(self.logger)
        self.model_preparer.set_logger(self.logger)
        self.optimizer_preparer.set_logger(self.logger)

    @utils.log_exceptions_inclass()
    def load_configs(self, config_file:str):
        self.logger.info(f"Loading Configs from `{config_file}`")
        self.config.from_file(config_file)

        self.tokenizer_preparer.set_config(self.config.tokenizer)
        self.dataloader_preparer.set_config(self.config.dataloader)
        self.model_preparer.set_config(self.config.model)
        self.optimizer_preparer.set_config(self.config.optimizer)
        
        
    @utils.log_exceptions_inclass()
    def load_args(self, args:Dict[str, Any]):
        self.logger.info("Loading Args...")
        self.config.from_dict(args)
        self.tokenizer_preparer.set_config(self.config.tokenizer)
        self.dataloader_preparer.set_config(self.config.dataloader)
        self.model_preparer.set_config(self.config.model)
        self.optimizer_preparer.set_config(self.config.optimizer)
        if self.config.optimizer.lr_scheduler_train_steps is None:
            self.config.optimizer.lr_scheduler_train_steps = len(self.dataloader) # self.config.train.max_epoches

    @utils.log_exceptions_inclass()
    def prepare(self):
        self.logger.info("Preparing Trainer...")
        self.tokenizer = self.tokenizer_preparer.prepare()
        self.dataloader = self.dataloader_preparer.prepare(self.tokenizer)

        if self.config.optimizer.lr_scheduler_train_steps is None:
            self.config.optimizer.lr_scheduler_train_steps = len(self.dataloader) # self.config.train.max_epoches

        if self.config.model.vocab_size != len(self.tokenizer):
            self.config.model.vocab_size = len(self.tokenizer)
            self.model_preparer.set_config(self.config.model)
        
        self.model = self.model_preparer.prepare()
        self.optimizer, self.lr_scheduler = self.optimizer_preparer.prepare(self.model.parameters())

        if self.config.train.mixed_precision is None:
            self.accelerator = Accelerator(gradient_accumulation_steps=self.config.train.gradient_accumulation_steps)
        else:
            self.accelerator = Accelerator(gradient_accumulation_steps=self.config.train.gradient_accumulation_steps, 
                                           mixed_precision=self.config.train.mixed_precision)
        
        self.model, self.optimizer, self.dataloader, self.lr_scheduler = self.accelerator.prepare(self.model, self.optimizer, self.dataloader, self.lr_scheduler)
        if self.accelerator.is_local_main_process:
            os.makedirs(self.config.train.save_folder, exist_ok=True)
            self.summary_writer = SummaryWriter(
                os.path.join(
                    self.config.train.save_folder, f"Tensorboard_{self.config.train.save_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
                ))

    @utils.log_exceptions_inclass()
    def train(self):
        self.logger.info("Training...")
        self.model.train()
        
        torch.autograd.set_detect_anomaly(self.config.train.detect_anomaly)
        for epoch in range(self.config.train.max_epoches):
            avg_loss_calculator = utils.MeanVarianceAccumulator()
            for step, batch in enumerate(tqdm(self.dataloader, total=self.config.optimizer.lr_scheduler_train_steps, disable=not self.accelerator.is_local_main_process)):
                with self.accelerator.accumulate(self.model):
                    input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss

                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                if step % self.config.train.check_per_step == 0:
                    if self.accelerator.is_local_main_process:
                        self.logger.info(f"Epoch:{epoch+1}, Step:{step+1}, Loss:{loss.item()}")
                avg_loss_calculator.accumulate(loss.item())

            if epoch % self.config.train.report_per_epoch == 0:
                if self.accelerator.is_local_main_process:
                    self.logger.info(f"Epoch:{epoch+1}, Loss Mean:{avg_loss_calculator.mean}, Loss Std:{avg_loss_calculator.std()}")
                    self.summary_writer.add_scalar("Loss", avg_loss_calculator.mean, epoch+1)

            if epoch % self.config.train.save_per_epoch == 0:
                if self.accelerator.is_local_main_process:
                    safe_serialization = True if self.config.train.save_format == ".safetensors" else False
                    self.model.save_pretrained(os.path.join(self.config.train.save_folder, f"{self.config.train.save_name}_epoch{epoch+1}"), safe_serialization=safe_serialization)
                    self.logger.info(f"Epoch {epoch+1} Checkpoint saved to {self.config.train.save_folder}")
                    

        if self.accelerator.is_local_main_process:
            self.summary_writer.close()


if __name__ == "__main__":
    trainer = AssetBERTMLMTrainer()
    trainer.set_logger()
    trainer.load_configs(r"bert_stock_saved\train.json")
    trainer.prepare()
    trainer.train()
   






