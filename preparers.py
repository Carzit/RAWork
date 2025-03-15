import logging
import argparse
from numbers import Number
from typing import Callable, Literal, Union, Optional, List, Dict, Any

import torch
import lion_pytorch
import dadaptation
import diffusers
import transformers
import bitsandbytes as bnb
from transformers import PreTrainedTokenizerFast, BertConfig, BertForMaskedLM
import safetensors

from configs import Config, OptimizerConfig, AssetBERTModelConfig
from datasets import WrappedTokenizer, Tokenizer, PreTrainedTokenizerFast, WordLevel, BertPreTokenizer
import utils

class Preparer:
    def __init__(self):
        self.logger:logging.Logger = None
        self.config:Config

    def set_config(self, config:Config):
        self.config = config

    def get_config(self):
        return self.config
    
    def set_logger(self, logger:logging.Logger):
        if not logger:
            logger = logging
        self.logger = logger

    @utils.log_exceptions_inclass()
    def prepare(self):
        ...

class AssetBERT_Preparer(Preparer):
    def __init__(self):
        super().__init__()
        self.config:AssetBERTModelConfig = AssetBERTModelConfig()

    def prepare(self) -> BertForMaskedLM:
        bert_config = BertConfig(
            vocab_size=self.config.vocab_size,  # 词表大小
            hidden_size=self.config.hidden_size,  # 隐藏层维度, 必须和嵌入向量长度一致
            num_hidden_layers=self.config.num_hidden_layers,  # BERT 层数
            num_attention_heads=self.config.num_attention_heads,  # 多头注意力
            intermediate_size=self.config.intermediate_size,  # FFN 维度
            max_position_embeddings=self.config.max_position_embeddings,  # 最大位置编码
            type_vocab_size=self.config.type_vocab_size,  # 类型词表大小
            )
        model = BertForMaskedLM(bert_config)
        return model
    
class Tokenizer_Preparer(Preparer):
    def __init__(self):
        super().__init__()
        self.config:AssetBERTModelConfig = AssetBERTModelConfig()

    def prepare(self) -> WrappedTokenizer:
        if self.pretrained_tokenizer_file is not None:
            self.vocab_file = None
            tokenizer = WrappedTokenizer.from_pretrained(self.pretrained_tokenizer_file)
        elif self.vocab_file is not None:
            self.pretrained_tokenizer_file = None
            inner_tokenizer = Tokenizer(WordLevel(vocab=self.vocab_file, unk_token="[UNK]"))
            inner_tokenizer.pre_tokenizer = BertPreTokenizer()
            bert_tokenizer = PreTrainedTokenizerFast(tokenizer_object=inner_tokenizer)
            bert_tokenizer.add_special_tokens(
                special_tokens_dict = {
                    "cls_token":"[CLS]",
                    "sep_token":"[SEP]",
                    "mask_token":"[MASK]",
                    "pad_token":"[PAD]",
                    "unk_token":"[UNK]",
                    }
            )
            tokenizer = WrappedTokenizer(bert_tokenizer)
        tokenizer.load_name2id_mapping(self.alias_file)
        return tokenizer



class Optimizer_Preparer(Preparer):
    def __init__(self) -> None:
        super().__init__()
        self.config:OptimizerConfig = OptimizerConfig()

    def prepare_optimizer(self, trainable_params) -> torch.optim.Optimizer:
        if self.config.optimizer_type == "Adam":
            optimizer_class = torch.optim.Adam
            filtered_kwargs, invalid_kwargs = utils.filter_func_kwargs(optimizer_class.__init__, self.config.optimizer_kwargs)
            if invalid_kwargs:
                self.logger.warning(f"Invalid kwargs: {invalid_kwargs}")
            optimizer = optimizer_class(trainable_params, lr=self.config.learning_rate, **filtered_kwargs)
        elif self.config.optimizer_type == "Adam8bit":
            optimizer_class = bnb.optim.Adam8bit
            filtered_kwargs, invalid_kwargs = utils.filter_func_kwargs(optimizer_class.__init__, self.config.optimizer_kwargs)
            if invalid_kwargs:
                self.logger.warning(f"Invalid kwargs: {invalid_kwargs}")
            optimizer = optimizer_class(trainable_params, lr=self.config.learning_rate, **filtered_kwargs)
        elif self.config.optimizer_type == "PagedAdam8bit":
            optimizer_class = bnb.optim.PagedAdam8bit
            filtered_kwargs, invalid_kwargs = utils.filter_func_kwargs(optimizer_class.__init__, self.config.optimizer_kwargs)
            if invalid_kwargs:
                self.logger.warning(f"Invalid kwargs: {invalid_kwargs}")
            optimizer = optimizer_class(trainable_params, lr=self.config.learning_rate, **filtered_kwargs)
        elif self.config.optimizer_type == "AdamW":
            optimizer_class = torch.optim.AdamW
            filtered_kwargs, invalid_kwargs = utils.filter_func_kwargs(optimizer_class.__init__, self.config.optimizer_kwargs)
            if invalid_kwargs:
                self.logger.warning(f"Invalid kwargs: {invalid_kwargs}")
            optimizer = optimizer_class(trainable_params, lr=self.config.learning_rate, **filtered_kwargs)
        elif self.config.optimizer_type == "AdamW8bit":
            optimizer_class = bnb.optim.AdamW8bit
            filtered_kwargs, invalid_kwargs = utils.filter_func_kwargs(optimizer_class.__init__, self.config.optimizer_kwargs)
            if invalid_kwargs:
                self.logger.warning(f"Invalid kwargs: {invalid_kwargs}")
            optimizer = optimizer_class(trainable_params, lr=self.config.learning_rate, **filtered_kwargs)
        elif self.config.optimizer_type == "PagedAdamW8bit":
            optimizer_class = bnb.optim.PagedAdamW8bit
            filtered_kwargs, invalid_kwargs = utils.filter_func_kwargs(optimizer_class.__init__, self.config.optimizer_kwargs)
            if invalid_kwargs:
                self.logger.warning(f"Invalid kwargs: {invalid_kwargs}")
            optimizer = optimizer_class(trainable_params, lr=self.config.learning_rate, **filtered_kwargs)
        elif self.config.optimizer_type == "Lion":
            optimizer_class = lion_pytorch.Lion
            filtered_kwargs, invalid_kwargs = utils.filter_func_kwargs(optimizer_class.__init__, self.config.optimizer_kwargs)
            if invalid_kwargs:
                self.logger.warning(f"Invalid kwargs: {invalid_kwargs}")
            optimizer = optimizer_class(trainable_params, lr=self.config.learning_rate, **filtered_kwargs)
        elif self.config.optimizer_type == "Lion8bit":
            optimizer_class = bnb.optim.Lion8bit
            filtered_kwargs, invalid_kwargs = utils.filter_func_kwargs(optimizer_class.__init__, self.config.optimizer_kwargs)
            if invalid_kwargs:
                self.logger.warning(f"Invalid kwargs: {invalid_kwargs}")
            optimizer = optimizer_class(trainable_params, lr=self.config.learning_rate, **filtered_kwargs)
        elif self.config.optimizer_type == "PagedLion8bit":
            optimizer_class = bnb.optim.PagedLion8bit
            filtered_kwargs, invalid_kwargs = utils.filter_func_kwargs(optimizer_class.__init__, self.config.optimizer_kwargs)
            if invalid_kwargs:
                self.logger.warning(f"Invalid kwargs: {invalid_kwargs}")
            optimizer = optimizer_class(trainable_params, lr=self.config.learning_rate, **filtered_kwargs)
        elif self.config.optimizer_type == "SGDNesterov":
            if "momentum" not in self.config.optimizer_kwargs:
                self.logger.critical(f"SGD with Nesterov must be with momentum, set momentum to 0.9")
                self.config.optimizer_kwargs["momentum"] = 0.9
            optimizer_class = torch.optim.SGD
            filtered_kwargs, invalid_kwargs = utils.filter_func_kwargs(optimizer_class.__init__, self.config.optimizer_kwargs)
            if invalid_kwargs:
                self.logger.warning(f"Invalid kwargs: {invalid_kwargs}")
            optimizer = optimizer_class(trainable_params, lr=self.config.learning_rate, nesterov=True, **filtered_kwargs)
        elif self.config.optimizer_type == "SGD8bit":
            optimizer_class = bnb.optim.SGD8bit
            filtered_kwargs, invalid_kwargs = utils.filter_func_kwargs(optimizer_class.__init__, self.config.optimizer_kwargs)
            if invalid_kwargs:
                self.logger.warning(f"Invalid kwargs: {invalid_kwargs}")
            optimizer = optimizer_class(trainable_params, lr=self.config.learning_rate, nesterov=True, **filtered_kwargs)
        elif self.config.optimizer_type == "DAdaptation":
            optimizer_class = dadaptation.DAdaptAdam
            filtered_kwargs, invalid_kwargs = utils.filter_func_kwargs(optimizer_class.__init__, self.config.optimizer_kwargs)
            if invalid_kwargs:
                self.logger.warning(f"Invalid kwargs: {invalid_kwargs}")
            optimizer = optimizer_class(trainable_params, lr=self.config.learning_rate, **filtered_kwargs)
        elif self.config.optimizer_type == "Adafactor":
            if "relative_step" not in self.config.optimizer_kwargs:
                self.config.optimizer_kwargs["relative_step"] = True  # default
            if not self.config.optimizer_kwargs["relative_step"] and self.config.optimizer_kwargs.get("warmup_init", False):
                self.logger.info(f"set relative_step to True because warmup_init is True.")
                self.config.optimizer_kwargs["relative_step"] = True
            if self.config.optimizer_kwargs["relative_step"]:
                if self.config.learning_rate != 0.0:
                    self.logger.info(f"Learning rate is used as initial_lr.")
                if self.config.lr_scheduler_type != "adafactor":
                    self.logger.info(f"Use adafactor_scheduler.")
                    self.config.lr_scheduler_type = "adafactor"
            optimizer_class = transformers.optimization.Adafactor
            filtered_kwargs, invalid_kwargs = utils.filter_func_kwargs(optimizer_class.__init__, self.config.optimizer_kwargs)
            if invalid_kwargs:
                self.logger.warning(f"Invalid kwargs: {invalid_kwargs}")
            optimizer = optimizer_class(trainable_params, lr=None, **filtered_kwargs)
        return optimizer
    
    def prepare_lr_scheduler(self, optimizer:torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        if self.config.lr_scheduler_type.lower() == "constant":
            lr_scheduler = diffusers.optimization.get_constant_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=self.config.lr_scheduler_warmup_steps)
        elif self.config.lr_scheduler_type.lower() == "linear":
            lr_scheduler = diffusers.optimization.get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=self.config.lr_scheduler_warmup_steps, 
                num_training_steps=self.config.lr_scheduler_train_steps)
        elif self.config.lr_scheduler_type.lower() == "cosine":
            lr_scheduler = diffusers.optimization.get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config.lr_scheduler_warmup_steps,
                num_training_steps=self.config.lr_scheduler_train_steps, 
                num_cycles=self.config.lr_scheduler_num_cycles)
        elif self.config.lr_scheduler_type.lower() == "cosine_with_restarts":
            lr_scheduler = diffusers.optimization.get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=self.config.lr_scheduler_warmup_steps, 
                num_training_steps=self.config.lr_scheduler_train_steps,
                num_cycles=self.config.lr_scheduler_num_cycles)
        elif self.config.lr_scheduler_type.lower() == "polynomial":
            lr_scheduler = diffusers.optimization.get_polynomial_decay_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=self.config.lr_scheduler_warmup_steps,
                num_training_steps=self.config.lr_scheduler_train_steps,
                power=self.config.lr_scheduler_power)
        elif self.config.lr_scheduler_type.lower() == "adafactor":
            assert type(optimizer) == transformers.optimization.Adafactor, f"Adafactor Scheduler must be used with Adafactor Optimizer. Unexpected optimizer type {type(optimizer)}"
            lr_scheduler = transformers.optimization.AdafactorSchedule(optimizer, initial_lr=self.config.learning_rate)
        return lr_scheduler
    
    @utils.log_exceptions_inclass()
    def prepare(self, trainable_params):
        optimizer = self.prepare_optimizer(trainable_params=trainable_params)
        lr_scheduler = self.prepare_lr_scheduler(optimizer=optimizer)
        print(optimizer, lr_scheduler)
        return optimizer, lr_scheduler
    
