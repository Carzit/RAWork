import os
import ast
import logging
from typing import List, Tuple, Dict, Literal, Union, Optional, Callable, Any
import json

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec, KeyedVectors
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import BertPreTokenizer
from tqdm import tqdm

import utils

# 1️⃣ **流式加载 CSV 文件**
class AssetTokenDataset:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def __iter__(self):
        for file in os.listdir(self.folder_path):
            if file.endswith(".csv"):
                file_path = os.path.join(self.folder_path, file)
                for chunk in pd.read_csv(file_path, chunksize=10000):  # 分块加载
                    for _, row in chunk.iterrows():
                        holdings = ast.literal_eval(row["Holdings"])  # 转换为列表
                        yield ["[CLS]"]+[str(h).zfill(6) for h in holdings]+["[SEP]"]  # 转换为字符串，适配 Word2Vec


class Word2VecTrainer:
    def __init__(self):
        self.dataset:AssetTokenDataset = None
        self.model:Word2Vec = None
        self.configs:Dict[str, Any] = {
            "embedding_dim": 100,
            "epochs": 10,
            "window": 5,
            "min_count": 1,
            "sg": 1,
            "sample": 1e-3, 
            "negative_sample": 10,
            "seed": 42,
            "workers": 4,
        }
        self.logger:logging.Logger = None

    def load_config(self, config_file:str):
        self.configs.update(utils.read_configs(config_file))

    def set_config(self, config_dict:Optional[Dict[str, Any]], **config_kwargs):
        if config_dict:
            self.configs.update(config_dict)
        if config_kwargs:
            self.configs.update(config_kwargs)

    def load_dataset(self, dataset:AssetTokenDataset):
        self.dataset = dataset

    def set_logger(self, 
                   name: str = 'Word2Vec Train', 
                   console_level=logging.DEBUG, 
                   file_level=logging.DEBUG,
                   log_file: str = 'Word2Vec_Train.log', 
                   max_bytes: int = 1e6, 
                   backup_count: int = 5):
        self.logger = utils.LoggerPreparer(name, console_level, file_level, log_file, max_bytes, backup_count).prepare()

    @utils.log_exceptions_inclass()
    def train(self):

        self.logger.info("Initializing Model...")
        self.model = Word2Vec(vector_size=self.configs["embedding_dim"], 
                            window=self.configs["window"], 
                            min_count=self.configs["min_count"], 
                            sg=self.configs["sg"], 
                            sample=self.configs["sample"], 
                            negative=self.configs["negative_sample"],
                            seed=self.configs["seed"],
                            workers=self.configs["workers"])
        
        self.logger.info("Building Vocabulary...")
        self.model.build_vocab(self.dataset)

        self.logger.info("Training Word2Vec...")
        for epoch in tqdm(range(self.configs["epochs"]), desc="Training Progress", unit="epoch"):
            self.model.train(self.dataset, total_examples=self.model.corpus_count, epochs=1)
            loss = self.model.get_latest_training_loss()
            self.logger.debug(f"Epoch[{epoch+1}] Loss: {loss}")

    def save(self, save_path="word2vec.model"):
        self.model.save(save_path)
        self.model.wv.save_word2vec_format(save_path + ".txt", binary=False) # 兼容性格式
        self.logger.info(f"Word2Vec model saved to {save_path}")

    

# 5️⃣ **完整管线**
if __name__ == "__main__":
    folder = r"data\preprocess\AssetEmbedding2019-2024\merged"  # 修改为你的数据集路径
    dataset = AssetTokenDataset(folder)
    w2v_trainer = Word2VecTrainer()
    w2v_trainer.load_dataset(dataset)
    w2v_trainer.set_logger()
    w2v_trainer.train()
    w2v_trainer.save()
