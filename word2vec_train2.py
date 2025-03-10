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
                        yield ["[CLS]"]+[str(h) for h in holdings]+["[SEP]"]  # 转换为字符串，适配 Word2Vec


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

    def train(self):
        try:
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

        except Exception as e:
            self.logger.error("Exception occurred", exc_info=True)

    def save(self, save_path="word2vec.model"):
        self.model.save(save_path)
        self.model.wv.save_word2vec_format(save_path + ".txt", binary=False) # 兼容性格式
        self.logger.info(f"Word2Vec model saved to {save_path}")

class BERTEmbeddingTransfer:
    def __init__(self):
        self.word2vec_model:Word2Vec = None
        self.embedding:nn.Embedding = None
        self.token2index_mapping:Dict[str, int] = None
        self.index2token_mapping:Dict[int, str] = None

    def load_word2vec_model(self, file_path:str):
        if file_path.endswith(".model"):
            self.word2vec_model = Word2Vec.load(file_path)
        elif file_path.endswith(".txt"):
            self.word2vec_model = KeyedVectors.load_word2vec_format(file_path, binary=False)
        else:
            raise ValueError(f"Unrecognized model file `{file_path}`")
        
        self.token2index_mapping = get_token2index_mapping(self.word2vec_model)
        self.index2token_mapping = get_index2token_mapping(self.word2vec_model)
        
    def save_mappings(self, folder_path):
        """保存 token-index 映射为 JSON 文件"""

        os.makedirs(folder_path, exist_ok=True)
        with open(os.path.join(folder_path, "stockid2index.json"), "w", encoding="utf-8") as f:
            json.dump(self.token2index_mapping, f, ensure_ascii=False, indent=2)
        with open(os.path.join(folder_path, "index2stockid.json"), "w", encoding="utf-8") as f:
            json.dump(self.index2token_mapping, f, ensure_ascii=False, indent=2)

        tokenizer = Tokenizer(WordLevel(vocab=self.token2index_mapping, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = BertPreTokenizer()
        tokenizer.save(os.path.join(folder_path, "tokenizer.json"))
        
        
    def transform(self) -> nn.Embedding:
        vocab_size = len(self.word2vec_model.wv)
        embedding_dim = self.word2vec_model.vector_size
        self.embedding = nn.Embedding(vocab_size + 3, embedding_dim)  # +3 处理特殊 token

        # 加载 word2vec 权重
        embedding_matrix = np.zeros((vocab_size + 3, embedding_dim))
        for i, word in enumerate(self.word2vec_model.wv.index_to_key):
            embedding_matrix[i] = self.word2vec_model.wv[word]

        # 初始化特殊 token
        special_tokens = ["[MASK]", "[PAD]", "[UNK]"]

        embedding_matrix[vocab_size + 0] = np.mean(embedding_matrix, axis=0)  # [MASK] 设为均值
        embedding_matrix[vocab_size + 1] = np.zeros((embedding_dim,))  # [PAD] 设为零向量
        embedding_matrix[vocab_size + 2] = np.random.normal(embedding_matrix.mean(), embedding_matrix.std(), (embedding_dim,))  # [UNK] 设为随机

        self.embedding.weight.data.copy_(torch.tensor(embedding_matrix, dtype=torch.float32))

        return self.embedding
    
def get_token2index_mapping(model:Word2Vec)->Dict[str, int]:
    """保存 token-index 映射"""
    token2index = {word: i for i, word in enumerate(model.wv.index_to_key)}

    # 追加特殊 token
    special_tokens = ["[MASK]", "[PAD]", "[UNK]"]
    for i, token in enumerate(special_tokens, start=len(token2index)):
        token2index[token] = i



    return token2index

def get_index2token_mapping(model:Word2Vec)->Dict[int, str]:
    """保存 token-index 映射"""
    index2token = {i:word for i, word in enumerate(model.wv.index_to_key)}

    # 追加特殊 token
    special_tokens = ["[MASK]", "[PAD]", "[UNK]"]
    for i, token in enumerate(special_tokens, start=len(index2token)):
        index2token[i] = token

    return index2token

    

# 5️⃣ **完整管线**
if __name__ == "__main__":
    #folder = r"data\preprocess\AssetEmbedding2019-2024\merged"  # 修改为你的数据集路径
    #dataset = AssetTokenDataset(folder)
    #w2v_trainer = Word2VecTrainer()
    #w2v_trainer.load_dataset(dataset)
    #w2v_trainer.set_logger()
    #w2v_trainer.train()
    #w2v_trainer.save()

    bert_transfer = BERTEmbeddingTransfer()
    bert_transfer.load_word2vec_model("word2vec.model")
    bert_transfer.save_mappings("./")
    bert_embedding = bert_transfer.transform()


    print("BERT Embedding ready!")
