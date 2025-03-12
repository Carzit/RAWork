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

import utils

class AssetEmbeddingMLMDataset(Dataset):
    def __init__(self, 
                 data_dir:str, 
                 tokenizer:PreTrainedTokenizerFast, 
                 max_length:int=20, 
                 mask_prob:float=0.15, 
                 cache_size:int=5):
        """
        :param data_dir: CSV 文件目录
        :param tokenizer: 自定义 Tokenizer（基于 index-vocab）
        :param max_length: 最大序列长度
        :param mask_prob: MLM 任务的 Mask 概率
        :param cache_size: 允许缓存的 CSV 文件数量，避免重复加载
        """
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob
        self.cache_size = cache_size  # 限制缓存大小，避免占用过多内存
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".csv")]

        # 计算数据集大小（行数总和）
        self.file_line_counts = {file: sum(1 for _ in open(file, "r")) - 1 for file in self.files}
        self.total_rows = sum(self.file_line_counts.values())

        # 缓存机制（文件级缓存策略 使用 OrderedDict 实现 LRU）
        self.file_cache = OrderedDict()

    def __len__(self):
        return self.total_rows
    
    def mask_tokens(self, input_ids, special_tokens_mask=None):
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mask_prob)

        # 生成特殊 token mask，确保 `[CLS]`, `[SEP]`, `[PAD]` 不被 Mask
        if special_tokens_mask is None:
            special_tokens_mask = torch.tensor(self.tokenizer.get_special_tokens_mask(input_ids.tolist(), 
                                                                                      already_has_special_tokens=True), 
                                               dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # 选择 Mask 位置
        mask_indices = torch.bernoulli(probability_matrix).bool()
        labels[~mask_indices] = -100  # 非 Mask 位置的 token 设为 -100（PyTorch 交叉熵忽略此值）

        # 80% 替换为 [MASK]
        mask_token_id = self.tokenizer.convert_tokens_to_ids("[MASK]")
        input_ids[mask_indices] = mask_token_id

        # 10% 替换为随机 token
        random_selection = torch.bernoulli(torch.full(labels.shape, 0.1)).bool() & mask_indices
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[random_selection] = random_words[random_selection]

        return input_ids, labels

    

    def load_csv(self, file_path):
        """读取 CSV 并缓存，若缓存已满，则移除最久未使用的文件"""
        if file_path in self.file_cache:
            self.file_cache.move_to_end(file_path)  # 更新 LRU 位置
            return self.file_cache[file_path]
        
        if len(self.file_cache) >= self.cache_size:
            self.file_cache.popitem(last=False)  # 移除最早插入的项（LRU 机制）

        df = pd.read_csv(file_path)
        self.file_cache[file_path] = df
        return df

    def get_file_and_line(self, index):
        """根据全局 index 映射到具体的文件和行"""
        cumulative_lines = 0
        for file, line_count in self.file_line_counts.items():
            if index < cumulative_lines + line_count:
                return file, index - cumulative_lines
            cumulative_lines += line_count
        return None, None

    def __getitem__(self, idx):
        file_path, line_idx = self.get_file_and_line(idx)
        if file_path is None:
            raise IndexError(f"索引 {idx} 超出范围")

        df = self.load_csv(file_path)
        row = df.iloc[line_idx]
        holdings = "[CLS] " + " ".join(str(id).zfill(6) for id in eval(row["Holdings"][1:-1])) + " [SEP]"  # 从字符串中提取持仓信息
        print(holdings)

        # Tokenization
        encoding = self.tokenizer(
            holdings,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids, labels = self.mask_tokens(encoding["input_ids"].squeeze(0))

        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels
        }

class WrappedTokenizer(PreTrainedTokenizerFast):
    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        """
        :param tokenizer: 预训练的 Tokenizer（必须是 PreTrainedTokenizerFast 类型）
        :param id_to_name: ID 到 Name 的映射，例如 {"00001": "平安银行"}
        """
        self.tokenizer:PreTrainedTokenizerFast = tokenizer
        self.name_to_id:Dict[str,str] = None
        self.id_to_name:Dict[str,str] = None

    def load_name2id_mapping(self, file_path:Optional[str]="stock_info_id_name.csv"):
        """加载 ID 到 Name 的映射"""
        if file_path is not None:
            stock_info_a_code_name_df = pd.read_csv(file_path)
        else:
            stock_info_a_code_name_df = ak.stock_info_a_code_name()
        self.name_to_id = dict(zip(stock_info_a_code_name_df["name"], stock_info_a_code_name_df["code"]))
        self.id_to_name = dict(zip(stock_info_a_code_name_df["code"], stock_info_a_code_name_df["name"]))

    def stock_encode(self, text, **kwargs):
        """
        :param text: 可以是 ID（如 "00001"）或 Name（如 "平安银行"）
        :return: token_id 列表
        """
        if text in self.id_to_name:  # 如果输入是 ID，转换为 Name
            text = self.id_to_name[text]
        return self.tokenizer.encode(text, **kwargs)
    
    def stock_decode(self, token_ids, mode="name", **kwargs):
        """
        :param token_ids: 需要解码的 token_id 列表
        :param mode: "name" 返回 Name，"id" 返回 ID
        :return: 解码后的字符串
        """
        text = self.tokenizer.decode(token_ids, **kwargs)
        
        if mode == "id" and text in self.name_to_id:
            return self.name_to_id[text]  # 返回 ID
        return text  # 默认返回 Name
    
    def __getattr__(self, item):
        """ 代理所有 Tokenizer 其他方法，保持完整 API 兼容性 """
        if item not in ["tokenizer", "name_to_id", "id_to_name"]:
            return getattr(self.tokenizer, item)

class W2V_BERT_Transfer:
    """Word2Vec 迁移到 BERT Tokenizer Embedding"""
    def __init__(self):
        self.logger:logging.Logger = None

        self.word2vec_model:Word2Vec = None
        self.bert_tokenizer:PreTrainedTokenizerFast = None
        self.bert_model:BertForMaskedLM = None

        self.token2index_mapping:Dict[str, int] = None
        self.index2token_mapping:Dict[int, str] = None

    @utils.log_exceptions_inclass()
    def load_word2vec_model(self, file_path:Optional[str]=None, model:Optional[Word2Vec]=None):
        self.logger.info("Loading Word2Vec Model...")
        if file_path is not None and model is not None:
            raise ValueError("file_path 和 model 参数只能选择一个")
        elif file_path is not None and model is None:
            if file_path.endswith(".model"):
                self.word2vec_model = Word2Vec.load(file_path)
            elif file_path.endswith(".txt"):
                self.word2vec_model = KeyedVectors.load_word2vec_format(file_path, binary=False)
            else:
                raise ValueError(f"Unrecognized model file `{file_path}`")
        elif file_path is None and model is not None:
            self.word2vec_model = model
        else:
            raise ValueError("file_path 和 model 参数必须选择一个")
        
        self.token2index_mapping = get_token2index_mapping(self.word2vec_model)
        self.index2token_mapping = get_index2token_mapping(self.word2vec_model)

    @utils.log_exceptions_inclass()
    def load_bert_model(self, model:BertForMaskedLM):
        self.logger.info("Loading BERT Model...")
        self.bert_model = model

    @utils.log_exceptions_inclass()
    def set_logger(self, 
                   name: str = 'Word2Vec-BERT Transfer', 
                   console_level=logging.DEBUG, 
                   file_level=logging.DEBUG,
                   log_file: str = 'W2V_BERT_Transfer.log', 
                   max_bytes: int = 1e6, 
                   backup_count: int = 5):
        self.logger = utils.LoggerPreparer(name, console_level, file_level, log_file, max_bytes, backup_count).prepare()

    @utils.log_exceptions_inclass()
    def save_configs(self, folder_path):
        """保存 token-index 映射为 JSON 文件, 保存 tokenizer配置文件，保存BERT模型配置文件"""
        self.logger.info(f"Saving Configs in `{folder_path}`")
        os.makedirs(folder_path, exist_ok=True)
        with open(os.path.join(folder_path, "stockid2index.json"), "w", encoding="utf-8") as f:
            json.dump(self.token2index_mapping, f, ensure_ascii=False, indent=2)
        with open(os.path.join(folder_path, "index2stockid.json"), "w", encoding="utf-8") as f:
            json.dump(self.index2token_mapping, f, ensure_ascii=False, indent=2)
        self.bert_tokenizer.save_pretrained(os.path.join(folder_path, "bert_tokenizer_config"))
        self.bert_model.config.save_pretrained(os.path.join(folder_path, "bert_model_config"))
    
    @utils.log_exceptions_inclass()
    def transfer_tokenizer(self, tokenizer_object:Optional[Tokenizer]=None, tokenizer_file:Optional[str]=None)->WrappedTokenizer:
        """转换 Tokenizer"""
        self.logger.info("Transferring Tokenizer...")
        if tokenizer_object is None and tokenizer_file is None:
            inner_tokenizer = Tokenizer(WordLevel(vocab=self.token2index_mapping, unk_token="[UNK]"))
            inner_tokenizer.pre_tokenizer = BertPreTokenizer()
            bert_tokenizer = PreTrainedTokenizerFast(tokenizer_object=inner_tokenizer)

        elif tokenizer_object is not None and tokenizer_file is None:
            bert_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_object)
        elif tokenizer_object is None and tokenizer_file is not None:
            bert_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
        else:
            raise ValueError("tokenizer_object 和 tokenizer_file 参数必须选择一个")
        # 添加特殊token
        bert_tokenizer.add_special_tokens(
            special_tokens_dict = {
                "cls_token":"[CLS]",
                "sep_token":"[SEP]",
                "mask_token":"[MASK]",
                "pad_token":"[PAD]",
                "unk_token":"[UNK]",
                }
        )
        self.bert_tokenizer = WrappedTokenizer(bert_tokenizer)
        self.bert_tokenizer.load_name2id_mapping()
        return self.bert_tokenizer

    def transfer_model(self)->BertForMaskedLM:
        """转换模型 迁移嵌入层"""
        self.logger.info("Transferring Model...")
        assert len(self.word2vec_model.wv)+5 == self.bert_model.config.vocab_size, f"词表大小不匹配: W2V:{len(self.word2vec_model.wv)} BERT:{self.bert_model.config.vocab_size}"

        vocab_size = len(self.word2vec_model.wv)
        embedding_dim = self.word2vec_model.vector_size

        # 加载 word2vec 权重
        embedding_matrix = np.zeros((vocab_size + 5, embedding_dim))
        for i, word in enumerate(self.word2vec_model.wv.index_to_key):
            embedding_matrix[i] = self.word2vec_model.wv[word]

        # 初始化特殊 token
        embedding_matrix[vocab_size + 3] = embedding_matrix[0]  # [CLS]        
        embedding_matrix[vocab_size + 4] = embedding_matrix[1]  # [SEP]
        embedding_matrix[vocab_size + 0] = np.mean(embedding_matrix, axis=0)  # [MASK] 设为均值
        embedding_matrix[vocab_size + 1] = np.zeros((embedding_dim,))  # [PAD] 设为零向量
        embedding_matrix[vocab_size + 2] = np.random.normal(embedding_matrix.mean(), embedding_matrix.std(), (embedding_dim,))  # [UNK] 设为随机

        self.bert_model.bert.embeddings.word_embeddings.weight.data.copy_(torch.tensor(embedding_matrix, dtype=torch.float32))

        return self.bert_model
    
def get_token2index_mapping(model:Word2Vec)->Dict[str, int]:
    """保存 token-index 映射"""
    token2index = {word.zfill(6): i for i, word in enumerate(model.wv.index_to_key)}

    # 追加特殊 token
    special_tokens = ["[MASK]", "[PAD]", "[UNK]"]
    for i, token in enumerate(special_tokens, start=len(token2index)):
        token2index[token] = i

    return token2index

def get_index2token_mapping(model:Word2Vec)->Dict[int, str]:
    """保存 token-index 映射"""
    index2token = {i:word.zfill(6) for i, word in enumerate(model.wv.index_to_key)}

    # 追加特殊 token
    special_tokens = ["[MASK]", "[PAD]", "[UNK]"]
    for i, token in enumerate(special_tokens, start=len(index2token)):
        index2token[i] = token

    return index2token








if __name__ == "__main__":
    
    # 读取数据
    transfer = W2V_BERT_Transfer()
    transfer.set_logger()
    transfer.load_word2vec_model(r"word2vec_stock.model")
    tokenizer = transfer.transfer_tokenizer()
    dataset = AssetEmbeddingMLMDataset(data_dir=r"data\preprocess\AssetEmbedding2019-2024\merged", tokenizer=tokenizer, cache_size=5)
    print(dataset[0])
    print(tokenizer.decode(dataset[0]["input_ids"]))
    print(tokenizer.encode("[CLS] [SEP] [MASK] [PAD] [UNK]"))

    # 自定义 BERT 配置
    config = BertConfig(
        vocab_size=len(tokenizer),  # 词表大小
        hidden_size=100,  # 隐藏层维度, 必须和嵌入向量长度一致
        num_hidden_layers=4,  # BERT 层数
        num_attention_heads=4,  # 多头注意力
        intermediate_size=512,  # FFN 维度
        max_position_embeddings=50,
        type_vocab_size=1)

    # 初始化 MLM 任务模型
    model = BertForMaskedLM(config).to("cuda")
    transfer.load_bert_model(model)
    transfer.transfer_model()
    transfer.save_configs("bert_stock_saved")
