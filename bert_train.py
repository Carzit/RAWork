import os
import sys
from collections import OrderedDict
from numbers import Number
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
from transformers import PreTrainedTokenizerFast



tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizer.json",
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]"
)
#tokenizer.model_max_length = 10
#a = tokenizer.encode("404 413 801", add_special_tokens=True, max_length=10, truncation=True, padding="max_length")
#print(a)
#print(tokenizer.decode(a))



#sys.exit(0)

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
from collections import OrderedDict

class AssetEmbeddingMLMDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=20, mask_prob=0.15, cache_size=5):
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

    def mask_tokens(self, input_ids):
        """对 token 进行 Mask 处理"""
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mask_prob)
        mask_indices = torch.bernoulli(probability_matrix).bool()

        # 80% 替换为 [MASK]
        input_ids[mask_indices] = self.tokenizer.convert_tokens_to_ids("[MASK]")
        
        # 10% 替换成随机 token
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        random_selection = torch.bernoulli(torch.full(labels.shape, 0.1)).bool()
        random_replace_indices = mask_indices & random_selection  # 确保随机替换仅在 Mask 位置发生
        input_ids[random_replace_indices] = random_words[random_replace_indices]

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
        holdings = "[CLS] " + row["Holdings"][1:-1].replace(",", "") + " [SEP]"  # 从字符串中提取持仓信息
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

# 读取数据
dataset = AssetEmbeddingMLMDataset(data_dir=r"data\preprocess\AssetEmbedding2019-2024\merged", tokenizer=tokenizer, cache_size=5)
print(dataset[0])
print(tokenizer.decode(dataset[0]["input_ids"]))
