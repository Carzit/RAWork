import os
import ast
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

class MyCorpus:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = sorted(os.listdir(data_dir))  # 确保按时间顺序加载
    
    def __iter__(self):
        for file in self.files:
            file_path = os.path.join(self.data_dir, file)
            df = pd.read_csv(file_path)
            for holdings in df["Holdings"]:
                try:
                    stock_ids = ast.literal_eval(holdings)  # 解析字符串列表
                    yield ["[CLS]"]+[str(stock_id) for stock_id in stock_ids]+["[SEP]"]  # 作为word2vec的输入, 这里考虑到BERT的输入添加了特殊token
                except Exception as e:
                    print(f"解析失败 {holdings}: {e}")

# 第一步：构建词汇表
def build_vocab(corpus:MyCorpus):
    vocab = set()
    for stocks in corpus:
        vocab.update(stocks)
    return list(vocab)

# 数据目录路径
data_dir = r"data\preprocess\AssetEmbedding2019-2024\merged"

# 生成词汇表
corpus = MyCorpus(data_dir)
vocab_list = build_vocab(corpus)

# 训练Word2Vec
model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, sg=1, workers=4)

# 保存模型
model.save("word2vec_stock.model")

# 示例：获取某个股票的embedding
print(model.wv["600614"])
