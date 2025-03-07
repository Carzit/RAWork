import os
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Literal, Union, Optional, Callable, Any
import re

import pandas as pd
from tqdm import tqdm

class StaticDataExtractor:
    """提取静态数据的组件"""
    
    def __init__(self):
        self.shareholders:Dict[str, Dict] = {}  # {ShareHolderID: {metadata}}
        self.stock_industries:Dict[str, Dict] = {}  # {Symbol: {IndustryCode, IndustryName}}
    
    def process_file(self, file_path):
        """处理单个文件提取静态数据"""
        with pd.read_csv(file_path, chunksize=10000) as reader:
            for chunk in reader:
                # 处理股东数据
                for _, row in chunk.drop_duplicates("ShareHolderID").iterrows():
                    sh_id = row["ShareHolderID"]
                    if sh_id not in self.shareholders:
                        self.shareholders[sh_id] = {
                            "SystematicsID": row["SystematicsID"],
                            "FundID": row["FundID"],
                            "CategoryCode": row["CategoryCode"],
                            "ShareHolderName": row["ShareHolderName"]
                        }
                
                # 处理股票行业数据
                for _, row in chunk.drop_duplicates("Symbol").iterrows():
                    symbol = row["Symbol"]
                    if symbol not in self.stock_industries:
                        self.stock_industries[symbol] = {
                            "IndustryCode": row["IndustryCode"],
                            "IndustryName": row["IndustryName"]
                        }

    def postprocess(self):
        """后处理静态数据"""
        # 对于缺失的行业数据，drop掉
        self.stock_industries = {k: v for k, v in self.stock_industries.items() if not pd.isna(v["IndustryCode"]) and not pd.isna(v["IndustryName"])}
        
    
    def save(self, output_dir:str):
        """保存静态数据表"""

        self.postprocess() # 后处理静态数据

        Path(output_dir).mkdir(exist_ok=True)
        shareholders = pd.DataFrame(self.shareholders.values(), index=self.shareholders.keys())
        shareholders.index.name = "ShareHolderID"
        shareholders.to_csv(
            f"{output_dir}/shareholders.csv", index=True
        )
        stock_industries = pd.DataFrame(self.stock_industries.values(), index=self.stock_industries.keys())
        stock_industries.index.name = "Symbol"
        stock_industries.to_csv(
            f"{output_dir}/stock_industries.csv", index=True
        )

class HoldingsProcessor:
    """处理动态持股数据"""
    
    def __init__(self, static_dir, output_dir):
        self.static_sh = pd.read_csv(f"{static_dir}/shareholders.csv")
        self.valid_sh_ids = set(self.static_sh["ShareHolderID"])
        self.output_dir = Path(output_dir)
        self.buffer =  defaultdict(lambda: defaultdict(pd.DataFrame)) # {ShareHolderID: {EndDate: DataFrame}}
    
    def _process_single_chunk(self, chunk):
        """处理单个数据块"""
        # 类型转换与过滤
        chunk["Holdshares"] = pd.to_numeric(chunk["Holdshares"], errors="coerce")
        chunk = chunk[chunk["Holdshares"] > 0]
        
        # 按机构分组处理
        for sh_id, group in chunk.groupby("ShareHolderID"):
            

            if sh_id not in self.valid_sh_ids:
                continue
            
            # 按时间分组处理
            for end_date, date_group in group.groupby("EndDate"):
                if end_date not in self.buffer[sh_id].keys():
                    self.buffer[sh_id][end_date] = date_group
                else:
                    self.buffer[sh_id][end_date] = pd.concat([self.buffer[sh_id][end_date], date_group])
    
    def _format_holdings(self, data: pd.DataFrame) -> str:
        """将持股数据格式化为字符串：Symbol1,Holdshares1;Symbol2,Holdshares2;..."""
        sorted_data = data.sort_values("Holdshares", ascending=False)
        holdings_list = [f"{row['Symbol']}" for _, row in sorted_data.iterrows()]
        return f"[{', '.join(holdings_list)}]"
    
    def process_files(self, input_dir):
        """处理全部年度文件"""
        for file_path in Path(input_dir).glob("*.csv"):
            try:
                year = int(file_path.stem)
            except ValueError:
                continue
            
            with pd.read_csv(file_path, chunksize=100000, index_col=False) as reader:
                for chunk_idx, chunk in enumerate(reader):
                    print(f"Processing {year} - Chunk {chunk_idx}")
                    self._process_single_chunk(chunk)
        
        # 触发持久化操作
        self._flush_buffer()
    
    def _flush_buffer(self):
        """将缓存数据写入磁盘"""
        i = 0
        for sh_id in self.buffer.keys():
            # 获取机构元数据
            #metadata = self.static_sh[self.static_sh["ShareHolderID"] == sh_id].iloc[0]
            #safe_name = self._sanitize_filename(metadata["ShareHolderName"])
            
            # 对记录按时间排序
            records = [{"EndDate": end_date, "Holdings": self._format_holdings(data)} for end_date, data in self.buffer[sh_id].items()]
            sorted_records = sorted(records, key=lambda x: datetime.strptime(x["EndDate"], "%Y-%m-%d"))
            
            # 持久化
            output_path = self.output_dir / f"{sh_id}.csv"
            pd.DataFrame(sorted_records).to_csv(output_path, index=False)
            i+=1
            if i>10:
                break
    
    def _sanitize_filename(self, name):
        """清理非法文件名字符"""
        return re.sub(r'[\\/*?:"<>|]', "_", str(name)).strip()


class DataPipeline:
    """数据管道控制器"""
    
    def __init__(self, input_dir, static_dir="static", output_dir="output"):
        self.input_dir = input_dir
        self.static_dir = static_dir
        self.output_dir = output_dir

        self.statstic_data_strategy: Callable = None
        self.holdings_data_strategy: Callable = None
        
    def execute(self):
        # Step 1: 提取静态数据
        #static_extractor = StaticDataExtractor()
        #for file_path in Path(self.input_dir).glob("*.csv"):
        #    static_extractor.process_file(file_path)
        #static_extractor.save(self.static_dir)
        #print("静态数据提取完成")
        #sys.exit(0)
        # Step 2: 处理动态持仓数据
        processor = HoldingsProcessor(
            static_dir=self.static_dir,
            output_dir=self.output_dir
        )
        processor.process_files(self.input_dir)
        print("动态数据处理完成")

if __name__ == "__main__":
    pipeline = DataPipeline(
        input_dir=r"data\raw\AssetEmbedding\test",
        static_dir=r"data\raw\AssetEmbedding\test\static", 
        output_dir=r"data\raw\AssetEmbedding\test\output"
    )
    pipeline.execute()
