import os
import json
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Dict, Literal, Union, Optional, Callable, Any
from tqdm import tqdm

import utils


class ConvertedDataProcessor:
    def __init__(self, converted_data_folder:str, output_folder:str):
        
        config_path = os.path.join(converted_data_folder, 'config.json')
        self.config:dict = utils.read_configs(config_path)
        self.inuput_folder:str = converted_data_folder
        self.output_folder = output_folder

        self.common_dates:List = []
        self.common_stocks:List = []
        self.common_factors:List = []

    def intersection(self, other:"ConvertedDataProcessor", attr:Literal["dates", "stocks", "factors"])->List:
        return sorted(list(set(self.config[attr]) & set(other.config[attr])))

    def __and__(self, other:"ConvertedDataProcessor")->Tuple[List, List, List]:
        common_dates = self.intersection(other, "dates")
        common_stocks = self.intersection(other, "stocks")
        common_factors = self.intersection(other, "factors")
        return common_dates, common_stocks, common_factors
    
    def process(self, 
                format:Literal["csv", "pkl", "parquet", "feather"]="csv",
                **kwargs)->None:
        """
        Process the converted data
        对文件夹下每个config['dates']中的日期为文件名，congfig["output_file_format"]为文件格式的文件进行处理
        保留每个文件的在common_stocks的行和common_factors的列
        """
        for date in self.common_dates:
            file_path = os.path.join(self.inuput_folder, f"{date}.{self.config['output_file_format']}")
            if os.path.exists(file_path):
                df = utils.load_dataframe(file_path, format=self.config['output_file_format'], **kwargs)
                df = df.loc[self.common_stocks, self.common_factors]
                utils.save_dataframe(df, 
                                     os.path.join(self.output_folder, f"{date}.{format}"), 
                                     format=format, 
                                     **kwargs)



class DataLoader:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.config = self._load_config()
        self.data = self._load_csv_files()

    def _load_config(self):
        """加载 config.json 文件"""
        config_path = os.path.join(self.folder_path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        return config

    def _load_csv_files(self):
        """加载文件夹中的所有 CSV 文件"""
        data = {}
        for date in self.config["dates"]:
            file_path = os.path.join(self.folder_path, f"{date}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col=0)
                data[date] = df
        return data

    def get_config(self):
        """获取配置文件"""
        return self.config

    def get_data(self):
        """获取加载的 CSV 数据"""
        return self.data
    
class DataAligner:
    def __init__(self, loader1, loader2):
        self.loader1 = loader1
        self.loader2 = loader2
        self.common_dates = None
        self.common_stocks = None
        self.common_factors = None
        self.aligned_data = None

    def align(self):
        """对齐两个数据集"""
        config1 = self.loader1.get_config()
        config2 = self.loader2.get_config()
        data1 = self.loader1.get_data()
        data2 = self.loader2.get_data()

        # 获取共同的日期、股票代码和因子
        self.common_dates = sorted(list(set(config1["dates"]) & set(config2["dates"])))
        self.common_stocks = sorted(list(set(config1["stocks"]) & set(config2["stocks"])))
        self.common_factors = sorted(list(set(config1["factors"]) & set(config2["factors"])))

        # 对齐数据
        self.aligned_data = {}
        for date in self.common_dates:
            if date in data1 and date in data2:
                df1 = data1[date].loc[self.common_stocks, self.common_factors]
                df2 = data2[date].loc[self.common_stocks, self.common_factors]
                aligned_df = pd.concat([df1, df2], axis=1)
                self.aligned_data[date] = aligned_df

    def get_aligned_data(self):
        """获取对齐后的数据"""
        return self.aligned_data

    def get_common_info(self):
        """获取共同的日期、股票代码和因子"""
        return self.common_dates, self.common_stocks, self.common_factors
    
def main(folder1, folder2):
    # 加载数据
    loader1 = DataLoader(folder1)
    loader2 = DataLoader(folder2)

    # 对齐数据
    aligner = DataAligner(loader1, loader2)
    aligner.align()

    # 获取对齐后的数据和共同信息
    aligned_data = aligner.get_aligned_data()
    common_dates, common_stocks, common_factors = aligner.get_common_info()

    # 输出对齐后的数据
    for date in common_dates:
        print(f"Date: {date}")
        print(aligned_data[date])
        print("\n")

    return aligned_data, common_dates, common_stocks, common_factors

if __name__ == "__main__":
    folder1 = "path_to_folder1"
    folder2 = "path_to_folder2"
    aligned_data, common_dates, common_stocks, common_factors = main(folder1, folder2)