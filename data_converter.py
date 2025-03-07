
import os
import json
import logging
from datetime import datetime
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Literal, Union, Optional, Callable, Any

import pandas as pd
from tqdm import tqdm
from dateutil import parser

import utils

_GLOBAL_DATE_FORMAT = "%Y-%m-%d"
_POSSIBLE_DATE_FORMATS = ["%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%Y%m"]

def date_parser(date_str):
    """
    自定义日期解析函数，支持多种日期格式。
        
    :param date_str: 原始日期字符串
    :return: 解析后的日期对象
    """
    for fmt in _POSSIBLE_DATE_FORMATS:  # 支持的日期格式列表
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported date format: {date_str}")

class DataConverter(ABC):
    """抽象基类，定义数据转换的接口"""
    
    def __init__(self, 
                 input_path:str, 
                 output_path:str):
        self.input_path:str = input_path
        self.output_path:str = output_path
        self.file_list:List[str] = []

        self.dates:List[str] = []
        self.stocks:List[str] = []
        self.factors:List[str] = []

        self.input_data_pool:Dict[str, pd.DataFrame] = {}
        self.output_data_pool:Dict[int, pd.DataFrame] = {}

    @abstractmethod
    def load_data(self, 
                  format:str,
                  **kwargs) -> None:
        # 读取指定文件夹中的所有文件，处理数据并存储在 factor_data 中，同时计算所有文件中共同的日期和股票代码。
        self.file_list = [f for f in os.listdir(self.input_path) if f.endswith(format)]
        
        for file_name in tqdm(self.file_list):
            file_path = os.path.join(self.input_path, file_name)
            self.input_data_pool.append(utils.load_dataframe(file_path, format, **kwargs))

    @abstractmethod
    def filter_data(self) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def merge_date(self, date:str) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def process(self, 
                format:Literal["csv", "pkl", "parquet", "feather"]="csv",
                **kwargs) -> None:
        logging.debug("merging data...")
        for date in tqdm(self.dates):
            merged_df = self.merge_date(date)
            parsed_date = parser.parse(date)
            utils.save_dataframe(df=merged_df,
                                 path=os.path.join(self.output_path, f"{parsed_date.strftime(_GLOBAL_DATE_FORMAT)}.{format}"),
                                 format=format,
                                 **kwargs)

class Format1Converter(DataConverter):
    """格式 1 转换器：以因子名为文件名，列名为股票代码，行名为日期"""
    
    def load_data(self, 
                  format:str,
                  **kwargs) -> None:
        self.file_list = [f for f in os.listdir(self.input_path) if f.endswith(format)]
        
        for file_name in tqdm(self.file_list, desc="Loading Data"):
            file_path = os.path.join(self.input_path, file_name)
            factor = file_name.split('.')[0]
            self.factors.append(factor)
            self.input_data_pool[factor] = utils.load_dataframe(file_path, format, **kwargs)

        self.factors.sort()

    def filter_data(self) -> pd.DataFrame:
        intersection_dates = list(set.intersection(*[set(df.index) for df in self.input_data_pool.values()]))
        intersection_stocks = list(set.intersection(*[set(df.columns) for df in self.input_data_pool.values()]))
        union_dates = list(set.union(*[set(df.index) for df in self.input_data_pool.values()]))
        union_stocks = list(set.union(*[set(df.columns) for df in self.input_data_pool.values()]))
        logging.info(f"{len(union_dates)-len(intersection_dates)} Dates dropped")
        logging.info(f"{len(union_stocks)-len(intersection_stocks)} Stocks dropped")

        self.dates = sorted(intersection_dates)
        self.stocks = sorted(intersection_stocks)

    def merge_date(self, date:str) -> pd.DataFrame:
        merged_df = pd.DataFrame(index=self.stocks, columns=self.factors)
        merged_df.index.name = 'stock'
        merged_df.columns.name = 'factor'

        for factor, df in self.input_data_pool.items():
            if date in df.index:
                series = df.loc[date]
                merged_df[factor] = series

        #merged_df.insert(0, 'date', date)
        #merged_df.reset_index(inplace=True, drop=False)
        return merged_df
    
    def process(self, 
                format:Literal["csv", "pkl", "parquet", "feather"]="csv",
                **kwargs) -> None:
        logging.debug("merging data...")
        for date in tqdm(self.dates):
            merged_df = self.merge_date(date)
            parsed_date = parser.parse(date)
            utils.save_dataframe(df=merged_df,
                                 path=os.path.join(self.output_path, f"{parsed_date.strftime(_GLOBAL_DATE_FORMAT)}.{format}"),
                                 format=format,
                                 **kwargs)
            
class Format2Converter(DataConverter):
    """以因子名为文件名，共有三列，分别为股票代码、日期、因子的数值"""
    def __init__(self, 
                 input_path:str, 
                 output_path:str,
                 stock_col:Union[str,int]=0,
                 date_col:Union[str,int]=1,
                 factor_col:Union[str,int]=2):
        super().__init__(input_path, output_path)
        self.stock_col:str = stock_col
        self.date_col:str = date_col
        self.factor_col:str = factor_col
    
    def _get_stock_series(self, df:pd.DataFrame) -> pd.Series:
        if isinstance(self.stock_col, int):
            return df.iloc[:, self.stock_col]
        else:
            return df[self.stock_col]
        
    def _get_date_series(self, df:pd.DataFrame) -> pd.Series:
        if isinstance(self.date_col, int):
            return df.iloc[:, self.date_col]
        else:
            return df[self.date_col]
        
    def _get_factor_series(self, df:pd.DataFrame, modif=0) -> pd.Series:
        if isinstance(self.factor_col, int):
            return df.iloc[:, self.factor_col+modif]
        else:
            return df[self.factor_col]

    def load_data(self, 
                  format:str,
                  **kwargs) -> None:
        self.file_list = [f for f in os.listdir(self.input_path) if f.endswith(format)]
        
        for file_name in tqdm(self.file_list):
            file_path = os.path.join(self.input_path, file_name)
            factor = file_name.split('.')[0]
            self.factors.append(factor)
            self.input_data_pool[factor] = utils.load_dataframe(file_path, format, **kwargs)

        self.factors.sort()

    def filter_data(self) -> pd.DataFrame:
        intersection_dates = list(set.intersection(*[set(self._get_date_series(df)) for df in self.input_data_pool.values()]))
        intersection_stocks = list(set.intersection(*[set(self._get_stock_series(df)) for df in self.input_data_pool.values()]))
        union_dates = list(set.union(*[set(self._get_date_series(df)) for df in self.input_data_pool.values()]))
        union_stocks = list(set.union(*[set(self._get_stock_series(df)) for df in self.input_data_pool.values()]))
        logging.info(f"{len(union_dates)-len(intersection_dates)} Dates dropped")
        logging.info(f"{len(union_stocks)-len(intersection_stocks)} Stocks dropped")

        self.dates = sorted(intersection_dates)
        self.stocks = sorted(intersection_stocks)

    def merge_date(self, date:str) -> pd.DataFrame:
        merged_df = pd.DataFrame(index=self.stocks, columns=self.factors)
        merged_df.index.name = 'stock'
        merged_df.columns.name = 'factor'

        for factor, df in self.input_data_pool.items():
            stock_col_name = self._get_stock_series(df).name
            series = self._get_factor_series(df[self._get_date_series(df) == date].set_index(stock_col_name), modif=-1)
            merged_df[factor] = series
        return merged_df
    
    def process(self, 
                format:Literal["csv", "pkl", "parquet", "feather"]="csv",
                **kwargs) -> None:
        logging.debug("merging data...")
        for date in tqdm(self.dates):
            merged_df = self.merge_date(date)
            parsed_date = date_parser(str(date))
            utils.save_dataframe(df=merged_df,
                                 path=os.path.join(self.output_path, f"{parsed_date.strftime(_GLOBAL_DATE_FORMAT)}.{format}"),
                                 format=format,
                                 **kwargs)
            break
            
class Format3Converter(DataConverter):
    """格式 3 转换器：以股票名或股票代码为文件名，列名为因子名，行名为日期"""
    
    def load_data(self, 
                  format:str,
                  **kwargs) -> None:
        self.file_list = [f for f in os.listdir(self.input_path) if f.endswith(format)]
        
        for file_name in tqdm(self.file_list, desc="Loading Data"):
            file_path = os.path.join(self.input_path, file_name)
            stock = file_name.split('.')[0]
            self.stocks.append(stock)
            self.input_data_pool[stock] = utils.load_dataframe(file_path, format, **kwargs)

        self.stocks.sort()

    def filter_data(self) -> pd.DataFrame:
        intersection_dates = list(set.intersection(*[set(df.index) for df in self.input_data_pool.values()]))
        intersection_factors = list(set.intersection(*[set(df.columns) for df in self.input_data_pool.values()]))
        union_dates = list(set.union(*[set(df.index) for df in self.input_data_pool.values()]))
        union_factors = list(set.union(*[set(df.columns) for df in self.input_data_pool.values()]))
        logging.info(f"{len(union_dates)-len(intersection_dates)} Dates dropped")
        logging.info(f"{len(union_factors)-len(intersection_factors)} Stocks dropped")

        self.dates = sorted(intersection_dates)
        self.factors = sorted(intersection_factors)

    def merge_date(self, date:str) -> pd.DataFrame:
        merged_df = pd.DataFrame(index=self.stocks, columns=self.factors)
        merged_df.index.name = 'stock'
        merged_df.columns.name = 'factor'

        for stock, df in self.input_data_pool.items():
            if date in df.index:
                series = df.loc[date]
                merged_df.loc[stock] = series
        return merged_df
    
    def process(self, 
                format:Literal["csv", "pkl", "parquet", "feather"]="csv",
                **kwargs) -> None:
        logging.debug("merging data...")
        for date in tqdm(self.dates):
            merged_df = self.merge_date(date)
            parsed_date = parser.parse(date)
            utils.save_dataframe(df=merged_df,
                                 path=os.path.join(self.output_path, f"{parsed_date.strftime(_GLOBAL_DATE_FORMAT)}.{format}"),
                                 format=format,
                                 **kwargs)

class Format4Converter(DataConverter):
    """以股票名或股票代码为文件名，共有三列，分别为日期、因子名、因子的数值"""
    def __init__(self, 
                 input_path:str, 
                 output_path:str,
                 date_col:Union[str,int]=0,
                 factor_name_col:Union[str,int]=1,
                 factor_value_col:Union[str,int]=2):
        super().__init__(input_path, output_path)
        self.date_col:str = date_col
        self.factor_name_col:str = factor_name_col
        self.factor_value_col:str = factor_value_col
        
    def _get_date_series(self, df:pd.DataFrame) -> pd.Series:
        if isinstance(self.date_col, int):
            return df.iloc[:, self.date_col]
        else:
            return df[self.date_col]
        
    def _get_factor_name_series(self, df:pd.DataFrame, modif=0) -> pd.Series:
        if isinstance(self.factor_col, int):
            return df.iloc[:, self.factor_name_col+modif]
        else:
            return df[self.factor_col]
        
    def _get_factor_value_series(self, df:pd.DataFrame, modif=0) -> pd.Series:
        if isinstance(self.factor_col, int):
            return df.iloc[:, self.factor_value_col+modif]
        else:
            return df[self.factor_col]

    def load_data(self, 
                  format:str,
                  **kwargs) -> None:
        self.file_list = [f for f in os.listdir(self.input_path) if f.endswith(format)]
        
        for file_name in tqdm(self.file_list):
            file_path = os.path.join(self.input_path, file_name)
            stock = file_name.split('.')[0]
            self.stocks.append(stock)
            self.input_data_pool[stock] = utils.load_dataframe(file_path, format, **kwargs)

        self.stocks.sort()

    def filter_data(self) -> pd.DataFrame:
        intersection_dates = list(set.intersection(*[set(self._get_date_series(df)) for df in self.input_data_pool.values()]))
        intersection_factors = list(set.intersection(*[set(self._get_factor_name_series(df)) for df in self.input_data_pool.values()]))
        union_dates = list(set.union(*[set(self._get_date_series(df)) for df in self.input_data_pool.values()]))
        union_factors = list(set.union(*[set(self._get_factor_name_series(df)) for df in self.input_data_pool.values()]))
        logging.info(f"{len(union_dates)-len(intersection_dates)} Dates dropped")
        logging.info(f"{len(union_factors)-len(intersection_factors)} Factors dropped")

        self.dates = sorted(intersection_dates)
        self.factors = sorted(intersection_factors)

    def merge_date(self, date:str) -> pd.DataFrame:
        merged_df = pd.DataFrame(index=self.stocks, columns=self.factors)
        merged_df.index.name = 'stock'
        merged_df.columns.name = 'factor'

        for stock, df in self.input_data_pool.items():
            series = self._get_factor_value_series(df[self._get_date_series(df) == date].set_index(self.factor_name_col), modif=-1)
            merged_df.loc[stock] = series
        return merged_df
    
    def process(self, 
                format:Literal["csv", "pkl", "parquet", "feather"]="csv",
                **kwargs) -> None:
        logging.debug("merging data...")
        for date in tqdm(self.dates):
            merged_df = self.merge_date(date)
            parsed_date = parser.parse(date)
            utils.save_dataframe(df=merged_df,
                                 path=os.path.join(self.output_path, f"{parsed_date.strftime(_GLOBAL_DATE_FORMAT)}.{format}"),
                                 format=format,
                                 **kwargs)

class Format5Converter(DataConverter):
    """格式 5 转换器：以日期为文件名，列名为股票代码，行名为因子名"""
    
    def load_data(self, 
                  format:str,
                  **kwargs) -> None:
        self.file_list = [f for f in os.listdir(self.input_path) if f.endswith(format)]
        
        for file_name in tqdm(self.file_list, desc="Loading Data"):
            file_path = os.path.join(self.input_path, file_name)
            date = file_name.split('.')[0]
            self.dates.append(date)
            self.input_data_pool[date] = utils.load_dataframe(file_path, format, **kwargs)

        self.dates.sort()

    def filter_data(self) -> pd.DataFrame:
        intersection_stocks = list(set.intersection(*[set(df.columns) for df in self.input_data_pool.values()]))
        intersection_factors = list(set.intersection(*[set(df.index) for df in self.input_data_pool.values()]))
        union_stocks = list(set.union(*[set(df.columns) for df in self.input_data_pool.values()]))
        union_factors = list(set.union(*[set(df.index) for df in self.input_data_pool.values()]))
        logging.info(f"Union Stocks - Intersection Stocks: {len(union_stocks)-len(intersection_stocks)}")
        logging.info(f"{len(union_factors)-len(intersection_factors)} Factors dropped")

        self.stocks = sorted(union_stocks)
        self.factors = sorted(intersection_factors)

    def merge_date(self, date:str) -> pd.DataFrame:
        df = self.input_data_pool[date]
        df = df.T
        df.index.name = 'stock'
        df.columns.name = 'factor'
        df = df.loc[:, self.factors]
        return df
    
    def process(self, 
                format:Literal["csv", "pkl", "parquet", "feather"]="csv",
                **kwargs) -> None:
        logging.debug("merging data...")
        for date in tqdm(self.dates):
            merged_df = self.merge_date(date)
            parsed_date = parser.parse(date)
            utils.save_dataframe(df=merged_df,
                                 path=os.path.join(self.output_path, f"{parsed_date.strftime(_GLOBAL_DATE_FORMAT)}.{format}"),
                                 format=format,
                                 **kwargs)

class Format6Converter(DataConverter):
    """格式 6 转换器：以日期为文件名，共有三列，分别为股票代码、因子名、因子的数值"""

    def __init__(self, 
                 input_path:str, 
                 output_path:str,
                 stock_col:Union[str,int]=0,
                 factor_name_col:Union[str,int]=1,
                 factor_value_col:Union[str,int]=2):
        super().__init__(input_path, output_path)
        self.stock_col:Union[str,int] = stock_col
        self.factor_name_col:Union[str,int] = factor_name_col
        self.factor_value_col:Union[str,int] = factor_value_col

    def _get_stock_series(self, df:pd.DataFrame) -> pd.Series:
        if isinstance(self.stock_col, int):
            return df.iloc[:, self.stock_col]
        else:
            return df[self.stock_col]
        
    def _get_factor_name_series(self, df:pd.DataFrame, modif=0) -> pd.Series:
        if isinstance(self.factor_col, int):
            return df.iloc[:, self.factor_name_col+modif]
        else:
            return df[self.factor_col]
        
    def _get_factor_value_series(self, df:pd.DataFrame, modif=0) -> pd.Series:
        if isinstance(self.factor_col, int):
            return df.iloc[:, self.factor_value_col+modif]
        else:
            return df[self.factor_col]

    def load_data(self, 
                  format:str,
                  **kwargs) -> None:
        self.file_list = [f for f in os.listdir(self.input_path) if f.endswith(format)]
        
        for file_name in tqdm(self.file_list):
            file_path = os.path.join(self.input_path, file_name)
            date = file_name.split('.')[0]
            self.dates.append(date)
            self.input_data_pool[date] = utils.load_dataframe(file_path, format, **kwargs)

        self.dates.sort()

    def filter_data(self) -> pd.DataFrame:
        intersection_stocks = list(set.intersection(*[set(self._get_stock_series(df)) for df in self.input_data_pool.values()]))
        intersection_factors = list(set.intersection(*[set(self._get_factor_name_series(df)) for df in self.input_data_pool.values()]))
        union_stocks = list(set.union(*[set(self._get_stock_series(df)) for df in self.input_data_pool.values()]))
        union_factors = list(set.union(*[set(self._get_factor_name_series(df)) for df in self.input_data_pool.values()]))
        logging.info(f"Union Stocks - Intersection Stocks: {len(union_stocks)-len(intersection_stocks)}")
        logging.info(f"{len(union_factors)-len(intersection_factors)} Factors dropped")

        self.stocks = sorted(union_stocks)
        self.factors = sorted(intersection_factors)

    def merge_date(self, date:str) -> pd.DataFrame:
        merged_df = pd.DataFrame(index=self.stocks, columns=self.factors)
        merged_df.index.name = 'stock'
        merged_df.columns.name = 'factor'

        for stock, df in self.input_data_pool.items():
            series = self._get_factor_value_series(df[self._get_date_series(df) == date].set_index(self.factor_name_col), modif=-1)
            merged_df.loc[stock] = series
        return merged_df
    
    def process(self, 
                format:Literal["csv", "pkl", "parquet", "feather"]="csv",
                **kwargs) -> None:
        logging.debug("merging data...")
        for date in tqdm(self.dates):
            merged_df = self.merge_date(date)
            parsed_date = parser.parse(date)
            utils.save_dataframe(df=merged_df,
                                 path=os.path.join(self.output_path, f"{parsed_date.strftime(_GLOBAL_DATE_FORMAT)}.{format}"),
                                 format=format,
                                 **kwargs)

class Format7Converter(DataConverter):
    """格式 7 转换器：一整张大表，列名包括股票代码、日期和各因子名"""

    def __init__(self,
                    input_path:str,
                    output_path:str,
                    stock_col:Union[str,int]=0,
                    date_col:Union[str,int]=1,
                    factor_col:Union[List[str], List[int]]=[2]):
            super().__init__(input_path, output_path)
            self.stock_col:Union[str,int] = stock_col
            self.date_col:Union[str,int] = date_col
            self.factor_col:Union[List[str], List[int]] = factor_col
    
    def load_data(self, 
                  format:str="csv",# 目前仅支持csv格式
                  avg_row_size:int=None,
                  memory_fraction:float=0.5,
                  **kwargs) -> None:
        chunk_size = utils.calculate_chunk_size(self.inputpath, avg_row_size, memory_fraction)
        self.input_data_pool = utils.load_dataframe(path=self.input_path, 
                                                    format=format, 
                                                    chunksize=chunk_size, 
                                                    use_cols=[self.stock_col, self.date_col, *self.factor_col],
                                                    parse_dates=['date'], 
                                                    date_parser=date_parser)

    def filter_data(self) -> pd.DataFrame:
        unique_dates = set()
        unique_stocks = set()
        for _, chunk in self.input_data_pool:
            chunk:pd.DataFrame
            unique_dates.update(chunk[self.date_col].unique())
            unique_stocks.update(chunk[self.stock_col].unique())
        self.dates = sorted(list(unique_dates))
        self.stocks = sorted(list(unique_stocks))
        self.factors = self.factor_col

    def merge_date(self, date:str) -> pd.DataFrame:
        pass
    
    def process(self, 
                format:Literal["csv", "pkl", "parquet", "feather"]="csv",
                **kwargs) -> None:
        date_dfs = {}

        for chunk in tqdm(self.input_data_pool, desc='Reading', unit='chunk'):
            grouped = chunk.groupby('date')
            for date, group in grouped:
                if date not in date_dfs:
                    date_dfs[date] = group
                else:
                    date_dfs[date] = pd.concat([date_dfs[date], group], ignore_index=True)

        for date, df in tqdm(date_dfs.items(), desc='Saving', unit='file'):
            utils.save_dataframe(df=df,
                                 path=os.path.join(self.output_path, f"{date.strftime(_GLOBAL_DATE_FORMAT)}.{format}"),
                                 format=format,
                                 **kwargs)
            break


class ConversionPipeline:
    """转换管道，根据输入文件格式选择合适的转换器"""
    
    FORMAT_MAPPING = {
        1: Format1Converter,
        2: Format2Converter,
        3: Format3Converter,
        4: Format4Converter,
        5: Format5Converter,
        6: Format6Converter,
        7: Format7Converter
    }
    
    def __init__(self, 
                 format_type:int, 
                 input_path:str, 
                 output_path:str,
                 **kwargs):
        self.converter:DataConverter = self.FORMAT_MAPPING[format_type](input_path, output_path, **kwargs)
        self.config = {
            "raw_data_format_type": format_type,
            "input_path": input_path,
            "output_path": output_path,
            "input_file_format": None,
            "input_kwargs": None,
            "output_file_format": None,
            "output_kwargs": None,
            "dates": None,
            "stocks": None,
            "factors": None
        }

    def load_data(self, format:Literal["csv", "pkl", "parquet", "feather"]="pkl", **kwargs) -> None:
        self.converter.load_data(format, **kwargs)
        self.converter.filter_data()
        self.config["input_file_format"] = format
        self.config["input_kwargs"] = kwargs
    
    def process(self, format:Literal["csv", "pkl", "parquet", "feather"]="pkl", **kwargs) -> None:
        self.converter.process(format, **kwargs)
        self.config["output_file_format"] = format
        self.config["output_kwargs"] = kwargs

        self.config["dates"] = self.converter.dates
        self.config["stocks"] = self.converter.stocks
        self.config["factors"] = self.converter.factors

    def save_config(self) -> None:
        with open(os.path.join(self.config["output_path"], "config.json"), 'w') as f:
            json.dump(self.config, f, indent=2)


if __name__ == "__main__":
    
    cp = ConversionPipeline(
        format_type=7,
        input_path=r'data\CRSP_compustat_merge\CRSP_compustat_merge.csv',
        output_path=r'data\test_save',
        stock_col="permno",
        date_col="date",
        factor_col=["ret", "shrout", "altprc", "primexch", "siccd", "mktcap", "exchange", "industry", "rf","ret_excess", "seq", "ceq", "at", "lt", "txditc", "txdb", "itcb", "mib", "pstkrv", "pstkl", "pstk", "capx", "oancf", "sale", "cogs", "xint", "xsga", "se", "ps", "dt", "be"])
    cp.load_data(format='csv', index_col=False)
    cp.process(format="csv")
    cp.save_config()

# ['permno','yyyymm', 'AM', 'AOP', 'AbnormalAccruals', 'Accruals', 'AccrualsBM', 'Activism1', 'Activism2', 'AdExp', 'AgeIPO', 'AnalystRevision', 'AnalystValue', 'AnnouncementReturn', 'AssetGrowth', 'BM', 'BMdec', 'BPEBM', 'Beta', 'BetaFP', 'BetaLiquidityPS', 'BetaTailRisk', 'BidAskSpread', 'BookLeverage', 'BrandInvest', 'CBOperProf', 'CF', 'CPVolSpread', 'Cash', 'CashProd', 'ChAssetTurnover', 'ChEQ', 'ChForecastAccrual', 'ChInv', 'ChInvIA', 'ChNAnalyst', 'ChNNCOA', 'ChNWC', 'ChTax', 'ChangeInRecommendation', 'CitationsRD', 'CompEquIss', 'CompositeDebtIssuance', 'ConsRecomm', 'ConvDebt', 'CoskewACX', 'Coskewness', 'CredRatDG', 'CustomerMomentum', 'DebtIssuance', 'DelBreadth', 'DelCOA', 'DelCOL', 'DelDRC', 'DelEqu', 'DelFINL', 'DelLTI', 'DelNetFin', 'DivInit', 'DivOmit', 'DivSeason', 'DivYieldST', 'DolVol', 'DownRecomm', 'EBM', 'EP', 'EarnSupBig', 'EarningsConsistency', 'EarningsForecastDisparity', 'EarningsStreak', 'EarningsSurprise', 'EntMult', 'EquityDuration', 'ExchSwitch', 'ExclExp', 'FEPS', 'FR', 'FirmAge', 'FirmAgeMom', 'ForecastDispersion', 'Frontier', 'GP', 'Governance', 'GrAdExp', 'GrLTNOA', 'GrSaleToGrInv', 'GrSaleToGrOverhead', 'Herf', 'HerfAsset', 'HerfBE', 'High52', 'IO_ShortInterest', 'IdioVol3F', 'IdioVolAHT', 'Illiquidity', 'IndIPO', 'IndMom', 'IndRetBig', 'IntMom', 'IntanBM', 'IntanCFP', 'IntanEP', 'IntanSP', 'InvGrowth', 'InvestPPEInv', 'Investment', 'LRreversal', 'Leverage', 'MRreversal', 'MS', 'MaxRet', 'MeanRankRevGrowth', 'Mom12m', 'Mom12mOffSeason', 'Mom6m', 'Mom6mJunk', 'MomOffSeason', 'MomOffSeason06YrPlus', 'MomOffSeason11YrPlus', 'MomOffSeason16YrPlus', 'MomRev', 'MomSeason', 'MomSeason06YrPlus', 'MomSeason11YrPlus', 'MomSeason16YrPlus', 'MomSeasonShort', 'MomVol', 'NOA', 'NetDebtFinance', 'NetDebtPrice', 'NetEquityFinance', 'NetPayoutYield', 'NumEarnIncrease', 'OPLeverage', 'OScore', 'OperProf', 'OperProfRD', 'OptionVolume1', 'OptionVolume2', 'OrderBacklog', 'OrderBacklogChg', 'OrgCap', 'PS', 'PatentsRD', 'PayoutYield', 'PctAcc', 'PctTotAcc', 'PredictedFE', 'PriceDelayRsq', 'PriceDelaySlope', 'PriceDelayTstat', 'ProbInformedTrading', 'RD', 'RDAbility', 'RDIPO', 'RDS', 'RDcap', 'REV6', 'RIO_Disp', 'RIO_MB', 'RIO_Turnover', 'RIO_Volatility', 'RIVolSpread', 'RealizedVol', 'Recomm_ShortInterest', 'ResidualMomentum', 'ReturnSkew', 'ReturnSkew3F', 'RevenueSurprise', 'RoE', 'SP', 'ShareIss1Y', 'ShareIss5Y', 'ShareRepurchase', 'ShareVol', 'ShortInterest', 'SmileSlope', 'Spinoff', 'SurpriseRD', 'Tax', 'TotalAccruals', 'TrendFactor', 'UpRecomm', 'VarCF', 'VolMkt', 'VolSD', 'VolumeTrend', 'XFIN', 'betaVIX', 'cfp', 'dCPVolSpread', 'dNoa', 'dVolCall', 'dVolPut', 'fgr5yrLag', 'grcapx', 'grcapx3y', 'hire', 'iomom_cust', 'iomom_supp', 'realestate', 'retConglomerate', 'roaq', 'sfe', 'sinAlgo', 'skew1', 'std_turn', 'tang', 'zerotrade12M', 'zerotrade1M', 'zerotrade6M']