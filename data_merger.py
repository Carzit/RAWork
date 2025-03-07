import os
import pandas as pd
from pathlib import Path
from typing import Callable


class CSVFileMerger:
    def __init__(self, 
                 input_dir:str, 
                 output_dir:str, 
                 max_file_size_mb:int=100,
                 filter_func: Callable[[pd.DataFrame], pd.DataFrame] = None):
        """
        初始化CSV文件合并器
        :param input_dir: 输入目录，包含所有CSV文件
        :param output_dir: 输出目录，存储合并后的CSV文件
        :param max_file_size_mb: 单个输出文件的最大大小（单位：MB）
        :param filter_func: 过滤函数，用于定义合并时保留行的逻辑
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在

        self.max_file_size_mb:int = max_file_size_mb # 单个输出文件的最大大小（单位：MB）
        self.filter_func:Callable[[pd.DataFrame], pd.DataFrame] = filter_func # 过滤函数
        

    def _get_file_size_mb(self, file_path):
        """获取文件大小（单位：MB）"""
        return os.path.getsize(file_path) / (1024 * 1024)

    def _filter_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        过滤DataFrame中的行
        :param df: 输入的DataFrame
        :return: 过滤后的DataFrame
        """
        if self.filter_func is not None:
            return self.filter_func(df)
        return df

    def merge_files(self):
        """合并CSV文件"""
        file_count = 0  # 输出文件计数器
        merged_df = pd.DataFrame()  # 当前合并的数据
        current_size_mb = 0  # 当前合并文件的大小（单位：MB）

        # 遍历所有输入文件
        csv_files = list(self.input_dir.glob("*.csv"))
        for csv_file in csv_files:
            # 读取单个CSV文件
            df = pd.read_csv(csv_file)
            # 过滤行
            df = self._filter_rows(df)
            # 将当前文件数据添加到合并的DataFrame
            merged_df = pd.concat([merged_df, df], ignore_index=True)
            # 计算当前合并文件的大小
            current_size_mb = merged_df.memory_usage(deep=True).sum() / (1024 * 1024)

            # 如果文件大小超过限制，保存当前合并文件并重置
            if current_size_mb >= self.max_file_size_mb:
                output_file = self.output_dir / f"merged_{file_count}.csv"
                merged_df.to_csv(output_file, index=False)
                print(f"Saved {output_file} with size {current_size_mb:.2f} MB")
                # 重置
                merged_df = pd.DataFrame()
                current_size_mb = 0
                file_count += 1

        # 保存剩余数据
        if not merged_df.empty:
            output_file = self.output_dir / f"merged_{file_count}.csv"
            merged_df.to_csv(output_file, index=False)
            print(f"Saved {output_file} with size {current_size_mb:.2f} MB")

    def run(self):
        """运行合并任务"""
        print(f"Starting merge task with max file size {self.max_file_size_mb} MB")
        self.merge_files()
        print("Merge task completed.")


# 示例用法
if __name__ == "__main__":
    input_dir = r"data\raw\AssetEmbedding\test\output"  # 输入目录，包含所有CSV文件
    output_dir = r"data\raw\AssetEmbedding\test\merge"  # 输出目录

    # 定义过滤函数(这里定义的是过滤Holdings列中数字元素个数不足2的行，也可以定义筛选其他条件的函数，例如在此基础上只保留持仓时间大于某个阈值的行、或者只保留某些机构（例如国家、政府背景的机构）的行)
    def filter_holdings(df: pd.DataFrame, n: int = 2) -> pd.DataFrame:
        """
        过滤Holdings列中数字元素个数不足n的行
        :param df: 输入的DataFrame
        :param n: 最小数字元素个数
        :return: 过滤后的DataFrame
        """
        def count_elements(holding_str: str) -> int:
            return len(holding_str.split(", ")) if isinstance(holding_str, str) else 0

        return df[df["Holdings"].apply(count_elements) >= n]

    # 创建CSVFileMerger实例并设置过滤函数
    merger = CSVFileMerger(input_dir, output_dir, max_file_size_mb=500, filter_func=filter_holdings)
    merger.run()
