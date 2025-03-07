import os
import pandas as pd
from pathlib import Path


class CSVFileMerger:
    def __init__(self, input_dir, output_dir, max_file_size_mb=100):
        """
        初始化CSV文件合并器
        :param input_dir: 输入目录，包含所有CSV文件
        :param output_dir: 输出目录，存储合并后的CSV文件
        :param max_file_size_mb: 单个输出文件的最大大小（单位：MB）
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.max_file_size_mb = max_file_size_mb
        self.output_dir.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在

    def _get_file_size_mb(self, file_path):
        """获取文件大小（单位：MB）"""
        return os.path.getsize(file_path) / (1024 * 1024)

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
    input_dir = "path/to/input_directory"  # 输入目录，包含所有CSV文件
    output_dir = "path/to/output_directory"  # 输出目录
    merger = CSVFileMerger(input_dir, output_dir, max_file_size_mb=500)  # 设置最大文件大小为500MB
    merger.run()
