import pandas as pd
import os
from datetime import datetime

def log_info(message):
    """输出日志信息"""
    print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")

def filter_medals_by_year(year_threshold=2000):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    input_file_list = ['summerOly_athletes.csv', 'summerOly_hosts.csv', 'summerOly_medal_counts.csv' ]
    for input_file in input_file_list:
        # 读取CSV文件
        log_info(f"正在读取文件: {input_file}")
        df = pd.read_csv(input_file)
        
        # 记录原始行数
        original_rows = len(df)
        
        # 过滤2000年及之后的数据
        df_filtered = df[df['Year'] >= year_threshold]
        
        # 记录过滤后的行数
        filtered_rows = len(df_filtered)
        
        # 保存文件
        output_file = f"filtered_{year_threshold}_{input_file}"
        
        # 保存过滤后的数据
        df_filtered.to_csv(output_file, index=False)
        
        log_info(f"过滤完成:")
        log_info(f"原始数据行数: {original_rows}")
        log_info(f"过滤后行数: {filtered_rows}")
        log_info(f"已删除 {original_rows - filtered_rows} 行")
        log_info(f"新文件已保存为: {output_file}")        


if __name__ == "__main__":
    filter_medals_by_year()