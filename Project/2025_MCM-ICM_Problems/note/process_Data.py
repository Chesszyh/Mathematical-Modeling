"""
数据准备与清洗示例:
1. 读取原始csv文件到DataFrame
2. 备份原始数据
3. 去重、处理缺失值、关键调试信息输出
4. 初步可视化呈现
请确保该脚本与所有csv文件位于同一文件夹下
"""
import pandas as pd
import os
import matplotlib.pyplot as plt

def main():
    # 当前脚本所在文件夹
    current_folder = os.path.dirname(os.path.abspath(__file__))

    # 1. 读取csv数据
    athletes_csv = os.path.join(current_folder, 'summerOly_athletes.csv')
    hosts_csv = os.path.join(current_folder, 'summerOly_hosts.csv')
    medal_csv = os.path.join(current_folder, 'summerOly_medal_counts.csv')
    programs_csv = os.path.join(current_folder, 'summerOly_programs.csv')

    df_athletes = pd.read_csv(athletes_csv)
    df_hosts = pd.read_csv(hosts_csv)
    df_medal = pd.read_csv(medal_csv)
    df_programs = pd.read_csv(programs_csv)

    print("===== 数据读取完成 =====")
    print("athletes 数据量:", df_athletes.shape)
    print("hosts 数据量:", df_hosts.shape)
    print("medal 数据量:", df_medal.shape)
    print("programs 数据量:", df_programs.shape)

    # 2. 创建备份副本
    df_athletes_backup = df_athletes.copy()
    df_hosts_backup = df_hosts.copy()
    df_medal_backup = df_medal.copy()
    df_programs_backup = df_programs.copy()

    print("\n===== 已创建副本备份 =====")

    # 3. 去重与缺失值处理
    # athletes
    print("\n===== athletes数据清洗前 =====")
    print(df_athletes.info())
    print(df_athletes.isnull().sum())

    # 去重
    before_dedupe = df_athletes.shape[0]
    df_athletes.drop_duplicates(inplace=True)
    after_dedupe = df_athletes.shape[0]
    print(f"athletes数据: 去重前共 {before_dedupe} 行, 去重后共 {after_dedupe} 行")

    # 缺失值填充或删除示例(根据业务需求可做不同策略)
    df_athletes.fillna({'Team':'Unknown_Team','Medal':'No medal'}, inplace=True)

    print("\n===== athletes数据清洗后 =====")
    print(df_athletes.info())
    print(df_athletes.isnull().sum())

    # hosts
    print("\n===== hosts数据清洗前 =====")
    print(df_hosts.info())
    print(df_hosts.isnull().sum())

    df_hosts.drop_duplicates(inplace=True)
    df_hosts.fillna("Unknown_Host", inplace=True)

    print("\n===== hosts数据清洗后 =====")
    print(df_hosts.info())
    print(df_hosts.isnull().sum())

    # medal
    print("\n===== medal数据清洗前 =====")
    print(df_medal.info())
    print(df_medal.isnull().sum())

    df_medal.drop_duplicates(inplace=True)
    df_medal.fillna({'Gold':0,'Silver':0,'Bronze':0,'Total':0}, inplace=True)

    print("\n===== medal数据清洗后 =====")
    print(df_medal.info())
    print(df_medal.isnull().sum())

    # programs
    print("\n===== programs数据清洗前 =====")
    print(df_programs.info())
    print(df_programs.isnull().sum())

    df_programs.drop_duplicates(inplace=True)
    df_programs.fillna(0, inplace=True)

    print("\n===== programs数据清洗后 =====")
    print(df_programs.info())
    print(df_programs.isnull().sum())

    # 4. 初步可视化: 以athletes为例, 简单绘制Medal列的数量分布
    medal_counts = df_athletes['Medal'].value_counts()
    plt.figure(figsize=(6,4))
    medal_counts.plot(kind='bar', color='skyblue')
    plt.title('Medal Distribution in Athletes Dataset')
    plt.xlabel('Medal Type')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(current_folder, 'medal_distribution.png'))
    plt.close()
    print("\n===== 已输出初步可视化图表: medal_distribution.png =====")

    # 将最终清洗后的数据另存为xxx_cleaned.csv
    df_athletes.to_csv(os.path.join(current_folder, 'summerOly_athletes_cleaned.csv'), index=False)
    df_hosts.to_csv(os.path.join(current_folder, 'summerOly_hosts_cleaned.csv'), index=False)
    df_medal.to_csv(os.path.join(current_folder, 'summerOly_medal_counts_cleaned.csv'), index=False)
    df_programs.to_csv(os.path.join(current_folder, 'summerOly_programs_cleaned.csv'), index=False)
    print("\n===== 清洗后数据已保存 =====")

if __name__ == "__main__":
    main()