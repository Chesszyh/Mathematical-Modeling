"""
项目-国家维度奖牌预测模型
按运动项目和国家，基于历史奖牌数据预测未来奖牌数
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# 全局配置
CONFIG = {
    "data_path": {
        "medals": "summerOly_medal_counts.csv",
        "hosts": "summerOly_hosts.csv", 
        "programs": "summerOly_programs.csv",
        "athletes": "summerOly_athletes.csv"
    },
    "n_years": 3,  # 使用过去3年数据
    "min_medal_threshold": 1,  # 至少获得过多少奖牌的项目才参与建模
    "model_type": "rf",  # 可选 xgboost/rf/linear
    "output_dir": "./sport_level_results/"
}

def load_and_preprocess():
    """数据加载与预处理"""
    # 加载运动员数据并聚合到项目-国家-年层级
    athletes = pd.read_csv(CONFIG["data_path"]["athletes"])
    athletes['has_medal'] = athletes['Medal'] != 'No Medal'
    
    # 生成项目-国家-年奖牌统计
    medal_detail = athletes.groupby(['Sport', 'NOC', 'Year']).agg(
        Gold=('Medal', lambda x: (x == 'Gold').sum()),
        Silver=('Medal', lambda x: (x == 'Silver').sum()),
        Bronze=('Medal', lambda x: (x == 'Bronze').sum()),
        Total_Medals=('has_medal', 'sum')
    ).reset_index()
    
    # 加载项目存在性数据
    programs = pd.read_csv(CONFIG["data_path"]["programs"])
    programs = programs.melt(id_vars=['Sport', 'Discipline', 'Code'], 
                            value_vars=[str(y) for y in range(2000, 2029, 4)],
                            var_name='Year', value_name='Exists')
    programs['Year'] = programs['Year'].astype(int)
    programs = programs[programs['Exists'] == 1]
    
    # 合并存在性标记
    medal_detail = medal_detail.merge(
        programs[['Sport', 'Year']].drop_duplicates(),
        on=['Sport', 'Year'],
        how='left',
        indicator=True
    )
    medal_detail['Sport_Exists'] = (medal_detail['_merge'] == 'both').astype(int)
    medal_detail.drop('_merge', axis=1, inplace=True)
    
    return medal_detail

def create_lag_features(df, n_years=3):
    """生成滞后特征"""
    features = []
    df = df.sort_values(['Sport', 'NOC', 'Year'])
    
    for lag in range(1, n_years+1):
        # 奖牌数滞后
        df[f'Gold_lag{lag}'] = df.groupby(['Sport', 'NOC'])['Gold'].shift(lag)
        df[f'Silver_lag{lag}'] = df.groupby(['Sport', 'NOC'])['Silver'].shift(lag)
        df[f'Bronze_lag{lag}'] = df.groupby(['Sport', 'NOC'])['Bronze'].shift(lag)
        
        # 总奖牌滞后
        df[f'Total_lag{lag}'] = df.groupby(['Sport', 'NOC'])['Total_Medals'].shift(lag)
        
        features.extend([
            f'Gold_lag{lag}', f'Silver_lag{lag}',
            f'Bronze_lag{lag}', f'Total_lag{lag}'
        ])
    
    # 计算滑动窗口统计量
    df['3yr_avg'] = df.groupby(['Sport', 'NOC'])['Total_Medals'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    features.append('3yr_avg')
    
    return df.dropna(), features

def add_host_features(df):
    """添加东道主特征"""
    hosts = pd.read_csv(CONFIG["data_path"]["hosts"])
    hosts['host_country'] = hosts['Host'].str.split(',').str[-1].str.strip()
    hosts = hosts[['Year', 'host_country']].rename(columns={'host_country': 'NOC'})
    
    df = df.merge(
        hosts.assign(is_host=1),
        on=['Year', 'NOC'],
        how='left'
    )
    df['is_host'] = df['is_host'].fillna(0)
    return df

def train_model(X_train, y_train, features):
    """训练预测模型"""
    if CONFIG['model_type'] == 'xgboost':
        model = xgb.XGBRegressor(
            max_depth=4,
            learning_rate=0.1,
            n_estimators=100
        )
    elif CONFIG['model_type'] == 'rf':
        model = RandomForestRegressor(
            random_state=42,    
            n_estimators=200,       
            max_depth=5,
            min_samples_split=5,
            n_jobs=-1
        )
    elif CONFIG['model_type'] == 'linear':
        model = LinearRegression()
    
    model.fit(X_train[features], y_train)
    return model

def evaluate_model(model, X_test, y_test, features):
    """模型评估"""
    pred = model.predict(X_test[features])
    return {
        'rmse': np.sqrt(mean_squared_error(y_test, pred)),
        'mae': mean_absolute_error(y_test, pred),
        'avg_pred': np.mean(pred),
        'avg_true': np.mean(y_test)
    }

def main():
    # 数据管道
    df = load_and_preprocess()
    df = add_host_features(df)
    
    # 过滤有效数据：至少有过奖牌且项目存在的记录
    df = df[
        (df['Total_Medals'] >= CONFIG['min_medal_threshold']) &
        (df['Sport_Exists'] == 1)
    ]
    
    # 生成滞后特征
    df, features = create_lag_features(df, CONFIG['n_years'])
    features += ['is_host']
    
    # 时间序列分割
    tscv = TimeSeriesSplit(n_splits=3)
    results = []
    
    for train_idx, test_idx in tscv.split(df):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        
        # 训练模型
        model = train_model(train, train['Total_Medals'], features)
        
        # 评估
        metrics = evaluate_model(model, test, test['Total_Medals'], features)
        results.append(metrics)
    
    # 输出平均性能
    avg_metrics = pd.DataFrame(results).mean()
    print(f"模型平均性能 ({CONFIG['model_type']}):\n{avg_metrics}")
    
    # 全量训练与预测
    final_model = train_model(df, df['Total_Medals'], features)
    
    # 生成2028年预测
    future = df[df.Year == 2024].copy()
    future['Year'] = 2028
    
    # 更新滞后特征：将2024年数据作为lag1
    for lag in range(1, CONFIG['n_years']+1):
        future[f'Gold_lag{lag}'] = future.groupby(['Sport', 'NOC'])['Gold'].shift(lag-1)
        future[f'Silver_lag{lag}'] = future.groupby(['Sport', 'NOC'])['Silver'].shift(lag-1)
        future[f'Bronze_lag{lag}'] = future.groupby(['Sport', 'NOC'])['Bronze'].shift(lag-1)
        future[f'Total_lag{lag}'] = future.groupby(['Sport', 'NOC'])['Total_Medals'].shift(lag-1)
    
    # 添加东道主标记
    future['is_host'] = (future['NOC'] == 'United States').astype(int)
    
    # 预测
    future['pred_medals'] = final_model.predict(future[features])
    
    # 保存结果
    future[['Sport', 'NOC', 'Year', 'pred_medals']].to_csv(
        f"{CONFIG['output_dir']}sport_level_predictions.csv",
        index=False
    )

if __name__ == "__main__":
    # 切换到当前目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()