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

CONFIG = {
    "data_path": {
        "medals": "summerOly_athletes.csv",
        "hosts": "summerOly_hosts.csv", 
        "programs": "summerOly_programs.csv",
        "athletes": "summerOly_athletes.csv"
    },
    "model_params": {
        "rf": { # 随机森林参数
            "n_estimators": 200,
            "max_depth": 5,
            "min_samples_split": 5,
            "random_state": 42
        },
        "gm": { # 灰色预测模型参数
            "n_predict": 1,
            "background_value": 0.5
        },
        "xgb": { # XGBoost参数
            "max_depth": 4,
            "learning_rate": 0.1,
            "n_estimators": 100
        }
    },
    "features": {
        "n_years": 3,  # 使用前n年数据
    },
    "output_dir": "output/"
}

def gm11(x, n_predict=1):
    """改进的灰色预测模型"""
    try:
        x = np.array(x, dtype=float)
        # 数据验证
        if len(x) < 3 or np.all(x == 0):
            return [np.mean(x)] * n_predict
        
        x1 = np.cumsum(x)
        z1 = (x1[:-1] + x1[1:]) / 2.0
        z1 = z1.reshape((len(z1), 1))
        B = np.append(-z1, np.ones_like(z1), axis=1)
        Y = x[1:].reshape((len(x)-1, 1))
        
        try:
            [[a], [b]] = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Y)
            predictions = []
            for i in range(n_predict):
                next_value = (x[0] - b/a) * np.exp(-a * (len(x)+i)) + b/a
                predictions.append(max(0, next_value))  # 确保预测值非负
            return predictions
        except np.linalg.LinAlgError:
            # 矩阵求逆失败时使用简单移动平均
            return [np.mean(x[-3:])] * n_predict
    except Exception as e:
        print(f"GM(1,1)预测失败: {str(e)}")
        return [x[-1]] * n_predict  # 使用最后一个值

def read_data(filename):
    encoding_list = ['utf-8', 'gbk', 'gb2312']
    for encoding in encoding_list:
        try:
            return pd.read_csv(filename, encoding=encoding)
        except UnicodeDecodeError:
            pass
    raise UnicodeDecodeError("无法解析文件编码")

def load_and_preprocess():
    """数据加载与预处理：项目-国家-年
    1. 自变量：前N年奖牌数
    2. 因变量：当前年奖牌数"""
    # 加载运动员数据并聚合到项目-国家-年层级
    athletes = read_data(CONFIG["data_path"]["athletes"]) # summerOly_athletes.csv
    athletes['has_medal'] = athletes['Medal'] != 'No medal'
    
    # 生成项目-国家-年奖牌统计
    medal_detail = athletes.groupby(['Sport', 'NOC', 'Year']).agg(
        Gold=('Medal', lambda x: (x == 'Gold').sum()),
        Silver=('Medal', lambda x: (x == 'Silver').sum()),
        Bronze=('Medal', lambda x: (x == 'Bronze').sum()),
        Total_Medals=('has_medal', 'sum')
    ).reset_index()
    
    # 加载项目存在性数据
    programs = read_data(CONFIG["data_path"]["programs"])

    # 获取实际的年份列
    # KeyError: "The following 'value_vars' are not present in the DataFrame: ['1916', '1940', '1944', '2028']" -- 战争年份没开奥运会，或者不能预测未来
    year_columns = [col for col in programs.columns if col.isdigit() and 1896 <= int(col) <= 2024]
    
    # 使用实际存在的年份列进行melt
    programs = programs.melt(id_vars=['Sport', 'Discipline', 'Code'],
                            value_vars=year_columns,
                            var_name='Year', 
                            value_name='Exists')
    
    # 清洗Exists列数据
    programs['Exists'] = pd.to_numeric(programs['Exists'], errors='coerce').fillna(0).astype(int) # 将非数值转为0
    programs['Year'] = programs['Year'].astype(int)
    programs = programs[programs['Exists'] == 1]
    
    # 合并存在性标记
    medal_detail = medal_detail.merge(
        programs[['Sport', 'Year']].drop_duplicates(),
        on=['Sport', 'Year'],
        how='left', # 保留所有medal_detail记录
        indicator=True
    )
    medal_detail['Sport_Exists'] = (medal_detail['_merge'] == 'both').astype(int)
    medal_detail.drop('_merge', axis=1, inplace=True)
    
    # 合并后结构：
    # Sport   NOC    Year    Gold  Silver  Bronze  Sport_Exists - 举办了则为1，否则为0
    
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

def create_features(df, n_years=3):
    """创建特征"""
    features = []
    df = df.sort_values(['Sport', 'NOC', 'Year'])
    
    for lag in range(1, n_years+1):
        for medal in ['Gold', 'Silver', 'Bronze']:
            col = f'{medal}_lag{lag}'
            df[col] = df.groupby(['Sport', 'NOC'])[medal].shift(lag)
            features.append(col)
    
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

def train_models(df, features):
    """训练模型"""
    models = {
        'rf': RandomForestRegressor(**CONFIG['model_params']['rf']),
        'xgb': xgb.XGBRegressor(**CONFIG['model_params']['xgb']),
        'gm': None  # GM模型不需要初始化
    }
    
    results = {}
    for name, model in models.items():
        if name != 'gm':
            model.fit(df[features], df['Total_Medals'])
        results[name] = model
    
    return results

def predict(models, X, history, features):
    """改进的预测函数"""
    predictions = {}
    
    # 机器学习模型预测
    for name, model in models.items():
        if name != 'gm':
            predictions[name] = model.predict(X[features])
    
    # 灰色预测
    gm_pred = []
    for _, group in history.groupby(['Sport', 'NOC']):
        if len(group) >= 3:  # 确保有足够数据
            pred = gm11(group['Total_Medals'].values[-4:],  # 只使用最近4个数据点
                        CONFIG['model_params']['gm']['n_predict'])[0]
        else:
            pred = group['Total_Medals'].values[-1]  # 使用最后一个值
        gm_pred.append(pred)
    predictions['gm'] = np.array(gm_pred)
    
    return predictions

def evaluate_model(model, X_test, y_test, features, model_name=''):
    """改进的模型评估函数"""
    # GM模型特殊处理
    if model_name == 'gm':
        gm_preds = []
        for _, row in X_test.iterrows():
            sport, noc = row['Sport'], row['NOC']
            history = X_test[
                (X_test['Sport'] == sport) & 
                (X_test['NOC'] == noc)
            ]['Total_Medals'].values[-4:]
            pred = gm11(history)[0]
            gm_preds.append(pred)
        pred = np.array(gm_preds)
    else:
        # 机器学习模型正常预测
        pred = model.predict(X_test[features])
    
    return {
        'model': model_name,
        'rmse': np.sqrt(mean_squared_error(y_test, pred)),
        'mae': mean_absolute_error(y_test, pred),
        'avg_pred': np.mean(pred),
        'avg_true': np.mean(y_test)
    }

def main():
    # 加载数据
    df = load_and_preprocess()
    
    # 创建特征
    df, features = create_features(df, CONFIG['features']['n_years'])
    
    # 训练测试分割
    tscv = TimeSeriesSplit(n_splits=3)
    results = []
    
    for train_idx, test_idx in tscv.split(df):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        
        # 训练模型
        models = train_models(train, features)

        # 预测并评估
        for model_name, model in models.items():
            metrics = evaluate_model(model, test, test['Total_Medals'], features, model_name)
            results.append(metrics)
    # 输出结果
    print(pd.DataFrame(results).groupby('model').mean())

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
    
# def main():
#     # 数据管道
#     df = load_and_preprocess()
#     df = add_host_features(df)
    
#     # 过滤有效数据：至少有过奖牌且项目存在的记录
#     df = df[
#         (df['Total_Medals'] >= CONFIG['min_medal_threshold']) &
#         (df['Sport_Exists'] == 1)
#     ]
    
#     # 生成滞后特征
#     df, features = create_lag_features(df, CONFIG['n_years'])
#     features += ['is_host']
    
#     # 时间序列分割
#     tscv = TimeSeriesSplit(n_splits=3)
#     results = []
    
#     for train_idx, test_idx in tscv.split(df):
#         train = df.iloc[train_idx]
#         test = df.iloc[test_idx]
        
#         # 训练模型
#         model = train_model(train, train['Total_Medals'], features)
        
#         # 评估
#         metrics = evaluate_model(model, test, test['Total_Medals'], features)
#         results.append(metrics)
    
#     # 输出平均性能
#     avg_metrics = pd.DataFrame(results).mean()
#     print(f"模型平均性能 ({CONFIG['model_type']}):\n{avg_metrics}")
    
#     # 全量训练与预测
#     final_model = train_model(df, df['Total_Medals'], features)
    
#     # 生成2028年预测
#     future = df[df.Year == 2024].copy()
#     future['Year'] = 2028
    
#     # 更新滞后特征：将2024年数据作为lag1
#     for lag in range(1, CONFIG['n_years']+1):
#         future[f'Gold_lag{lag}'] = future.groupby(['Sport', 'NOC'])['Gold'].shift(lag-1).fillna(0)
#         future[f'Silver_lag{lag}'] = future.groupby(['Sport', 'NOC'])['Silver'].shift(lag-1).fillna(0)
#         future[f'Bronze_lag{lag}'] = future.groupby(['Sport', 'NOC'])['Bronze'].shift(lag-1).fillna(0)
#         future[f'Total_lag{lag}'] = future.groupby(['Sport', 'NOC'])['Total_Medals'].shift(lag-1).fillna(0)
    
#     # 添加东道主标记
#     future['is_host'] = (future['NOC'] == 'United States').astype(int)
    
#     # 确保所有特征列都有值
#     future[features] = future[features].fillna(0)
    
#     # 预测前检查数据完整性
#     print(f"预测数据形状: {future[features].shape}")
#     print(f"特征列: {features}")
    
#     # 预测
#     future['pred_medals'] = final_model.predict(future[features])
    
#     # 保存结果
#     future[['Sport', 'NOC', 'Year', 'pred_medals']].to_csv(
#         f"{CONFIG['output_dir']}sport_level_predictions.csv",
#         index=False
#     )

# if __name__ == "__main__":
#     # 切换到当前目录
#     os.chdir(os.path.dirname(os.path.abspath(__file__)))
#     main()