"""
项目-国家维度奖牌预测模型
按运动项目和国家，基于历史奖牌数据预测未来奖牌数
算法参考：https://github.com/TheAlgorithms/Python/
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
        "n_years": 3,  # 使用前n届数据作为特征
    },
    "output_dir": "output/",
    "min_medal_threshold": 2  # 最小奖牌数阈值
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
            print("矩阵求逆失败，使用简单移动平均")
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
    """数据加载与预处理：项目-国家-年"""
    # 1. 加载运动员数据
    athletes = read_data(CONFIG["data_path"]["athletes"])
    print("\n=== 原始运动员数据 ===")
    print(f"Shape: {athletes.shape}")
    print(f"Columns: {athletes.columns.tolist()}")
    print("\n前5行示例:")
    print(athletes.head())
    
    athletes['has_medal'] = athletes['Medal'] != 'No medal'
    
    # 2. 统计奖牌
    medal_detail = athletes.groupby(['Sport', 'NOC', 'Year']).agg(
        Gold=('Medal', lambda x: (x == 'Gold').sum()),
        Silver=('Medal', lambda x: (x == 'Silver').sum()),
        Bronze=('Medal', lambda x: (x == 'Bronze').sum()),
        Total_Medals=('has_medal', 'sum')
    ).reset_index()
    
    print("\n=== 奖牌统计结果 ===")
    print(f"Shape: {medal_detail.shape}")
    print("\n前5行示例:")
    print(medal_detail.head())
    
    # 3. 加载项目数据
    programs = read_data(CONFIG["data_path"]["programs"])
    print("\n=== 项目数据 ===")
    print(f"Shape: {programs.shape}")
    print(f"Columns: {programs.columns.tolist()}")
    
    # 4. 处理年份列
    year_columns = [col for col in programs.columns if col.isdigit() and 1896 <= int(col) <= 2024]
    print(f"\n有效年份列: {len(year_columns)}个")
    print(f"年份范围: {min(year_columns)}-{max(year_columns)}")
    
    # 添加调试信息
    print("\n=== Melt前的数据 ===")
    print(programs.head())

    programs = programs.melt(
        id_vars=['Sport', 'Discipline', 'Code'],  # 保持不变的标识列
        value_vars=year_columns,                  # 需要转换的年份列
        var_name='Year',                         # 新的年份列名
        value_name='Exists'                      # 新的存在性列名
    )

    print("\n=== Melt后的数据 ===")
    print(programs.head())

    # 数据类型转换和清理
    programs['Exists'] = pd.to_numeric(programs['Exists'], errors='coerce').fillna(0).astype(int)
    programs['Year'] = programs['Year'].astype(int)

    print("\n=== 类型转换后的数据 ===")
    print(programs.dtypes)

    # 筛选实际举办的项目
    programs = programs[programs['Exists'] == 1]

    print("\n=== 筛选后的数据 ===")
    print(f"总行数: {len(programs)}")
    print(programs.head())
    
    print("\n=== 处理后的项目数据 ===")
    print(f"Shape: {programs.shape}")
    print("\n前5行示例:")
    print(programs.head())
    
    # 6. 合并数据
    medal_detail = medal_detail.merge(
        programs[['Sport', 'Year']].drop_duplicates(),
        on=['Sport', 'Year'],
        how='left',
        indicator=True
    )
    
    medal_detail['Sport_Exists'] = (medal_detail['_merge'] == 'both').astype(int)
    medal_detail.drop('_merge', axis=1, inplace=True)
    
    print("\n=== 最终数据 ===")
    print(f"Shape: {medal_detail.shape}")
    print("\n列名:", medal_detail.columns.tolist())
    print("\n前5行示例:")
    print(medal_detail.head())
    
    # 7. 数据质量检查
    print("\n=== 数据质量检查 ===")
    print("缺失值统计:")
    print(medal_detail.isnull().sum())
    print("\n数值统计:")
    print(medal_detail.describe())
    
    return medal_detail

def create_features(df, n_years=3):
    """增强特征工程"""
    features = []
    df = df.sort_values(['Sport', 'NOC', 'Year'])
    
    # 1. 基础滞后特征
    for lag in range(1, n_years+1):
        for medal in ['Gold', 'Silver', 'Bronze', 'Total_Medals']:
            col = f'{medal}_lag{lag}'
            df[col] = df.groupby(['Sport', 'NOC'])[medal].shift(lag)
            features.append(col)
    
    # 2. 趋势特征
    df['medal_trend'] = df.groupby(['Sport', 'NOC'])['Total_Medals'].transform(
        lambda x: x.ewm(alpha=0.5).mean()
    )
    features.append('medal_trend')
    
    # 3. 比赛项目重要性
    df['sport_importance'] = df.groupby('Sport')['Total_Medals'].transform('sum')
    features.append('sport_importance')
    
    # 4. 国家实力
    df['country_strength'] = df.groupby('NOC')['Total_Medals'].transform('mean')
    features.append('country_strength')
    
    return df, features

def train_models(df, target, features):
    """改进的模型训练"""
    models = {
        'rf': RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            min_samples_split=5,
            random_state=42
        ),
        'xgb': xgb.XGBRegressor(
            max_depth=4,
            learning_rate=0.1,
            n_estimators=100,
            objective='reg:squarederror'
        )
    }
    
    # 训练模型
    trained_models = {}
    for name, model in models.items():
        model.fit(df[features], df[target])
        trained_models[name] = model
        
        # 特征重要性
        if name == 'rf':
            importances = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            print(f"\n{name.upper()} Feature Importances:")
            print(importances.head())
    
    return trained_models

def evaluate_predictions(y_true, predictions, model_names):
    """评估预测结果"""
    metrics = []
    for name in model_names:
        y_pred = predictions[name]
        metrics.append({
            'model': name,
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100
        })
    return pd.DataFrame(metrics)

def main():
    # 1. 数据加载与预处理
    df = load_and_preprocess()
    print(f"Initial shape: {df.shape}")
    
    # 2. 特征工程
    df, features = create_features(df)
    df = df.dropna()
    print(f"After feature engineering: {df.shape}")
    
    # 3. 时间序列分割
    train = df[df['Year'] <= 2016]
    test = df[df['Year'] == 2020]
    
    # 4. 分别训练金牌和总奖牌模型
    targets = ['Gold', 'Total_Medals']
    all_metrics = []
    
    for target in targets:
        print(f"\nTraining models for {target}")
        models = train_models(train, target, features)
        
        # 预测与评估
        predictions = {}
        for name, model in models.items():
            predictions[name] = model.predict(test[features])
        
        metrics = evaluate_predictions(test[target], predictions, models.keys())
        metrics['target'] = target
        all_metrics.append(metrics)
        
        print(f"\nResults for {target}:")
        print(metrics)
    
    # 5. 输出综合结果
    final_metrics = pd.concat(all_metrics)
    print("\nFinal Results:")
    print(final_metrics.groupby(['target', 'model']).mean())

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()