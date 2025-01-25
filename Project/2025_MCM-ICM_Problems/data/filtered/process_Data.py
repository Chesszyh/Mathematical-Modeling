"""
奥运奖牌预测模型 - 问题C第一小问增强版
新增随机森林实现与模型组合逻辑
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
import statsmodels.formula.api as smf
import xgboost as xgb
from sklearn.utils import resample
from sklearn.preprocessing import OneHotEncoder

# 全局配置增强
CONFIG = {
    "data_path": {
        "medals": "summerOly_medal_counts.csv",
        "hosts": "summerOly_hosts.csv",
        "programs": "summerOly_programs.csv",
        "athletes": "summerOly_athletes.csv"
    },
    "output_dir": "./results/",
    "current_year": 2024,
    "predict_year": 2028,
    "n_bootstrap": 100,
    "model_params": {
        "xgboost": {
            "max_depth": 4,
            "eta": 0.1,
            "objective": "reg:squarederror"
        },
        "random_forest": {
            "n_estimators": 200,
            "max_depth": 5,
            "min_samples_split": 5
        }
    }
}

# 切换到当前目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def load_data(config):
    """数据加载与初步处理"""
    # 读取原始数据
    medals = pd.read_csv(config["data_path"]["medals"])
    hosts = pd.read_csv(config["data_path"]["hosts"])
    programs = pd.read_csv(config["data_path"]["programs"]) 
    athletes = pd.read_csv(config["data_path"]["athletes"])

    # 数据清洗
    def preprocess_hosts(df):
        """提取东道主国家并生成虚拟变量"""
        df['host_country'] = df['Host'].str.split(',').str[-1].str.strip()
        return df[['Year', 'host_country']]
    
    # 合并核心数据
    df = medals.merge(
        preprocess_hosts(hosts), 
        on='Year', 
        how='left'
    )
    return df, programs, athletes

def feature_engineering(df, programs, athletes):
    """特征工程管道"""
    # 滞后特征生成
    def add_lag_features(df, lags=[1,2,3]):
        """生成过去n届的奖牌数特征"""
        for lag in lags:
            df = df.sort_values(['NOC', 'Year'])
            df[f'Gold_lag{lag}'] = df.groupby('NOC')['Gold'].shift(lag)
            df[f'Total_lag{lag}'] = df.groupby('NOC')['Total'].shift(lag)
        return df
    
    # 东道主特征
    df['is_host'] = (df['NOC'] == df['host_country']).astype(int)
    
    # 项目参与度特征（需根据programs数据计算）
    # ...（此处省略具体实现，需计算各国每年参与项目数）
    
    # 运动员特征（聚合到国家-年层级）
    athlete_features = athletes.groupby(['NOC', 'Year', 'Sport']).agg(
        athlete_count=('Name', 'nunique'),
        has_medal=('Medal', lambda x: (x != 'No Medal').sum())
    ).reset_index()
    # ...（进一步聚合到国家-年层级）
    
    # 合并所有特征
    df = add_lag_features(df)
    return df



class MedalPredictor:
    """奖牌预测模型容器（支持混合效应/XGBoost/随机森林）"""
    
    def __init__(self, model_type='ensemble'):
        self.models = {}
        self.model_type = model_type
        self.encoders = {}  # 类别型特征编码器
        
    def _encode_features(self, df, is_training=True):
        """类别型特征编码（扩展性处理）"""
        # 示例：对离散特征进行OneHot编码
        if 'region' in df.columns:  # 假设有地区特征
            if is_training:
                self.encoders['region'] = OneHotEncoder()
                self.encoders['region'].fit(df[['region']])
            df_encoded = self.encoders['region'].transform(df[['region']])
            df = pd.concat([df, df_encoded], axis=1)
        return df
    
    def train(self, df, target='Gold'):
        """训练模型（支持多模型类型）"""
        train_df = df[df.Year < CONFIG['current_year']].dropna()
        features = ['Gold_lag1', 'Total_lag1', 'is_host', 'project_coverage']
        
        # 特征编码
        train_df = self._encode_features(train_df, is_training=True)
        
        if self.model_type == 'mixed_effect':
            formula = f"{target} ~ Gold_lag1 + Total_lag1 + is_host + project_coverage"
            self.models[target] = smf.mixedlm(
                formula, 
                train_df, 
                groups=train_df["NOC"]
            ).fit()
            
        elif self.model_type == 'xgboost':
            dtrain = xgb.DMatrix(
                train_df[features], 
                label=train_df[target]
            )
            self.models[target] = xgb.train(
                CONFIG['model_params']['xgboost'], 
                dtrain
            )
            
        elif self.model_type == 'random_forest':
            self.models[target] = RandomForestRegressor(
                **CONFIG['model_params']['random_forest'],
                n_jobs=-1
            )
            self.models[target].fit(
                train_df[features], 
                train_df[target]
            )
            
        elif self.model_type == 'ensemble':
            # 混合模型：用混合效应结果作为XGBoost特征
            me_model = smf.mixedlm(
                f"{target} ~ Gold_lag1 + is_host", 
                train_df, 
                groups=train_df["NOC"]
            ).fit()
            train_df['me_residual'] = train_df[target] - me_model.predict(train_df)
            
            # XGBoost训练残差
            dtrain = xgb.DMatrix(
                train_df[features + ['me_residual']], 
                label=train_df[target]
            )
            self.models[target] = {
                'me': me_model,
                'xgb': xgb.train(CONFIG['model_params']['xgboost'], dtrain)
            }
    
    def predict(self, df, target='Gold'):
        """生成预测及区间（多模型支持）"""
        df = self._encode_features(df, is_training=False)
        features = ['Gold_lag1', 'Total_lag1', 'is_host', 'project_coverage']
        
        if self.model_type == 'mixed_effect':
            pred = self.models[target].predict(df)
        elif self.model_type == 'xgboost':
            dtest = xgb.DMatrix(df[features])
            pred = self.models[target].predict(dtest)
        elif self.model_type == 'random_forest':
            pred = self.models[target].predict(df[features])
        elif self.model_type == 'ensemble':
            me_pred = self.models[target]['me'].predict(df)
            dtest = xgb.DMatrix(df[features + ['me_residual']])
            xgb_pred = self.models[target]['xgb'].predict(dtest)
            pred = 0.7 * me_pred + 0.3 * xgb_pred  # 加权集成
        
        # Bootstrap不确定性估计
        boot_preds = []
        for _ in range(CONFIG['n_bootstrap']):
            sample = resample(df)
            temp_model = self.__class__(model_type=self.model_type)
            temp_model.train(sample, target)
            boot_preds.append(temp_model.predict(sample, target)[0])
        
        ci_low = np.percentile(boot_preds, 5, axis=0)
        ci_high = np.percentile(boot_preds, 95, axis=0)
        
        return pred, ci_low, ci_high

    def evaluate(self, df):
        """模型评估"""
        test_df = df[df.Year == CONFIG['current_year']]
        metrics = {}
        
        for target in ['Gold', 'Total']:
            y_true = test_df[target]
            y_pred, _, _ = self.predict(test_df, target)
            
            metrics[target] = {
                'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                'MAE': mean_absolute_error(y_true, y_pred)
            }
        return metrics

def predict_2028(df, model):
    """生成2028年预测"""
    # 构建预测输入
    future = df[df.Year == CONFIG['current_year']].copy()
    future['Year'] = CONFIG['predict_year']
    future['is_host'] = (future['NOC'] == 'United States').astype(int)
    
    # 生成预测
    gold_pred, gold_low, gold_high = model.predict(future, 'Gold')
    total_pred, total_low, total_high = model.predict(future, 'Total')
    
    # 结果包装
    results = pd.DataFrame({
        'NOC': future['NOC'],
        'Gold_pred': gold_pred,
        'Gold_CI': list(zip(gold_low, gold_high)),
        'Total_pred': total_pred,
        'Total_CI': list(zip(total_low, total_high))
    })
    
    # 识别表现变化国家
    current = df[df.Year == CONFIG['current_year']]
    merged = results.merge(
        current[['NOC', 'Gold', 'Total']],
        on='NOC',
        suffixes=('_2028', '_2024')
    )
    merged['Gold_change'] = merged['Gold_pred'] - merged['Gold']
    merged['Total_change'] = merged['Total_pred'] - merged['Total']
    
    return merged

def main():
    # 数据管道
    df, programs, athletes = load_data(CONFIG)
    df = feature_engineering(df, programs, athletes)
    
    # 模型训练
    predictor = MedalPredictor(model_type='mixed_effect')
    predictor.train(df, 'Gold')
    predictor.train(df, 'Total')
    
    # 模型评估
    metrics = predictor.evaluate(df)
    print(f"Model Performance:\n{metrics}")
    
    # 生成预测
    predictions = predict_2028(df, predictor)
    
    # 保存结果
    predictions.to_csv(f"{CONFIG['output_dir']}q1_predictions.csv", index=False)
    
    # 输出关键结论
    top_growth = predictions.nlargest(5, 'Gold_change')['NOC'].tolist()
    top_decline = predictions.nsmallest(5, 'Gold_change')['NOC'].tolist()
    
    print(f"\n2028预测结论：")
    print(f"表现提升国家：{', '.join(top_growth)}")
    print(f"表现下滑国家：{', '.join(top_decline)}")

if __name__ == "__main__":
    main()