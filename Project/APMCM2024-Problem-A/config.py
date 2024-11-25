# config.py

# 色偏特征阈值
COLOR_CAST_THRESHOLDS = {
    'delta_gray': 5,
    'delta_lab': 0.5,
    'delta_hsv': 0.2
}

# 低光照特征阈值
LOW_LIGHT_THRESHOLDS = {
    'mean_brightness': 100,
    'dark_ratio': 0.5,
    'skewness': -1
}

# 模糊特征阈值
BLUR_THRESHOLDS = {
    'laplacian_var': 125,
    'high_freq_ratio': 0.4
}

# 特征权重
COLOR_WEIGHTS = {
    'delta_gray': 0.8,
    'delta_lab': 0.1,
    'delta_hsv': 0.1
}

# 低光照权重
LOW_LIGHT_WEIGHTS = {
    'mean_brightness': 0.6,
    'dark_ratio': 0.2,
    'skewness': 0.2
}

# 模糊特征权重
BLUR_WEIGHTS = {
    'laplacian_var': 0.75,
    'high_freq_ratio': 0.25
}

# 归一化参数
SCORE_THRESHOLD = 0.4        # 最终得分阈值
CONFIDENCE_LEVELS = {
    'HIGH': 0.7,   # 降低高置信度阈值
    'MEDIUM': 0.5, # 降低中等置信度阈值
    'LOW': 0.3     # 降低低置信度阈值
}

# 特征方向：'positive'表示数值越大退化越严重，'negative'表示数值越小退化越严重
FEATURE_DIRECTIONS = {
    # 色偏特征（值越大，色偏越严重）
    'delta_gray': 'positive',
    'delta_lab': 'positive',
    'delta_hsv': 'positive',
    # 低光照特征
    'mean_brightness': 'negative',  # 均值越小，光照越差
    'dark_ratio': 'positive',       # 暗像素比例越大，光照越差
    'skewness': 'negative',         # 偏度越小（负），光照越差
    # 模糊特征
    'laplacian_var': 'negative',    # 拉普拉斯方差越小，越模糊
    'high_freq_ratio': 'positive',  # 高频能量比越大，图像越清晰
}

# 可视化参数
HISTOGRAM_BINS = 256  # 直方图的bins数量

# 任务3的增强参数

# 白平衡参数
WHITE_BALANCE_METHOD = 'gray_world'  # 白平衡方法，可选'gray_world'或'perfect_reflector'
WHITE_BALANCE_CLIP = 0.8  # 限制白平衡增益的最大值，范围建议：0.6-0.9
WHITE_BALANCE_WARMTH = 0.9  # 暖色调补偿因子(降低暖色调)，范围建议：0.8-1.0
WHITE_BALANCE_WEIGHT = 0.7  # 原始图像和校正图像的混合权重，范围建议：0.5-0.8

# 亮度增强参数
GAMMA_CORRECTION_VALUE = 1.5  # 伽马校正的gamma值，范围建议：1.2-2.0
DCP_WINDOW_SIZE = 9  # 暗通道先验的窗口大小，范围建议：7-31
ATM_LIGHT_THRESHOLD = 0.08  # 大气光估计的阈值，范围建议：0.05-0.2

# 去模糊参数
DEBLUR_KERNEL_SIZE = 3  # 去模糊的卷积核大小，范围建议：3-5
DEBLUR_SIGMA = 0.8  # 添加高斯核标准差参数，范围建议：0.8-1.5
SHARPENING_STRENGTH = 1.8  # 添加锐化强度参数，范围建议：1.2-2.0

# 任务4的深度学习模型参数
EPOCHS = 100            # 训练轮数，根据实验调整
BATCH_SIZE = 16         # 范围建议：8-32
LEARNING_RATE = 1e-4    # 范围建议：1e-5 - 1e-3

# 学习率调度参数
LR_SCHEDULER_STEP = 10
LR_GAMMA = 0.5
WEIGHT_DECAY = 1e-4  # 权重衰减，范围建议：1e-5 - 1e-3
MOMENTUM = 0.9  # 添加动量参数

# 早停参数
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_DELTA = 1e-4
MODEL_SAVE_PATH = 'models/best_model.pth' # 模型保存路径
PERCEPTUAL_LOSS_WEIGHT = 0.1 # 感知损失的权重
MODEL_TYPE = 'UNet'  # 模型类型配置: UNet 或 SimpleCNN

# 可视化参数
FIGURE_SIZE = (16, 8)  # 图像展示的尺寸
