# utils.py
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from config import *
from model import *
from skimage import img_as_float
from skimage.metrics import structural_similarity as ssim

# 图像处理
def read_image(image_path):
    """读取图像并确保是BGR格式
    参数:
        image_path: 图像路径
    返回:
        image: BGR格式的图像，如果读取失败返回None
    """
    try:
        # 读取图像
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # 强制读取为BGR
        if image is None:
            print(f"无法读取图像: {image_path}")
            return None
            
        # 检查通道数
        if len(image.shape) != 3 or image.shape[2] != 3:
            print(f"图像 {image_path} 不是3通道图像")
            # 如果是单通道图像，转换为3通道
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                print(f"已将单通道图像转换为3通道: {image_path}")
            
        return image
    except Exception as e:
        print(f"读取图像时出错 {image_path}: {str(e)}")
        return None

def calculate_mean_std(image):
    """计算图像各通道的均值和标准差
    参数:
        image: 输入的图像
    返回:
        mean: 各通道均值
        std: 各通道标准差
    """
    mean, std = cv2.meanStdDev(image)
    return mean.flatten(), std.flatten()

def convert_to_gray(image):
    """将图像转换为灰度图
    参数:
        image: 输入的彩色图像
    返回:
        gray: 转换后的灰度图
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

# 图像计算
def analyze_color_cast(image):
    """色偏检测"""
    # RGB空间分析
    mean, std = calculate_mean_std(image)
    delta_gray = (abs(mean[2] - mean[1]) + 
                 abs(mean[1] - mean[0]) + 
                 abs(mean[0] - mean[2])) / mean.mean()  # 归一化
    
    # LAB空间分析
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    delta_lab = np.sqrt(np.mean(a - 128)**2 + np.mean(b - 128)**2) / 128.0
    
    # HSV空间分析
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    delta_hsv = np.std(hsv[:,:,1]) / 255.0
    
    return {
        'delta_gray': delta_gray,
        'delta_lab': delta_lab,
        'delta_hsv': delta_hsv
    }

def analyze_low_light(image):
    """低光照检测模型"""
    gray = convert_to_gray(image)

    # 计算亮度均值和标准差
    mu_I = np.mean(gray)
    sigma_I = np.std(gray)

    # 计算偏度
    if sigma_I != 0:
        skewness = np.mean(((gray - mu_I) / sigma_I) ** 3)
    else:
        skewness = 0

    # 计算暗像素比例
    dark_ratio = np.sum(gray < LOW_LIGHT_THRESHOLDS['mean_brightness']) / gray.size

    return {
        'mean_brightness': mu_I,
        'dark_ratio': dark_ratio,
        'skewness': skewness
    }
    
def analyze_blur(image):
    """基于模糊检测模型"""
    gray = convert_to_gray(image)

    # 拉普拉斯分析
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sigma_L = np.var(laplacian)  # 拉普拉斯方差

    # 频域分析
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    # 计算高频能量比
    total_energy = np.sum(magnitude)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    high_freq_energy = total_energy - np.sum(magnitude[crow - 10:crow + 10, ccol - 10:ccol + 10])
    high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0

    return {
        'laplacian_var': sigma_L,
        'high_freq_ratio': high_freq_ratio
    }

def calculate_histogram(image, bins=256):
    """计算图像的直方图
    参数:
        image: 输入的图像
        bins: 直方图的bins数量
    返回:
        hist: 计算得到的直方图
    """
    hist = cv2.calcHist([image], [0], None, [bins], [0, 256])
    return hist

def calculate_skewness(image):
    """计算灰度图的偏度
    参数:
        image: 输入的灰度图像
    返回:
        skewness: 计算得到的偏度值
    """
    pixels = image.flatten()
    mean = np.mean(pixels)
    std = np.std(pixels)
    skewness = np.mean(((pixels - mean) / std) ** 3)
    return skewness

def calculate_laplacian_variance(image):
    """计算拉普拉斯方差用于模糊检测
    参数:
        image: 输入的彩色图像
    返回:
        variance: 拉普拉斯方差值
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    variance = lap.var()
    return variance

def save_image(image, image_path):
    """保存图像
    参数:
        image: 要保存的图像
        image_path: 保存路径
    """
    cv2.imwrite(image_path, image)

# 图像评价
def calculate_psnr(img1, img2):
    """计算PSNR
    参数:
        img1: 原始图像
        img2: 目标图像
    返回:
        psnr_value: 计算得到的PSNR值
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    psnr_value = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return psnr_value

def calculate_ssim(img1, img2):
    """计算SSIM
    参数:
        img1: 原始图像
        img2: 目标图像
    返回:
        ssim_value: 计算得到的SSIM值
    """
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ssim_value = ssim(img1_gray, img2_gray, data_range=img2_gray.max() - img2_gray.min())
    return ssim_value

def calculate_uciqe(image):
    """计算UCIQE指标
    参数:
        image: 输入的图像
    返回:
        uciqe_value: 计算得到的UCIQE值
    """
    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    chroma = np.sqrt(a ** 2 + b ** 2)
    saturation = chroma / (l + 1e-6)
    uciqe_value = np.std(saturation) * 0.4680 + np.mean(saturation) * 0.2745 + np.std(l) * 0.2576
    return uciqe_value

def calculate_uiqm(image):
    """计算UIQM指标
    参数:
        image: 输入的图像
    返回:
        uiqm_value: 计算得到的UIQM值
    """
    img_float = img_as_float(image)
    uiqm_value = 0.0282 * calculate_uicm(img_float) + 0.2953 * calculate_uism(img_float) + 3.5753 * calculate_uiconm(img_float)
    return uiqm_value

def calculate_uicm(image):
    """计算UICM，用于UIQM指标"""
    r = image[:, :, 2].flatten()
    g = image[:, :, 1].flatten()
    b = image[:, :, 0].flatten()
    rg = r - g
    yb = 0.5 * (r + g) - b
    uicm_value = np.sqrt(np.var(rg) + np.var(yb))
    return uicm_value

def calculate_uism(image):
    """计算UISM，用于UIQM指标"""
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    sobel_h = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobel_v = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    gradient_magnitude = np.sqrt(sobel_h ** 2 + sobel_v ** 2)
    uism_value = np.mean(gradient_magnitude)
    return uism_value

def calculate_uiconm(image):
    """计算UIConM，用于UIQM指标"""
    hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1] / 255.0
    uiconm_value = np.mean(saturation)
    return uiconm_value

# 加载分类结果
def load_classification_results(csv_path):
    """加载任务1的分类结果
    参数:
        csv_path: 分类结果的CSV文件路径
    返回:
        classification_dict: 字典，键为图像名，值为分类列表
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    classification_dict = {}
    for index, row in df.iterrows():
        image_name = row['ImageName']
        classifications = []
        if row['ColorCast'] == 1:
            classifications.append('Color Cast')
        if row['LowIllumination'] == 1:
            classifications.append('Low Illumination')
        if row['Blur'] == 1:
            classifications.append('Blur')
        classification_dict[image_name] = classifications
    return classification_dict

# 模型处理
def select_model(model_type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_type == 'UNet':
        model = UNetEnhancer().to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model

def save_model(model, path):
    torch.save({'model_state_dict': model.state_dict()}, path)

def load_model(model, path):
    checkpoint = torch.load(path,weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def prune_model(model, amount=0.2):
    # 针对卷积层进行剪枝
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            # 可以选择是否去除剪枝的参数以减少模型大小
            # prune.remove(module, 'weight')
    return model