# enhancement.py
import cv2
import torch
import numpy as np
from config import *

def white_balance(image):
    """白平衡校正（针对偏色）
    参数:
        image: 输入的图像
    返回:
        result: 处理后的图像
    """
    result = image.copy().astype(np.float32) # 转换为浮点数
    if WHITE_BALANCE_METHOD == 'gray_world': # 灰度世界假设
        # 计算各通道均值
        mean_b = np.mean(result[:, :, 0])
        mean_g = np.mean(result[:, :, 1])
        mean_r = np.mean(result[:, :, 2])
        mean_gray = (mean_b + mean_g + mean_r) / 3
        # 计算增益并限制范围
        kb = min(mean_gray / mean_b, WHITE_BALANCE_CLIP)
        kg = min(mean_gray / mean_g, WHITE_BALANCE_CLIP)
        kr = min(mean_gray / mean_r, WHITE_BALANCE_CLIP) * WHITE_BALANCE_WARMTH
        
        # 应用增益
        result[:, :, 0] = np.clip(result[:, :, 0] * kb, 0, 255)
        result[:, :, 1] = np.clip(result[:, :, 1] * kg, 0, 255)
        result[:, :, 2] = np.clip(result[:, :, 2] * kr, 0, 255)
        
        # 与原图加权融合
        result = cv2.addWeighted(result, WHITE_BALANCE_WEIGHT, 
                               image.astype(np.float32), 
                               1 - WHITE_BALANCE_WEIGHT, 0)
        
    elif WHITE_BALANCE_METHOD == 'perfect_reflector': # 理想反射体，暂未优化
        max_b = np.max(result[:, :, 0])
        max_g = np.max(result[:, :, 1])
        max_r = np.max(result[:, :, 2])
        max_value = max(max_b, max_g, max_r)
        result[:, :, 0] = np.clip(result[:, :, 0] * (max_value / max_b), 0, 255)
        result[:, :, 1] = np.clip(result[:, :, 1] * (max_value / max_g), 0, 255)
        result[:, :, 2] = np.clip(result[:, :, 2] * (max_value / max_r), 0, 255)
    return result.astype(np.uint8)

def gamma_correction(image):
    """伽马校正（针对弱光）
    参数:
        image: 输入的图像
    返回:
        result: 处理后的图像
    """
    gamma = GAMMA_CORRECTION_VALUE
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
    result = cv2.LUT(image, table)
    return result

def deblur(image):
    """去模糊处理
    参数:
        image: 输入的图像
    返回:
        result: 处理后的图像
    """
    # 1. 维纳滤波
    kernel_size = DEBLUR_KERNEL_SIZE
    psf = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    deconv = cv2.filter2D(image, -1, psf)
    
    # 2. Unsharp Masking
    # 高斯模糊
    gaussian = cv2.GaussianBlur(deconv, 
                               (DEBLUR_KERNEL_SIZE, DEBLUR_KERNEL_SIZE), 
                               DEBLUR_SIGMA)
    # 计算高频分量
    high_freq = cv2.addWeighted(deconv, 1.0, gaussian, -1.0, 0)
    
    # 3. 结果融合
    result = cv2.addWeighted(deconv, 1.0,
                            high_freq, SHARPENING_STRENGTH, 0)
    
    return np.clip(result, 0, 255).astype(np.uint8)

def get_dark_channel(image, window_size):
    """计算暗通道
    参数:
        image: 输入的归一化图像
        window_size: 窗口大小
    返回:
        dark_channel: 暗通道图像
    """
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel

def estimate_atmospheric_light(image, dark_channel):
    """估计大气光
    参数:
        image: 输入的归一化图像
        dark_channel: 暗通道图像
    返回:
        atmospheric_light: 大气光值
    """
    flat_image = image.reshape(-1, 3)
    flat_dark = dark_channel.ravel()
    num_pixels = len(flat_dark)
    num_search_pixels = int(max(num_pixels * ATM_LIGHT_THRESHOLD, 1))
    indices = np.argsort(flat_dark)[-num_search_pixels:]
    atmospheric_light = np.mean(flat_image[indices], axis=0)
    return atmospheric_light

def estimate_transmission(image, atmospheric_light, window_size):
    """估计透射率
    参数:
        image: 输入的归一化图像
        atmospheric_light: 大气光值
        window_size: 窗口大小
    返回:
        transmission: 透射率图像
    """
    omega = 0.95  # 衰减系数
    norm_image = image / atmospheric_light
    dark_channel = get_dark_channel(norm_image, window_size)
    transmission = 1 - omega * dark_channel
    return transmission

def recover_scene_radiance(image, transmission, atmospheric_light):
    """恢复场景辐亮度
    参数:
        image: 输入的归一化图像
        transmission: 透射率图像
        atmospheric_light: 大气光值
    返回:
        result: 恢复的图像
    """
    t0 = 0.1  # 最小透射率
    transmission = np.clip(transmission, t0, 1)
    result = (image - atmospheric_light) / transmission[:, :, np.newaxis] + atmospheric_light
    return result

def enhance_with_basic_methods(image, classifications):
    """使用基础方法增强图像"""
    # 调用三种方法的顺序，也可能影响结果
    enhanced_task3 = image.copy()
    if 'Color Cast' in classifications:
        enhanced_task3 = white_balance(enhanced_task3)
    if 'Low Illumination' in classifications:
        # 1. gamma校正
        enhanced_task3 = gamma_correction(enhanced_task3)
        # 2. 暗通道先验
        normalized_image = enhanced_task3.astype(np.float32) / 255.0
        dark_channel = get_dark_channel(normalized_image, DCP_WINDOW_SIZE)
        atmospheric_light = estimate_atmospheric_light(normalized_image, dark_channel)
        transmission = estimate_transmission(normalized_image, atmospheric_light, DCP_WINDOW_SIZE)
        enhanced_task3 = recover_scene_radiance(normalized_image, transmission, atmospheric_light)
        enhanced_task3 = (np.clip(enhanced_task3, 0, 1) * 255).astype(np.uint8)
    if 'Blur' in classifications:
        enhanced_task3 = deblur(enhanced_task3)
    return enhanced_task3

def enhance_with_model(image, model):
    """使用深度学习模型增强图像
    参数:
        image: 输入的图像
        model: 训练好的模型
    返回:
        result: 增强后的图像
    """
    model.eval()
    image_tensor = torch.from_numpy(image.transpose((2, 0, 1))).unsqueeze(0).float() / 255.0
    image_tensor = image_tensor.to('cuda')
    with torch.no_grad():
        output_tensor = model(image_tensor)
    output_image = output_tensor.squeeze(0).cpu().numpy().transpose((1, 2, 0))
    output_image = np.clip(output_image * 255, 0, 255).astype(np.uint8)
    return output_image