# main_category.py is a solution file for the main task of the APMCM 2024 Problem A.
# It contains the main logic for the image classification task.

import os
import cv2
import random
import numpy as np
import pandas as pd
from utils import *
from visualization import *
from config import *

# -----------------------Settings-----------------------
# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
CATEGORY_MODE = "test" # "train" or "test"
MAX_PICTURE_NUM = 401 if CATEGORY_MODE == "train" else 13
RANDOM_CHOICE = 5 if CATEGORY_MODE == "train" else 2

# -----------------------Settings-----------------------

def calculate_weighted_score(features, thresholds, weights, feature_directions):
    """计算加权得分"""
    scores = []
    for key, value in features.items():
        if key in thresholds and key in weights:
            direction = feature_directions.get(key, 'positive')
            if direction == 'positive':
                normalized_score = value / thresholds[key]
            elif direction == 'negative':
                normalized_score = 1 - (value / thresholds[key])
            else:
                normalized_score = value / thresholds[key]
            normalized_score = min(max(normalized_score, 0), 1)  # 限制在0到1之间
            scores.append(normalized_score * weights[key])
    total_score = sum(scores) / sum(weights.values()) if weights else 0
    return total_score

def classify_image(color_features, light_features, blur_features):
    """分类图像并返回置信度"""
    # 计算色偏得分
    color_score = calculate_weighted_score(
        color_features, 
        COLOR_CAST_THRESHOLDS,
        COLOR_WEIGHTS,
        FEATURE_DIRECTIONS
    )
    
    # 计算低光得分
    light_score = calculate_weighted_score(
        light_features,
        LOW_LIGHT_THRESHOLDS,
        LOW_LIGHT_WEIGHTS,
        FEATURE_DIRECTIONS
    )
    
    # 计算模糊得分
    blur_score = calculate_weighted_score(
        blur_features,
        BLUR_THRESHOLDS,
        BLUR_WEIGHTS,
        FEATURE_DIRECTIONS
    )
    
    # 归一化处理
    color_score = min(max(color_score, 0), 1)
    light_score = min(max(light_score, 0), 1)
    blur_score = min(max(blur_score, 0), 1)
    
    # 确定置信度级别
    results = []
    if color_score > SCORE_THRESHOLD:
        confidence = 'HIGH' if color_score > CONFIDENCE_LEVELS['HIGH'] else \
                    'MEDIUM' if color_score > CONFIDENCE_LEVELS['MEDIUM'] else 'LOW'
        results.append(('Color Cast', color_score, confidence))
        
    if light_score > SCORE_THRESHOLD:
        confidence = 'HIGH' if light_score > CONFIDENCE_LEVELS['HIGH'] else \
                    'MEDIUM' if light_score > CONFIDENCE_LEVELS['MEDIUM'] else 'LOW'
        results.append(('Low Light', light_score, confidence))
        
    if blur_score > SCORE_THRESHOLD:
        confidence = 'HIGH' if blur_score > CONFIDENCE_LEVELS['HIGH'] else \
                    'MEDIUM' if blur_score > CONFIDENCE_LEVELS['MEDIUM'] else 'LOW'
        results.append(('Blur', blur_score, confidence))
        
    return results

def main():
    # 定义图片和输出文件夹路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    attachment_folder = 'Attachment1' if CATEGORY_MODE == "train" else 'Attachment2'
    visualization_folder = 'visualization_results_train' if CATEGORY_MODE == "train" else 'visualization_results_test'
    
    image_folder = os.path.join(current_dir, attachment_folder)  # 图片文件夹
    output_folder = os.path.join(current_dir, visualization_folder)  # 可视化结果保存文件夹
    os.makedirs(output_folder, exist_ok=True)  # 如果文件夹不存在则创建
    print(f"Image folder: {image_folder}")
    print(f"Output folder: {output_folder}")
    
    # 准备结果列表，用于保存每张图片的分类结果
    results = []
    # 可视化分析列表，随机选取5张图片
    select_list = random.sample(range(1, MAX_PICTURE_NUM), RANDOM_CHOICE)
    
    # 遍历所有图片，假设图片命名为image_001.png到image_400.png
    for i in range(1, MAX_PICTURE_NUM):
        try:
            # 检测图像后缀：png or jpg?
            for ext in ['.png', '.jpg']:
                suffix = 'image' if CATEGORY_MODE == "train" else 'test'
                image_name = f'{suffix}_{i:03d}{ext}'
                image_path = os.path.join(image_folder, image_name)
                if os.path.exists(image_path):
                    break
            image = read_image(image_path)  # 读取图像

            if image is None:
                print(f"Image {image_name} not found.")
                continue
            else:
                print(f"Processing image: {image_name}")
                
            # 特征分析
            color_features = analyze_color_cast(image)
            light_features = analyze_low_light(image)
            blur_features = analyze_blur(image)
            
            # 分类决策
            classifications = classify_image(color_features, light_features, blur_features)

            # 修改结果记录方式
            result = {
                'ImageName': os.path.basename(image_path),
                'ColorCast': 0,
                'LowIllumination': 0,
                'Blur': 0,
                'ClassificationDecision': []
            }
            
            # 根据分类结果设置对应的标志
            for class_name, score, confidence in classifications:
                if class_name == 'Color Cast':
                    result['ColorCast'] = 1
                elif class_name == 'Low Light':
                    result['LowIllumination'] = 1
                elif class_name == 'Blur':
                    result['Blur'] = 1
                result['ClassificationDecision'].append(f"{class_name}({confidence}:{score:.2f})")
            
            # 如果没有分类结果，标记为Normal
            if not result['ClassificationDecision']:
                result['ClassificationDecision'] = ['Normal']
                
            # 将决策列表转换为字符串
            result['ClassificationDecision'] = ';'.join(result['ClassificationDecision'])
            
            results.append(result)
            
            # 随机选取5张图片进行可视化分析
            if i in select_list:
                # 保存图像的颜色直方图，可视化分析偏色
                save_histogram(image, image_name, output_folder)
                # 保存灰度图的直方图，可视化分析亮度分布
                save_gray_histogram(convert_to_gray(image), image_name, output_folder)
                # 保存频谱图，可视化分析高频能量
                save_fft_spectrum(convert_to_gray(image), image_name, output_folder)
                # 保存高斯模糊图像，可视化分析图像模糊
                save_gaussian_blur(image, image_name, output_folder)
                # 保存LAB颜色空间分析
                save_lab_analysis(image, image_name, output_folder)
                # 保存暗区域标记
                save_dark_regions(image, image_name, output_folder)
                # 保存边缘检测结果
                save_edge_detection(image, image_name, output_folder)
                # 保存综合分析报告
                save_comprehensive_report(image, {
                    'color': color_features,
                    'light': light_features,
                    'blur': blur_features
                }, image_name, output_folder)

        except Exception as e:
            print(f"Error processing image {image_name}: {str(e)}")
            continue

    # 创建DataFrame时使用正确的列名
    df = pd.DataFrame(results)
    
    # 确保输出目录存在
    output_path = os.path.join(current_dir, attachment_folder, 'Answer.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存CSV文件时添加错误处理
    try:
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error saving CSV file: {str(e)}")

if __name__ == '__main__':
    main()
