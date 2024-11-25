# main_enhancement.py is a solution file for the enhancement tasks of the APMCM 2024 Problem A.
# It contains the main logic for the image enhancement tasks.

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from config import *
from utils import *
from enhancement import *
from visualization import *
from train import *
from model import *
from dataset import UnderwaterDataset
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
MODE='test' # 'train' or 'test'

def main():
    # 定义图片和输出文件夹路径
    image_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Attachment1')
    test_image_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Attachment2')
    output_folder_task3 = os.path.join(image_folder, 'enhanced_task3')
    output_folder_task4 = os.path.join(test_image_folder, 'enhanced_task4')
    visualization_folder = os.path.join('visualization_results')
    os.makedirs(output_folder_task3, exist_ok=True)
    os.makedirs(output_folder_task4, exist_ok=True)
    os.makedirs(visualization_folder, exist_ok=True)

    # 加载任务1的分类结果
    classification_results = load_classification_results(os.path.join(image_folder,'Answer.csv'))

    # 准备结果列表，用于保存每张图片的评价指标
    results = []
    
    # 选择模式：训练或测试
    if MODE == 'train':
        # 任务3：首先遍历400张图片，根据分类结果应用对应的增强方法；之后再把这里改成遍历test前缀的12张图片
        for i in range(1, 401):
            for ext in ['.png', '.jpg']:
                image_name = f'image_{i:03d}{ext}'
                image_path = os.path.join(image_folder, image_name)
                if os.path.exists(image_path):
                    break
            image = read_image(image_path)  # 读取图像

            # 调试信息
            print(f"Processing image: {image_name}")
            print(f"Image path: {image_path}")

            if not os.path.exists(image_path):
                print(f"Image file {image_path} does not exist.")
                continue

            # 获取图像的分类结果
            classifications = classification_results.get(image_name, [])

            # 任务3：根据分类结果，应用对应的增强方法
            enhanced_task3 = enhance_with_basic_methods(image, classifications)
            
            # 保存增强后的图像
            output_image_folder = os.path.join(output_folder_task3, image_name)
            save_image(enhanced_task3, output_image_folder)
            print(f"Enhanced image saved to {output_image_folder}")
            # 计算评价指标
            psnr_task3 = calculate_psnr(image, enhanced_task3)
            ssim_task3 = calculate_ssim(image, enhanced_task3)
            uciqe_task3 = calculate_uciqe(enhanced_task3)
            uiqm_task3 = calculate_uiqm(enhanced_task3)

        # 任务4：在400张图片基础上，训练深度学习模型
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 定义数据增强变换
        data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
        ])

        # 初始化数据集时传入transform
        original_image_folder = image_folder # 应是Attachment1文件夹
        enhanced_image_folder = os.path.join(output_folder_task3) # 应是Attachment1/enhanced_task3文件夹
        print(f"Degraded image folder: {original_image_folder}, Enhanced image folder: {enhanced_image_folder}")
        train_dataset = UnderwaterDataset(original_image_folder, enhanced_image_folder, transform=data_transforms)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=False  # 关闭pin_memory
        )

        # 选择模型
        model = select_model(MODEL_TYPE)

        # 训练模型
        model = train_model(train_loader, model, EPOCHS, device, MODEL_SAVE_PATH)
        
        # 保存模型
        save_model(model, MODEL_SAVE_PATH)

    # 加载模型进行推理
    model = select_model(MODEL_TYPE)
    model = load_model(model, MODEL_SAVE_PATH)
    # 模型剪枝
    model = prune_model(model, amount=0.2)
    model.eval()

    # 任务4：遍历12张图片，使用基础模型 + 深度学习模型进行增强
    for i in range(1, 13):
        image_name = f'test_{i:03d}.png'
        image_path = os.path.join(test_image_folder, image_name)                
        if not os.path.exists(image_path):
            print(f"Image file {image_path} does not exist.")
            continue
        
        image = read_image(image_path)  # 读取图像
        print(f"Processing image: {image_name}")

        # 在处理测试图片前，正确加载分类结果
        classification_results = load_classification_results(os.path.join(test_image_folder, 'Answer.csv'))

        # 获取当前图片的分类结果
        classifications = classification_results.get(image_name, [])
        enhanced_task3 = enhance_with_basic_methods(image, classifications)
        enhanced_task4 = enhance_with_model(image, model)
        save_image(enhanced_task4, os.path.join(output_folder_task4, image_name))

        # 计算评价指标
        psnr_task3 = calculate_psnr(image, enhanced_task3)
        ssim_task3 = calculate_ssim(image, enhanced_task3)
        uciqe_task3 = calculate_uciqe(enhanced_task3)
        uiqm_task3 = calculate_uiqm(enhanced_task3)
        psnr_task4 = calculate_psnr(image, enhanced_task4)
        ssim_task4 = calculate_ssim(image, enhanced_task4)
        uciqe_task4 = calculate_uciqe(enhanced_task4)
        uiqm_task4 = calculate_uiqm(enhanced_task4)

        # 可视化对比结果
        compare_images(image, enhanced_task3, image_name, 'Task3', visualization_folder)
        compare_images(image, enhanced_task4, image_name, 'Task4', visualization_folder)

        # 绘制颜色分析可视化图
        color_features = analyze_color_cast(image)
        plot_color_analysis(image, color_features, os.path.join(visualization_folder, f'{image_name}_color_analysis.png'))
        print(f"Color analysis saved to {os.path.join(visualization_folder, f'{image_name}_color_analysis.png')}")

        # 记录结果
        results.append({
            'ImageName': image_name,
            'ImageClassification': ', '.join(classifications),
            'PSNR_Task3': psnr_task3,
            'UCIQE_Task3': uciqe_task3,
            'UIQM_Task3': uiqm_task3,
            'SSIM_Task3': ssim_task3,
            'PSNR_Task4': psnr_task4,
            'UCIQE_Task4': uciqe_task4,
            'UIQM_Task4': uiqm_task4,
            'SSIM_Task4': ssim_task4
        })

    # 将结果保存到Answer.csv文件
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(test_image_folder, 'Answer_with_metric.csv'), index=False)
    print(f"Results saved to {os.path.join(test_image_folder, 'Answer_with_metric.csv')}")

    # 任务5：结果比较与分析
    # 绘制评价指标的对比图
    plot_metrics(df, visualization_folder)
    print(f"Metrics comparison saved to {os.path.join(visualization_folder, 'metrics_comparison.png')}")
    
if __name__ == '__main__':
    main()