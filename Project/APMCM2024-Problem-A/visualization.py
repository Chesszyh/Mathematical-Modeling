# visualization.py
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from config import *

def save_histogram(image, image_name, output_dir):
    """保存颜色直方图分析"""
    plt.figure(figsize=(10, 4))
    colors = ('b', 'g', 'r')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        # 添加平滑处理
        hist = cv2.GaussianBlur(hist, (1, 7), 1.5)
        plt.plot(hist, color=color, alpha=0.8, label=f'{color.upper()} Channel')
    
    plt.title('Color Distribution Analysis')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # 添加y轴对数刻度，更好地显示分布
    plt.yscale('log')
    plt.savefig(os.path.join(output_dir, f'{image_name}_color_hist.png'))
    plt.close()

def save_gray_histogram(gray_image, image_name, output_dir):
    """保存灰度直方图分析"""
    plt.figure(figsize=(10, 4))
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    plt.plot(hist, 'k')
    plt.fill_between(range(256), hist.ravel(), alpha=0.3)
    plt.axvline(x=LOW_LIGHT_THRESHOLDS['mean_brightness'], 
                color='r', linestyle='--', 
                label='Low Light Threshold')
    plt.title('Brightness Distribution Analysis')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'{image_name}_gray_hist.png'))
    plt.close()

def save_fft_spectrum(gray_image, image_name, output_dir):
    """保存频谱分析图"""
    f = np.fft.fft2(gray_image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(magnitude_spectrum, cmap='viridis')
    plt.title('Frequency Spectrum')
    plt.colorbar(label='Magnitude (dB)')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f'{image_name}_spectrum.png'))
    plt.close()

def save_gaussian_blur(image, image_name, output_dir):
    """保存高斯模糊分析"""
    # 生成不同程度的高斯模糊
    blurred_3 = cv2.GaussianBlur(image, (3, 3), 0)
    blurred_5 = cv2.GaussianBlur(image, (5, 5), 0)
    blurred_7 = cv2.GaussianBlur(image, (7, 7), 0)
    
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original')
    axes[1].imshow(cv2.cvtColor(blurred_3, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Gaussian 3x3')
    axes[2].imshow(cv2.cvtColor(blurred_5, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Gaussian 5x5')
    axes[3].imshow(cv2.cvtColor(blurred_7, cv2.COLOR_BGR2RGB))
    axes[3].set_title('Gaussian 7x7')
    
    for ax in axes:
        ax.axis('off')
    
    plt.savefig(os.path.join(output_dir, f'{image_name}_gaussian_blur.png'))
    plt.close()

def save_lab_analysis(image, image_name, output_dir):
    """保存LAB颜色空间分析"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original')
    axes[1].imshow(l, cmap='gray')
    axes[1].set_title('L Channel')
    axes[2].imshow(a, cmap='RdYlGn')
    axes[2].set_title('a Channel')
    axes[3].imshow(b, cmap='RdYlBu')
    axes[3].set_title('b Channel')
    
    for ax in axes:
        ax.axis('off')
        
    plt.savefig(os.path.join(output_dir, f'{image_name}_lab_analysis.png'))
    plt.close()

def save_dark_regions(image, image_name, output_dir):
    """保存暗区域标记图"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用自适应阈值检测暗区域
    mean_brightness = np.mean(gray)
    dark_threshold = mean_brightness * 0.6  # 根据平均亮度动态调整阈值
    dark_mask = gray < dark_threshold
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(dark_mask, cmap='hot')
    plt.title('Dark Regions')
    plt.colorbar(label='Dark pixel probability')
    plt.axis('off')
    
    plt.savefig(os.path.join(output_dir, f'{image_name}_dark_regions.png'))
    plt.close()

def save_edge_detection(image, image_name, output_dir):
    """保存边缘检测分析"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 添加高斯模糊预处理
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 使用Canny边缘检测替代简单的Sobel
    edges_canny = cv2.Canny(blurred, 50, 150)
    
    # 增强Laplacian边缘
    edges_laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    edges_laplacian = np.uint8(np.absolute(edges_laplacian))
    edges_laplacian = cv2.normalize(edges_laplacian, None, 0, 255, cv2.NORM_MINMAX)
    
    plt.figure(figsize=(15, 4))
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(edges_canny, cmap='gray')
    plt.title('Canny Edge Detection')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(edges_laplacian, cmap='gray')
    plt.title('Enhanced Laplacian Edge')
    plt.axis('off')
    
    plt.savefig(os.path.join(output_dir, f'{image_name}_edge_detection.png'))
    plt.close()

def save_comprehensive_report(image, features, image_name, output_dir):
    """保存综合分析报告"""
    fig = plt.figure(figsize=(15, 10))
    
    # 原始图像
    plt.subplot(331)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    # 颜色分析
    plt.subplot(332)
    for i, color in enumerate(['b', 'g', 'r']):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color, alpha=0.8)
    plt.title(f'Color Analysis\nΔgray={features["color"]["delta_gray"]:.2f}')
    
    # LAB分析
    plt.subplot(333)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    plt.imshow(lab[:,:,0], cmap='gray')
    plt.title(f'LAB Analysis\nΔlab={features["color"]["delta_lab"]:.2f}')
    plt.axis('off')
    
    # HSV分析
    plt.subplot(334)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    plt.hist(hsv[:,:,2].flatten(), bins=256, range=(0,256))  # 使用V通道而不是S通道
    plt.title(f'HSV Analysis\nΔhsv={features["color"]["delta_hsv"]:.2f}')
    
    # 亮度分析
    plt.subplot(335)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.hist(gray.ravel(), bins=256)
    plt.title(f'Brightness Analysis\nμ={features["light"]["mean_brightness"]:.2f}')
    
    # 暗区域
    plt.subplot(336)
    dark_mask = gray < LOW_LIGHT_THRESHOLDS['mean_brightness']
    plt.imshow(dark_mask, cmap='hot')
    plt.title(f'Dark Regions\nRatio={features["light"]["dark_ratio"]:.2f}')
    plt.axis('off')
    
    # 边缘检测
    plt.subplot(337)
    edges = cv2.Laplacian(gray, cv2.CV_64F)
    plt.imshow(np.abs(edges), cmap='gray')
    plt.title(f'Edge Analysis\nVar={features["blur"]["laplacian_var"]:.2f}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{image_name}_comprehensive.png'))
    plt.close()


def compare_images(original, enhanced, image_name, method_name, output_dir):
    """保存原始图像和增强后图像的对比
    参数:
        original: 原始图像
        enhanced: 增强后图像
        image_name: 图像名称
        method_name: 方法名称，用于区分任务3和任务4
        output_dir: 保存目录
    """
    plt.figure(figsize=FIGURE_SIZE)
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    plt.title(f'Enhanced Image ({method_name})')
    plt.axis('off')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, f"{image_name}_{method_name}_comparison.png"))
    plt.close()

def plot_metrics(df, output_dir):
    """绘制评价指标的对比图
    参数:
        df: 包含评价指标的DataFrame
        output_dir: 保存目录
    """
    metrics = ['PSNR', 'UCIQE', 'UIQM', 'SSIM']
    for metric in metrics:
        values_task3 = df[f'{metric}_Task3'].values
        values_task4 = df[f'{metric}_Task4'].values
        plt.figure(figsize=(10, 6))
        index = np.arange(len(df))
        bar_width = 0.35
        plt.bar(index, values_task3, bar_width, label='Task3')
        plt.bar(index + bar_width, values_task4, bar_width, label='Task4')
        plt.xlabel('Image Index')
        plt.ylabel(metric)
        plt.title(f'Comparison of {metric} between Task3 and Task4')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric}_comparison.png'))
        plt.close()
        
def plot_color_analysis(image, features, save_path):
    """绘制颜色分析可视化图"""
    plt.figure(figsize=(15, 5))
    
    # RGB直方图
    plt.subplot(131)
    colors = ('b', 'g', 'r')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color, alpha=0.8, label=f'{colors[i].upper()} Channel')
    plt.title(f'RGB Histogram\nΔgray={features["delta_gray"]:.2f}')
    plt.legend()
    
    # LAB空间散点图
    plt.subplot(132)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    plt.scatter(lab[:,:,1].flatten(), lab[:,:,2].flatten(), 
               alpha=0.1, s=1)
    plt.axhline(y=128, color='r', linestyle='--')
    plt.axvline(x=128, color='r', linestyle='--')
    plt.title(f'LAB Color Space\nΔlab={features["delta_lab"]:.2f}')
    
    # HSV饱和度分布
    plt.subplot(133)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    plt.hist(hsv[:,:,1].flatten(), bins=50)
    plt.title(f'Saturation Distribution\nΔhsv={features["delta_hsv"]:.2f}')
    
    plt.savefig(save_path)
    plt.close()
    