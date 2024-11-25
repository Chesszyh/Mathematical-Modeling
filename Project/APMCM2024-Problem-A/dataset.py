import os
import cv2
import numpy as np
from utils import read_image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from config import *

class UnderwaterDataset(Dataset):
    """自定义数据集：使用原图作为输入，增强图像作为目标"""
    def __init__(self, original_folder, enhanced_folder, transform=None):
        self.original_folder = original_folder
        self.enhanced_folder = enhanced_folder
        self.transform = transform
        
        # 获取两个文件夹中共有的图像
        original_images = set(os.listdir(original_folder))
        enhanced_images = set(os.listdir(enhanced_folder))
        self.image_names = list(original_images.intersection(enhanced_images))
        
        print(f"Found {len(self.image_names)} images for training")
        if len(self.image_names) == 0:
            raise ValueError(f"No matching images found in {original_folder} and {enhanced_folder}")

    def __len__(self):
        """返回数据集的样本数量"""
        return len(self.image_names)
    
    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        
        # 读取原始图像和增强图像
        input_path = os.path.join(self.original_folder, image_name)
        target_path = os.path.join(self.enhanced_folder, image_name)
        
        input_image = read_image(input_path)
        target_image = read_image(target_path)
        
        if input_image is None or target_image is None:
            raise FileNotFoundError(f"Image {image_name} not found")

        # 调整图像大小
        input_image = cv2.resize(input_image, (256, 256))
        target_image = cv2.resize(target_image, (256, 256))

        # 数据增强
        if self.transform:
            # 使用相同的随机变换
            seed = np.random.randint(2147483647)
            
            torch.manual_seed(seed)
            input_pil = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
            input_tensor = self.transform(input_pil)
            
            torch.manual_seed(seed)
            target_pil = Image.fromarray(cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB))
            target_tensor = self.transform(target_pil)
        else:
            input_tensor = torch.from_numpy(input_image.transpose((2, 0, 1))).float() / 255.0
            target_tensor = torch.from_numpy(target_image.transpose((2, 0, 1))).float() / 255.0

        return {
            'input': input_tensor,
            'target': target_tensor
        }

def get_transforms(phase='train'):
    """获取数据增强转换"""
    if phase == 'train':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self):
        self.patience = EARLY_STOPPING_PATIENCE
        self.min_delta = EARLY_STOPPING_DELTA
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
        
    def __call__(self, val_loss, model):
        """
        Args:
            val_loss: 当前验证损失
            model: 当前模型
        Returns:
            early_stop: 是否应该停止训练
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop
    
    def save_checkpoint(self, model):
        """保存最佳模型"""
        self.best_model = {key: value.cpu() for key, value in model.state_dict().items()}

    def load_best_model(self, model):
        """加载最佳模型"""
        if self.best_model is not None:
            model.load_state_dict(self.best_model)