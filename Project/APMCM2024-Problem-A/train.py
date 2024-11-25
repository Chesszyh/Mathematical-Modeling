# train.py
import os
import gc
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from config import LEARNING_RATE, WEIGHT_DECAY, LR_SCHEDULER_STEP, LR_GAMMA
from torch.optim.lr_scheduler import StepLR
from model import SSIMLoss
from dataset import EarlyStopping   

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # 临时修复Jupyter Notebook中的OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized错误

class PerceptualLoss(nn.Module):
    """感知损失"""
    def __init__(self):
        super().__init__()
        vgg = vgg16(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
    def forward(self, enhanced, original):
        enhanced_features = self.feature_extractor(enhanced)
        original_features = self.feature_extractor(original)
        return F.mse_loss(enhanced_features, original_features)

def plot_learning_curves(log_dir, epoch):
    """绘制学习曲线
    Args:
        log_dir: TensorBoard日志目录
        epoch: 当前训练轮数
    """
    plt.figure(figsize=(15, 5))
    
    # 使用event_accumulator获取数据
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    loss_steps = ea.Scalars('Loss/epoch')
    lr_steps = ea.Scalars('lr')
    
    # 绘制损失曲线
    plt.subplot(1, 3, 1)
    if loss_steps:
        steps = [x.step for x in loss_steps]
        values = [x.value for x in loss_steps]
        plt.plot(steps, values, 'b-', label='Training Loss')
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
    
    # 绘制学习率曲线
    plt.subplot(1, 3, 2)
    if lr_steps:
        steps = [x.step for x in lr_steps]
        values = [x.value for x in lr_steps]
        plt.plot(steps, values, 'r-', label='Learning Rate')
        plt.title('Learning Rate Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        plt.legend()
    
    # 保存图像
    save_dir = 'training_visualization'
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'learning_curves_epoch_{epoch}.png'))
    plt.close()

def print_memory_usage():
    import psutil
    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
def train_model(train_loader, model, num_epochs, device, save_path):
    """改进的训练函数"""
    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 创建tensorboard日志目录
    log_dir = 'runs/underwater_enhancement'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    scaler = torch.amp.GradScaler('cuda')  # 混合精度训练
    
    # 损失函数
    criterion_pixel = nn.MSELoss()
    criterion_perceptual = PerceptualLoss().to(device)
    criterion_ssim = SSIMLoss().to(device)
    
    # 优化器和学习率调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    scheduler = StepLR(
        optimizer, 
        step_size=LR_SCHEDULER_STEP,
        gamma=LR_GAMMA
    )
    
    # 早停
    early_stopping = EarlyStopping()
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, data in enumerate(train_loader):
            inputs = data['input'].to(device)
            targets = data['target'].to(device)  # 添加目标图像
            
            outputs = model(inputs)
            
            # 确保输出和目标尺寸一致
            if outputs.size() != targets.size():
                targets = F.interpolate(targets, size=outputs.size()[2:], mode='bilinear', align_corners=False)
            
            loss_pixel = criterion_pixel(outputs, targets)
            loss_perceptual = criterion_perceptual(outputs, targets)
            loss_ssim = criterion_ssim(outputs, targets)
            loss = loss_pixel + 0.1 * loss_perceptual + 0.05 * loss_ssim
            
            # 检查损失值是否为NaN或Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print("Loss is NaN or Inf, skipping this batch.")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 记录batch级别的损失
            if batch_idx % 10 == 0:
                writer.add_scalar('Loss/batch', loss.item(), 
                                epoch * len(train_loader) + batch_idx)
                
                # 可视化样本
                if batch_idx == 0:
                    writer.add_images('Input', inputs[:4], epoch)
                    writer.add_images('Output', outputs[:4], epoch)
                    writer.add_images('Target', targets[:4], epoch)
                    
                gc.collect() # 回收内存，防止Dataloader Out of Memory
                if device == 'cuda':
                    torch.cuda.empty_cache()

        # 计算并记录epoch级别的损失
        avg_loss = epoch_loss / len(train_loader)
        writer.add_scalar('Loss/epoch', avg_loss, epoch)
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('lr', current_lr, epoch)
        
        # 更新学习率
        scheduler.step(avg_loss)
        
        print_memory_usage()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, LR: {current_lr:.6f}')
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_path)
        
        # 早停
        early_stopping(avg_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        # 每5个epoch绘制一次学习曲线
        if (epoch + 1) % 5 == 0:
            plot_learning_curves(log_dir, epoch)
    
    writer.close()
    return model