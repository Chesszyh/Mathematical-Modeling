# CCF-A类相关论文笔记

Prompt: 根据摘要，用中文概述以下方法的核心insight和最显著优点。

！：已读论文
!!：已读的官方参考文献
.：废弃的文献
..: 没代码或代码star很少，且废弃的文献

## 1. TIP2023 Pugan - Physical model-guided underwater image enhancement using gan with dual-discriminators

[Github](https://github.com/rmcong/PUGAN_TIP2023)

### Insight 和优点概述

这篇论文提出了**PUGAN（Physical model-guided GAN）**，结合物理模型与生成对抗网络（GAN）两种方法的优势，解决水下图像增强（UIE）任务中的关键问题，以下是主要的贡献和优点：

1. **方法创新**：
   - **物理模型指导生成**：引入物理模型，设计参数估计子网络 (**Par-subnet**)，通过逆向计算物理参数生成颜色增强图像，作为主网络的辅助信息，提高模型对不同场景的适应性。
   - **两流交互增强子网络（TSIE-subnet）**：利用**两流交互机制**增强水下图像，设计了退化量化（DQ）模块以分析场景退化程度，针对重点区域进行强化增强。

2. **视觉质量优化**：
   - **双判别器设计**：采用内容风格双判别器，实现对抗约束，平衡真实感和视觉美感，使得生成图像在细节和整体色调上更符合人眼审美。

3. **实用性和鲁棒性**：
   - 结合物理模型和GAN技术的优点，既能在复杂水下场景中自适应不同退化条件，又在增强过程中保留自然的视觉风格。
   - 提高了在色彩失真、低对比度和模糊细节上的恢复能力，使其能够兼顾定量指标和定性视觉效果。

4. **性能验证**：
   - 在三个基准数据集上进行了广泛实验，结果表明 PUGAN 在定量（例如 PSNR、SSIM 等）和定性（视觉质量）指标上均优于现有的最先进方法。

---

**优点总结**：

- **结合物理模型与深度学习的优点**，克服单一方法的局限性，既能自适应退化场景又提升视觉质量。
- **创新的退化量化模块**：对退化程度的量化使得增强算法更精细化，重点区域处理更突出。
- **内容风格双判别器**：在对抗约束中综合考虑真实感和审美效果，生成图像的艺术性和真实性均得到提升。
- **全面的实验验证**：展示出在多种退化场景下的泛化性和优越性能，具有较强的应用潜力。



## 2. TIP2023 U-shape transformer for underwater image enhancement

[Github](https://github.com/LintaoPeng/U-shape_Transformer_for_Underwater_Image_Enhancement)

![alt text](image.png)

### GPT总结

#### 核心思想 (Insights)：

1. **大规模数据集构建**：  
   - 构建了一个包含4279对图像的大规模水下图像（LSUI）数据集，涵盖多种水下场景和高保真参考图像，解决了现有方法缺乏多样化、标准化数据集的问题。
   - **Insight**：丰富的数据资源为数据驱动方法提供了坚实基础，提升了模型的泛化能力。

2. **U型Transformer网络**：  
   - 首次将Transformer引入水下图像增强任务，设计了一个U型结构的Transformer网络，专注于多尺度特征融合和全局特征建模。  
   - **模块创新**：  
     - **CMSFFT模块（通道多尺度特征融合Transformer）**：增强模型对不同颜色通道的关注，特别是针对衰减严重的区域。  
     - **SGFMT模块（空间全局特征建模Transformer）**：捕获空间维度的全局特征，提升复杂退化场景的还原能力。

3. **新颖的多颜色空间损失函数**：  
   - 根据人类视觉原理，结合RGB、LAB和LCH颜色空间设计了一种新损失函数，旨在提升图像的对比度和饱和度。  
   - **Insight**：多颜色空间的融合优化了增强结果的视觉效果，贴近人眼感知。

---

#### 优点 (Advantages)：

1. **数据集贡献**：  
   LSUI 数据集为水下图像增强领域提供了一个高质量的开源数据基准，推动了该领域的研究进展。

2. **性能提升**：  
   实验表明，提出的方法在现有数据集上超越了SOTA技术，在量化指标上提升超过 **2dB**，具有显著的性能优势。

3. **Transformer创新**：  
   将Transformer模型成功应用于水下图像增强，开创了新方向，尤其是CMSFFT和SGFMT模块的设计，使模型更加关注颜色通道和空间区域的复杂衰减问题。

4. **视觉质量优化**：  
   使用多颜色空间损失函数，使得增强后的图像对比度更高、色彩更自然，满足实际应用需求。

---

#### 总结：  
该方法通过大规模数据集的构建、U型Transformer网络的创新设计和多颜色空间损失函数的引入，显著提升了水下图像的增强效果，不仅在理论上开辟了新方向，而且在实际性能和视觉质量上实现了全面超越。

### 我的笔记

285引用，2023年的TIP，比上面的多(也有可能是虚高，毕竟提供了一个大数据集)

## 3. TIP2023 Domain adaptation for underwater image enhancement

### 我的笔记

2023，引用68

## 4. TIP2021-A Underwater image enhancement via medium transmission-guided multi-color space embedding

结合物理模型和深度学习，

## 5. IJCV2024 HCLR-Net Hybrid Contrastive Learning Regularization with Locally Randomized Perturbation for Underwater Image Enhancement

- Intro
  - 分类法与赛题观点很类似
    - Traditional methods
      - Model-free:directly process the pixel values
      - Model-based: 复杂环境下难以适应
    - Data-Driven methods：UIE(Underwater Image Enhancement)
      - overlook potential relationships in feature space，导致难以提取有用特征
      - require abundant pairwise data: hard to obtain 不现实
  - 本文认为的关键；accurate encoding of absolute positional information
    - **AHA(Adaptive Hybrid Attention)**:effectively capturing both short and long-range
    - **HCLR(Hybrid Contrastive Learning Regularization)**:enhancing the feature representation(?)

## Overall

- 工具：
  - [MindSpore Lite](https://www.mindspore.cn/lite/docs/zh-CN/r2.4.0/use/downloads.html)

### 1. Introduction

### 2. Related Work

- 由于手工制作的特征的代表性有限，传统方法的性能往往无法与基于深度学习的方法相比
- **传统UIE(Underwater Image Enhancement)方法**：调整像素值，**对应问题3的解决方案**
  - **论文**：
    - [CCFA IEEE2018 Color Balance and Fusion for Underwater Image Enhancement]：更好的黑暗区域曝光
  - **分类**：单指标、多指标(1引用)
    - 动态像素范围拉伸[15]
    - 像素分布调整[16]
    - 图像融合[17]-[19]
  - **缺点**
    - 基于物理模型的方法，重点是估计物理模型中的所有参数并**反演**(即逆转退化过程)清晰的水下图像；通常依赖于先验知识和假设，对复杂场景适应性差；
      - 水下暗信道先验[13]、衰减曲线先验[15]、模糊先验[18]和最小信息先验[20]等(2)
      - 目前广泛应用的水下图像增强方法是基于物理模型的，基于先验信息估计水下成像模型的参数。这些先验包括红信道先验[28]、水下暗信道先验[3]、最小信息先验[4]、模糊先验[29]、一般暗信道先验[30]等。例如，Peng和Cosman[29]提出了一种基于图像模糊和光吸收的水下图像深度估计算法。(4)


利用估计的深度，根据水下成像模型恢复清晰的水下图像。Peng等人


[30]进一步提出在处理恶劣天气下捕获的不同图像之前，对暗通道进行泛化。提出了一种新的b[19]水下图像形成模型。


基于该模型，提出了一种基于水下RGB-D图像的水下图像色彩校正方法[21]。


这些基于物理模型的方法要么耗时，要么对水下图像的类型敏感。


此外，复杂水下成像参数的准确估计对当前基于物理模型的方法[3]，[4]，[28]-[30]提出了挑战。例如，先前在[29]中使用的模糊度并不总是有效，特别是对于清晰的水下图像。相比之下，我们的方法利用了基于物理模型和基于数据驱动的方法的优点，可以更准确地恢复水下图像。
    - 基于视觉先验的方法，忽视了**水下图像在不同颜色通道和空间区域的衰减程度不一致**，在复杂水下环境的真实水下图像上表现不佳。

- 基于(深度)学习的UIE方法，或数据驱动的方法：**对应问题4的解决方案**
  - 两种技术路线：(2)
    - 设计端到端模块
      - 基于水下场景先验的水下图像训练的新颖CNN架构: [PR2020-B Underwater scene prior inspired deep underwater image and video enhancement]
      - 
    - 直接利用深度模型估计物理参数，然后基于退化模型恢复干净图像。
  - CNN的问题：(2)
    - 均匀卷积核不能表征水下图像在不同颜色通道和空间区域的不一致衰减
    - CNN架构更关注局部特征，而对长依赖和全局特征建模无效(这是CNN的广义局限性吗？)
    - CNN + Transformer: 引导网络更加关注衰减更严重的颜色通道和空间区域
  - 缺点：
    - **图像数量少、水下场景少、甚至不是真实场景**(2)
      - 合成数据与真实数据之间存在显著域间隙，即**域间间隙**(3)
      - 复杂多变的水下环境也导致了真实数据本身之间存在很大的分布差距，即**域内间隙**(3)
        - 论文2提供了一个大型数据集，4279对真实水下图像(是否有3上面那条提到的问题？)
    - 将现有神经网络直接应用于水下图像增强任务，效果不佳：忽略了水下成像的独特性
- 比较两种方法在不同应用场景下的优劣势，以及如何结合两种方法的优势，能够提出实际应用中水下视觉增强的可行性建议

### 3. Proposed Method

### 4. Experiments

#### State-of-the-art Methods

比较方法包括6种非学习方法（即GDCP[4]、ACDE[25]、HLRP[67]、MLLE[68]、UNTV[69]、SPDF[70]）和8种深度学习方法（即deep-sesr[59]、FUnIE-GAN[5]、WaterNet[2]、UWCNN[46]、JI-Net[56]、ACPAB[53]、TOPAL[55]、Ucolor[6]）。需要说明的是，WaterNet[2]、FUnIE-GAN[5]、TOPAL[55]也是基于gan的方法。(1引用)

#### Evaluation Metrics

- 关键评估指标
  - **PSNR**：值越高表示图像内容越接近
    - 有关PSNR的争议文章：[IEEE2012-PSNR Peak signal-to-noise ratio revisited - Is simple beautiful]
  - **SSIM**；值越高表示结构和纹理越相似
- 辅助指标/无参考指标：值越高表明人类视觉感知越好，但不一定能准确度量性能
  - **UCIQE**：[论文](TIP2015-A UCIQE - An Underwater Color Image Quality Evaluation Metric)，水下彩色图像质量评价：色彩密度、饱和度和对比度的线性组合，用于定量评价水下图像的不均匀色偏、模糊和低对比度(参考赛题翻译)
  - **UIQM**：水下图像质量测量：综合考虑水下图像的颜色度、清晰度/锐度和对比度属性的评价指标（参考赛题翻译）
  - **NIQE**：自然图像质量评价，数值越小，视觉质量越高
    - 参考：A. Mittal, R. Soundararajan, and A. C. Bovik, “Making a “completely
blind” image quality analyzer,” IEEE Signal Process. Lett., vol. 20, no. 3,
pp. 209–212, 2013. 8


### 5. Conclusion



## 参考论文

### 理论基础

- 光散射模型：.CVPR2019-A Sea-thru A Method For Removing Water From Underwater Images
  - Derya Akkaynak and Tali Treibitz. 2019. Sea-thru: A method for removing water
from underwater images. In Proceedings of the IEEE/CVF conference on Computer
Vision and Pattern Recognition. 1682–1691.

