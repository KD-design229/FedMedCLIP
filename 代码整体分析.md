# Federated CLIP for Resource-Efficient Heterogeneous Medical Image Classification - 代码整体分析

**论文**: Federated CLIP for Resource-Efficient Heterogeneous Medical Image Classification (AAAI 2026)
**作者**: Yihang Wu, Ahmad Chaddad
**论文链接**: https://arxiv.org/abs/2511.07929

---

## 一、项目概述

本项目是一个**联邦学习(Federated Learning)**框架，用于在资源受限的异构环境中进行医学图像分类。核心创新是结合了CLIP(Contrastive Language-Image Pre-training)模型与联邦平均(FedAVG)和参数高效微调(PEFT)方法，实现跨多个医疗中心的协作学习。

### 主要特点：
- **联邦学习架构**：多客户端(多医疗中心)协作学习，服务器端汇聚
- **CLIP基础**：利用视觉-语言预训练模型的迁移学习能力
- **参数高效**：通过特征注意力模块(FAM)和MLP微调，减少参数量
- **模型量化**：通过模型压缩和解压来优化通信成本
- **多种方法对比**：包含FedAVG、FedCLIP、PromptFL、LoRA、LP++等多种基准和提出方法

---

## 二、核心系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    联邦学习系统架构                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           全局服务器 (Global Server)                    │   │
│  │ ┌────────────────────────────────────────────────────┐ │   │
│  │ │ - CLIP模型 (冻结的视觉编码器)                      │ │   │
│  │ │ - FAM (特征注意力模块)                              │ │   │
│  │ │ - 聚合器 (Aggregator)                               │ │   │
│  │ └────────────────────────────────────────────────────┘ │   │
│  └─────────────────┬──────────────────────────────────────┘   │
│                    │                                             │
│        ┌───────────┴───────────┬──────────────┬─────────────┐  │
│        │                       │              │             │  │
│  ┌─────▼──────┐         ┌──────▼──────┐ ┌───▼──────┐ ┌────▼──┐ │
│  │  Client 0  │         │  Client 1   │ │ Client 2 │ │Client N│ │
│  │(医疗中心1)  │         │(医疗中心2)   │ │(医疗中心3)│ │(全局)  │ │
│  └────────────┘         └─────────────┘ └──────────┘ └────────┘ │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

每个Client包含:
  ├─ CLIP模型副本 (冻结)
  ├─ FAM特征注意力模块 (可训练)
  └─ MLP分类头 (可训练)
```

---

## 三、文件结构及功能说明

### 3.1 主程序入口文件

#### **Ours.py** (核心方法)
- **功能**: 实现提出的FAM-MLP联邦学习方法
- **关键参数**:
  - `--aggmode='att'`: 使用注意力聚合(PEFT方法)
  - `--method='attn_mlp'`: 特征注意力+MLP
  - `--lr=5e-4`: FAM学习率
  - `--lr_mlp=1e-4`: MLP学习率
- **流程**:
  1. 初始化CLIP模型和FAM模块
  2. 每个联邦轮迭代中，客户端本地训练
  3. 使用量化通信进行参数聚合
  4. 验证和测试阶段

#### **DN-FedAVG-1.py** (基准方法)
- **功能**: 标准的联邦平均算法，更新整个CLIP模型
- **关键特点**:
  - `--aggmode='avg'`: 平均聚合所有参数
  - 直接微调CLIP的所有参数(包括冻结的部分)

#### **DN-FedCLIP-1.py** (对比方法)
- **功能**: 仅微调FAM模块的FedAVG
- **关键特点**:
  - `--aggmode='att'`: 注意力聚合
  - 只训练fea_attn参数

#### **LP++.py** (线性探针++)
- **功能**: 线性探针+++ 方法，仅训练线性分类头
- **关键特点**: 最小化可训练参数

#### **LoRA_OF-2.py**
- **功能**: 使用LoRA(Low-Rank Adaptation)微调CLIP
- **关键特点**: 低秩分解来减少参数

#### **DN-Prompt.py**
- **功能**: 提示学习方法，学习文本提示

#### **DN-Zero-Shot-1.py**
- **功能**: 零样本学习基准(无学习)

#### **DN-Individual-1.py**
- **功能**: 本地训练基准(无联邦学习)

---

### 3.2 核心模型文件 (`nets/`)

#### **models.py** (模型定义)

**关键类**:

1. **ClipModelat** - 主要CLIP包装类
   ```python
   class ClipModelat(nn.Module):
       def __init__(self, model_name='ViT-B/32', attention=True, freezepy=True):
           # model_name支持: RN50, RN101, RN50x4, ViT-B/32, ViT-L/14等
           # attention=True: 包含FAM模块
           # freezepy=True: 冻结CLIP参数
   ```

2. **特征注意力模块 (FAM)** - 核心创新
   ```python
   self.fea_attn = nn.Sequential(
       nn.Linear(512, 512),           # 线性层
       nn.InstanceNorm1d(512),        # 实例归一化
       nn.ReLU6(),                    # ReLU6激活
       nn.Linear(512, 512),           # 第二线性层
       nn.Softmax(dim=1)              # Softmax注意力
   )
   ```
   - **作用**: 学习对CLIP特征的注意力权重
   - **参数量**: 仅需~526K参数 (vs CLIP的几亿参数)

3. **MLP分类头**
   ```python
   class MLP(nn.Module):
       def __init__(self, input_size, hidden_size, num_classes):
           self.linear1 -> self.bn1 -> self.relu1 ->
           self.linear2 -> self.bn2 -> self.relu2 ->
           self.linear3 -> self.bn3 -> self.relu3 ->
           self.fc4 (输出层)
   ```
   - **功能**: 在CLIP特征上的分类头
   - **参数**: 约310K参数

4. **ClipModelat_quantize** - 量化版本
   - 用于模型压缩和通信优化

#### **PromptCLIP.py** (提示学习)

**关键类**:

1. **PromptLearner_client** - 学习文本提示
   - 初始化可学习的上下文向量
   - 支持类别特定的上下文(CSC)

2. **CustomCLIP_client** - CLIP+提示学习
   - 冻结图像编码器
   - 学习文本提示

3. **TextEncoder** - 提示编码器
   - 对提示进行编码

#### **CoCoOpCLIP.py** (CoCoOp方法)
- 上下文条件化提示学习

#### **LinearProbeV2.py** (线性探针v2)
- 线性分类器实现
- 计算类别中心

#### **MLPs.py** (辅助MLP)

```python
class ImageMlp(nn.Module):
    # 图像特征MLP变换
    # fc1(512->512) -> ReLU -> fc2(512->512) -> Tanh

class TextMlp(nn.Module):
    # 文本特征MLP变换
    # 与ImageMlp结构类似
```

---

### 3.3 工具函数文件 (`utils/`)

#### **training.py** (训练逻辑 - 1084行)

**关键函数**:

1. **train()** - 本地客户端训练
   ```python
   def train(args, model, train_loader, optimizer, device, ...):
       - 前向传播: image -> CLIP编码 -> FAM注意力 -> MLP
       - 损失函数: CE + 自适应损失 + 对抗损失
       - 反向传播与优化
   ```

2. **损失函数组合**:
   - CrossEntropyLoss: 基础分类损失
   - LMMDLoss: 本地最大均值差异损失(域适应)
   - CORAL: 相关对齐损失
   - AdversarialLoss: 对抗域适应损失
   - FocalLoss: 处理类别不平衡

3. **辅助函数**:
   - `set_grad()`: 设置可训练参数
   - `set_parameters()`: 更新参数(融合知识)
   - `js_divergence()`: JS散度计算
   - `totrain()`: 切换到训练模式

#### **testing.py** (测试逻辑 - 51KB)

**关键函数**:

1. **test()** - 客户端测试
   - 计算准确率、F1、精度、召回率
   - 支持测试时自适应(TTA)
   - 计算校准度量(NetBenefit, TPR, FPR)

2. **Glotest()** - 全局模型测试
   - 无MLP, 直接使用CLIP+FAM

#### **aggregation.py** (联邦聚合 - 305行)

**关键函数**:

1. **communication()** - 标准聚合
   ```python
   # 根据aggmode选择:
   if aggmode == 'att':  # 只聚合FAM
       server_model.fea_attn = avg(client_fea_attn)
   elif aggmode == 'avg':  # 聚合整个模型
       server_model = avg(client_model)
   ```

2. **communication_quantized()** - 量化聚合
   - 压缩模型参数以减少通信开销
   - 在聚合前后进行压缩/解压

3. **comunicationfedamp()** - FedAMP聚合
   - 自适应聚合率

#### **clip_util.py** (CLIP工具 - 428行)

**关键函数**:

1. **freeze_param()** - 冻结所有参数
2. **get_image_features()** - 提取CLIP图像特征
3. **get_text_features_list()** - 提取CLIP文本特征
4. **compress_model()** - 模型压缩(量化)
5. **decompress_model()** - 模型解压
6. **FocalLossWithSmoothing** - Focal损失实现

#### **prepare_data_dg_clip.py** (数据处理 - 17.5KB)

**关键函数**:

1. **get_data()** - 根据数据集返回数据加载器工厂
2. **数据分割**: 60% 训练, 20% 验证, 20% 测试
3. **支持数据集**:
   - 医学: BraTS, Prostate, brain_ICH, ISIC2019
   - 通用: OfficeHome, DomainNet, miniDomainNet
   - 其他: Food101, DTD等

#### **config.py** (配置 - 187行)

**关键函数**:

1. **img_param_init()** - 初始化参数
   - 设置域/客户端列表
   - 设置类别数量
2. **set_random_seed()** - 设置随机种子

#### **loss_function.py** (损失函数)

```python
class CrossEntropyLabelSmooth: 标签平滑的交叉熵
class MILoss: 互信息损失
class LinearDiscriminantLoss: 线性判别损失
```

---

### 3.4 其他辅助文件

#### **adaptation.py** (域适应 - 254行)

**关键类**:

1. **MMDLoss** - 最大均值差异
   - 高斯核版本(RBF)
   - 线性核版本

2. **LMMDLoss** - 本地MMD
   - 类别级别的权重计算
   - 动态λ调度

3. **AdversarialLoss** - 对抗域适应
   ```python
   class Discriminator(nn.Module):
       # 简单的二分类判别器
       # 用于区分源域和目标域
   ```
   - ReverseLayerF: 梯度反转层

4. **CORAL** - 相关对齐
   - Frobenius范数最小化

#### **GetLog.py** (日志记录)
- 记录训练和测试指标
- 生成CSV输出

#### **sheduler.py** (学习率调度)
```python
class LambdaSheduler:
    # λ(t) = 2/(1+exp(-γp)) - 1
    # 动态加权域适应损失
```

---

## 四、关键算法与创新

### 4.1 特征注意力模块(FAM)

**设计思想**:
```
CLIP图像特征 f -> Linear(f) -> InstanceNorm -> ReLU6 -> 
                   Linear -> Softmax -> α (注意力权重)

加权特征: α ⊙ f (逐元素乘法)
```

**优势**:
- 参数少(~526K vs CLIP的几亿)
- 利用预训练CLIP的知识
- 可适应客户端的特定任务

### 4.2 联邦学习流程

```
for communication_round in range(num_rounds):
    
    # 本地训练
    for each_client in clients:
        for local_epoch in range(wk_iters):
            for batch in train_loader:
                x -> CLIP(冻结) -> FAM(可训) -> MLP(可训) -> output
                loss = CE + λ*LMMDLoss + μ*AdversarialLoss
                backward & update FAM, MLP
    
    # 服务器聚合
    if aggmode == 'att':
        server_fea_attn = avg(client_fea_attn)
        for each_client:
            client_fea_attn = server_fea_attn  # 广播
    
    # 量化通信
    models = compress_model(models)
    communication_quantized(...)
    models = decompress_model(models)
    
    # 验证
    for each_client:
        acc, f1, ... = test(model, val_loader)
    
    # 保存最优模型
    if val_acc > best_acc:
        save_checkpoint()
```

### 4.3 模型量化优化

**量化步骤**:
1. 提取可训练参数(FAM, MLP)
2. 压缩为低精度格式(ZIP压缩)
3. 传输压缩参数
4. 聚合后解压恢复

**通信成本**: 从MB级降低到KB级

---

## 五、数据流概览

```
输入图像 (H×W×3)
    ↓
[CLIP 视觉编码器] (冻结, 权重不更新)
    ↓
图像特征 f (1×512)
    ↓
[FAM 特征注意力模块] (可训练)
    ↓
加权特征 (1×512)
    ↓
[MLP 分类头] (可训练)
    ↓
Logits (1×num_classes)
    ↓
[Softmax]
    ↓
分类概率输出 (1×num_classes)
```

---

## 六、超参数配置

### 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--net` | 'ViT-B/32' | CLIP主干(视觉编码器) |
| `--batch` | 32 | 批大小 |
| `--iters` | 50 | 联邦通信轮数 |
| `--wk_iters` | 1 | 本地训练轮数 |
| `--lr` | 5e-5 | FAM学习率 |
| `--lr_mlp` | 1e-3 | MLP学习率 |
| `--beta1` | 0.9 | Adam β₁ |
| `--beta2` | 0.98 | Adam β₂ |
| `--weight_decay` | 0.02 | L2正则化 |
| `--aggmode` | 'att' | 聚合模式(att/avg) |
| `--test_envs` | [3] | 测试客户端索引 |
| `--seed` | 0 | 随机种子 |

### 学习率调度

```python
# 每个联邦轮后衰减
args.lr *= 0.97
args.lr_mlp *= 0.97
```

---

## 七、支持的数据集

### 医学图像数据集
- **BraTS**: 脑肿瘤分割(2类)
- **Prostate**: 前列腺MRI(2类)
- **brain_ICH**: 颅内出血检测(5类)
- **ISIC2019**: 皮肤病变分类(6类)

### 通用域泛化数据集
- **OfficeHome**: 办公用品(65类)
- **DomainNet**: 绘图-照片转移(345类)
- **miniDomainNet**: 小规模DomainNet(65类)
- **Food101**: 食物分类(101类)
- **DTD**: 纹理描述(47类)

### 数据组织结构

```
data/
├─ BraTS/
│  ├─ client_0/
│  │  ├─ class_1/
│  │  │  ├─ image_001.jpg
│  │  │  └─ ...
│  │  └─ class_2/
│  ├─ client_1/
│  └─ Global/
├─ Prostate/
│  ├─ Center_1/
│  ├─ Center_2/
│  └─ ...
└─ ...
```

---

## 八、方法对比总结

### 各方法的关键区别

| 方法 | CLIP冻结 | 可训模块 | 聚合模式 | 通信成本 |
|------|---------|---------|---------|---------|
| **Ours** | ✓ | FAM + MLP | att | 极低(量化) |
| **FedAVG** | ✗ | 全部 | avg | 高 |
| **FedCLIP** | ✓ | FAM only | att | 中等 |
| **LoRA** | ✓ | LoRA | avg | 低 |
| **LP++** | ✓ | 线性分类头 | avg | 极低 |
| **PromptFL** | ✓ | 文本提示 | att | 极低 |
| **CoCoOp** | ✓ | 条件提示 | att | 极低 |

---

## 九、核心创新点

### 1. **特征注意力模块(FAM)**
   - 轻量级参数(只需526K)
   - 保留CLIP预训练知识
   - 适应客户端异构数据

### 2. **分布式+量化**
   - 模型压缩: 参数→低精度→ZIP压缩
   - 减少通信开销
   - 保持精度损失最小

### 3. **多损失函数组合**
   - CE: 基础分类
   - LMMD: 类级域对齐
   - Adversarial: 特征级对齐
   - 动态权重调度

### 4. **客户端特定的MLP**
   - 允许数据异构(不同类别)
   - 自适应分类头
   - 通过参数重用融合全局知识

---

## 十、执行流程

### 快速开始

```bash
# 基础配置初始化
python Ours.py

# 或指定参数
python Ours.py --dataset BraTS --net ViT-B/32 --iters 50 --wk_iters 1

# 其他方法
python DN-FedAVG-1.py      # 标准FedAVG
python DN-FedCLIP-1.py     # 仅FAM
python LP++.py             # 线性探针
python LoRA_OF-2.py        # LoRA
```

### 输出结果

```
SavedModel/
├─ OfficeHome/AAAI2026/ACPR/
│  └─ Fed_attnMLP/ViT-B-32/
│     ├─ Client0.pth          # 客户端FAM参数
│     ├─ mlp0.pth             # 客户端MLP参数
│     └─ Client_quantized_0.pth

results/
├─ /OfficeHome/AAAI2026/ACPR/
│  ├─ ValidationMetrics(*.csv)  # 验证指标
│  ├─ TestingMetrics(*.csv)     # 测试指标
│  ├─ NetBenefits(*.csv)        # 校准度量
│  ├─ TPR(*.csv)                # 真正率
│  └─ FPR(*.csv)                # 假正率
```

---

## 十一、关键类关系图

```
nn.Module (PyTorch基类)
    ├─ ClipModelat ───────┬─ model (CLIP视觉编码器)
    │                     ├─ fea_attn (FAM)
    │                     └─ labels (类别)
    │
    ├─ MLP ────────────────── 4层全连接 + BN + ReLU
    │
    ├─ ImageMlp ──────────── MLP变换
    │
    ├─ MMDLoss ──────────┬─ 高斯核
    │                    └─ 线性mmd2
    │
    ├─ LMMDLoss ─────────── 类级权重MMD
    │
    ├─ AdversarialLoss ──┬─ Discriminator
    │                    └─ ReverseLayerF
    │
    ├─ FocalLoss ─────────── 样本级权重CE
    │
    └─ TextEncoder ────────── 提示编码
```

---

## 十二、性能指标

### 输出度量

1. **分类指标**
   - Accuracy: 准确率
   - Balanced Accuracy: 平衡准确率(处理不平衡)
   - F1 Score: F1分数
   - Precision: 精度
   - Recall: 召回率

2. **校准指标** (医学图像应用)
   - Net Benefit: 临床净获益
   - TPR/FPR: 真阳性率/假阳性率
   - Reliability Diagram: 校准图

3. **统计检验**
   - P-Value: 方法间统计显著性

---

## 总结

这是一个完整的**联邦学习+CLIP微调**框架，设计用于医学图像分类等资源受限场景。核心创新包括：

1. **轻量级FAM模块**: 参数少但有效
2. **量化通信**: 大幅降低通信成本
3. **多域自适应**: LMMD + 对抗损失
4. **灵活的聚合**: 支持多种参数共享策略
5. **严格的医学评估**: 校准度量和统计检验

框架支持多个基准和提出方法的对比，为联邦学习在医学领域的应用提供了完整的工程实现。
