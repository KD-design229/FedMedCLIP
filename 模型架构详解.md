# 模型架构详解

## 一、核心模型组件

### 1.1 CLIP 视觉编码器 (冻结)

```python
class ClipModelat(nn.Module):
    """
    CLIP模型包装类,包含冻结的视觉编码器和可训练的特征注意力模块
    """
    
    def __init__(self, model_name='ViT-B/32', attention=True, freezepy=True):
        """
        参数:
        - model_name: CLIP预训练模型名称
          支持: RN50, RN101, RN50x4, ViT-B/32(推荐), ViT-B/16, ViT-L/14, ViT-L/14@336px
        - attention: 是否包含FAM模块 (True)
        - freezepy: 是否冻结CLIP参数 (True)
        """
```

**支持的模型架构**:

```
Vision Encoders (冻结的预训练权重)
├─ ResNet系列
│  ├─ RN50      (50层残差网络)
│  ├─ RN101     (101层残差网络)
│  ├─ RN50x4    (宽度4倍)
│  ├─ RN50x16   (宽度16倍)
│  └─ RN50x64   (宽度64倍)
│
└─ Vision Transformer系列 (推荐)
   ├─ ViT-B/32  (大补丁,适合有限资源)
   ├─ ViT-B/16  (中补丁,平衡精度和速度)
   └─ ViT-L/14  (大模型,最高精度)
```

**输出特征**:
- 维度: 512 (对所有模型标准化为512维)
- 类型: float32 或 float16
- 归一化: L2归一化 (||f|| = 1)

### 1.2 特征注意力模块 (FAM) - 核心创新

```python
# 初始化代码 (models.py, Line 58-66)
self.fea_attn = nn.Sequential(
    nn.Linear(512, 512),                  # 第一线性变换
    nn.InstanceNorm1d(512),               # 实例归一化(保留个体差异)
    nn.ReLU6(),                           # ReLU6激活(限制范围)
    nn.Linear(512, 512),                  # 第二线性变换
    nn.Softmax(dim=1)                     # Softmax注意力权重
)
```

**模块流程**:

```
输入特征 f ∈ ℝ^(B×512)
    ↓
线性1: W₁f + b₁ ∈ ℝ^(B×512)
    ↓
InstanceNorm: (f - μ)/σ  (归一化)
    ↓
ReLU6: min(max(f, 0), 6)  (限制在[0,6])
    ↓
线性2: W₂f + b₂ ∈ ℝ^(B×512)
    ↓
Softmax: α = exp(f)/Σexp(f) ∈ ℝ^(B×512)  (注意力权重)
    ↓
输出: α ⊙ f ∈ ℝ^(B×512)  (加权特征)
```

**参数统计**:
```
第一线性层: 512×512 + 512 = 262,656
第二线性层: 512×512 + 512 = 262,656
总计: ~525K参数 (相比CLIP的338M参数,只需0.16%)
```

**设计的好处**:

1. **参数高效**: 极少的可训练参数
2. **保留知识**: 冻结CLIP,利用预训练知识
3. **自适应**: 每个客户端学习自己的注意力
4. **稳定性**: InstanceNorm避免批次间的统计偏差

### 1.3 MLP 分类头

```python
class MLP(nn.Module):
    """
    多层感知机分类头
    输入: CLIP特征 (512维)
    输出: 类别logits
    """
    
    def __init__(self, input_size=512, hidden_size=512, num_classes):
        super(MLP, self).__init__()
        
        # 第一层
        self.linear1 = MaskedMLP(512, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        
        # 第二层
        self.linear2 = MaskedMLP(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.ReLU()
        
        # 第三层
        self.linear3 = MaskedMLP(512, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.relu3 = nn.ReLU()
        
        # 输出层
        self.fc4 = MaskedMLP(512, num_classes)
    
    def forward(self, x):
        # x: (batch_size, 512)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.linear3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.fc4(x)  # (batch_size, num_classes)
        return x
```

**参数配置** (以OfficeHome为例, num_classes=65):

```
Layer 1: 512×512 + 512 = 262,656
Layer 2: 512×512 + 512 = 262,656
Layer 3: 512×512 + 512 = 262,656
Layer 4: 512×65 + 65 = 33,345
总计: ~821K参数

加BatchNorm参数:
BN1: 512×2 = 1,024
BN2: 512×2 = 1,024
BN3: 512×2 = 1,024
总计: 3,072

MLP总参数: ~824K
```

---

## 二、完整前向传播

### 2.1 单样本前向传播

```
输入: 彩色医学图像 (3×224×224)
    ↓
[CLIP 图像编码器 - 冻结]
    预处理 (缩放,中心裁剪,归一化)
    传递通过卷积层/Transformer层
    ↓
图像特征 f_img ∈ ℝ^(1×512) (L2归一化)
    ↓
[FAM - 特征注意力模块 - 可训练]
    α = Softmax(Linear2(ReLU6(InstanceNorm(Linear1(f_img)))))
    f_att = α ⊙ f_img
    ↓
加权特征 f_att ∈ ℝ^(1×512)
    ↓
[MLP 分类头 - 可训练]
    x1 = ReLU(BN1(Linear1(f_att)))
    x2 = ReLU(BN2(Linear2(x1)))
    x3 = ReLU(BN3(Linear3(x2)))
    logits = Linear4(x3)
    ↓
Logits ∈ ℝ^(1×num_classes)
    ↓
[Softmax]
    概率分布 p ∈ ℝ^(1×num_classes)
    ↓
预测类别: argmax(p)
```

### 2.2 批处理前向传播

```
输入批次: (batch_size×3×224×224)
    ↓
[CLIP Vision Encoder]
    批处理通过预训练编码器
    输出: (batch_size×512)
    ↓
[FAM]
    逐样本注意力加权
    输出: (batch_size×512)
    ↓
[MLP]
    批处理通过4层FC
    每层后跟BN和ReLU
    ↓
输出Logits: (batch_size×num_classes)
```

---

## 三、损失函数架构

### 3.1 多损失加权组合

```
总损失 = L_ce + λ₁·L_lmmd + λ₂·L_adv + λ₃·L_reg

其中:
```

**1. 交叉熵损失 (CrossEntropy) - 主要损失**

```python
L_ce = CrossEntropyLoss(logits, labels)
# 或带标签平滑的版本:
L_ce = CrossEntropyLabelSmooth(logits, labels, epsilon=0.1)
```

作用: 基础分类监督信号

**2. 本地MMD损失 (LMMDLoss) - 域自适应**

```python
L_lmmd = LMMDLoss(source_features, target_logits, 
                  source_labels, num_classes)

# 计算过程:
# 1. 构建类级权重矩阵 weight_ss, weight_tt, weight_st
# 2. 计算高斯核矩阵 K
# 3. L = trace(weight_ss·K_ss + weight_tt·K_tt - 2·weight_st·K_st)
# 4. 动态加权: L = λ(t)·L (λ从0增加到1)
```

**权重计算**:
```
source_label ──→ one-hot ──→ 行归一化 ──→ weight_ss
target_logit ──→ softmax ──→ 行归一化 ──→ weight_tt
```

作用: 对齐源域(训练)和目标域(客户端特定)的特征分布

**3. 对抗损失 (AdversarialLoss) - 特征级适应**

```python
class Discriminator(nn.Module):
    # 4层: Linear(512→512) + BN + ReLU
    #      Linear(512→512) + BN + ReLU
    #      Linear(512→1) + Sigmoid
    
    def forward(self, x):
        # 输出: P(域 = 源域) ∈ [0,1]

# 对抗损失:
# L_adv = BCE(D(f_s), 1) + BCE(D(f_t), 0)
# 其中f_s是源域特征, f_t是目标域特征
```

梯度反转:
```
前向: x → ReverseLayerF(x, α) → x (不变)
反向: grad ← -α·grad (反转)
```

作用: 迫使编码器学习域不变特征

**4. 正则化损失**

```python
L_reg = weight_decay · ||θ||₂
```

---

## 四、聚合机制

### 4.1 注意力聚合 (att)

```
用于: Ours, FedCLIP, PromptFL
聚合对象: 只有FAM模块

server_fea_attn_new = Σ(client_weight_i × client_fea_attn_i)

其中client_weight_i = 1/num_training_clients
```

流程:
```
客户端0的FAM参数 ──┐
客户端1的FAM参数 ──┼─→ [平均] ──→ 服务器FAM参数 ──┐
客户端2的FAM参数 ──┘                              │
                                                   ├─→ 广播到所有客户端
                                               │   │
                                               └───┘
```

**优势**:
- 通信量少 (仅~524K参数)
- 保留CLIP预训练
- 只学习任务特定的特征变换

### 4.2 平均聚合 (avg)

```
用于: FedAVG, LoRA, LP++
聚合对象: 整个模型参数

server_param_new = Σ(client_weight_i × client_param_i)
```

包括:
- CLIP所有权重
- FAM权重 (如果存在)
- MLP权重 (不聚合, 保持客户端特定)

**问题**:
- 通信量大 (CLIP: 338M+)
- 破坏预训练知识
- 慢收敛

### 4.3 量化聚合 (communication_quantized)

```
通信优化步骤:

客户端端:
  1. 提取可训参数 params
  2. 转为float32 convert_to_fp32(params)
  3. 展平: params_flat = flatten(params)
  4. 序列化: bytes = serialize(params_flat)
  5. 压缩: compressed = zlib.compress(bytes)
     → 通常50-90%压缩率
  
服务器端:
  6. 解压: bytes = zlib.decompress(compressed)
  7. 反序列化: params_flat = deserialize(bytes)
  8. 聚合: avg_params = avg(all_params)
  9. 广播: 发送给所有客户端
  
客户端端:
  10. 接收: compressed_params
  11. 解压→反序列化
  12. 恢复原始形状
```

**通信成本对比**:

```
方法           可训参数量    未压缩大小    压缩后大小    压缩率
─────────────────────────────────────────────────────
Ours (FAM)     ~524K        2.1 MB        0.2-0.4 MB   85-90%
FedCLIP        ~524K        2.1 MB        0.2-0.4 MB   85-90%
LoRA           ~100K        0.4 MB        0.05 MB      88%
LP++           ~33K         0.2 MB        0.02 MB      90%
FedAVG         ~338M        1.4 GB        0.5-1 GB     50-65%
```

---

## 五、完整管道示意图

### 5.1 联邦学习管道

```
    ┌──────────────────────────────────────────────────────────────┐
    │                    Fed Round t                                │
    └──────────────────────────────────────────────────────────────┘
              │
              ├──────────────────────┬──────────────────────┐
              │                      │                      │
          ┌───▼────┐            ┌────▼────┐          ┌─────▼────┐
          │Client 0 │            │ Client 1 │          │ Client N │
          └────┬────┘            └────┬────┘          └─────┬────┘
               │                      │                      │
               ├─ 本地训练 (wi, wk_iters)
               │  for local_epoch:
               │    for batch in train_loader:
               │      forward: x → CLIP → FAM → MLP → logits
               │      loss = CE + LMMD + Adv
               │      backward & optimize FAM, MLP
               │  
               ├─ 客户端更新完成 ✓
               │
               └──────────────────┬──────────────────────┘
                                  │
                          ┌───────▼────────┐
                          │ 聚合步骤        │
                          ├─────────────────┤
                          │ 1. 压缩参数     │
                          │ 2. 上传FAM      │
                          │ 3. 服务器聚合   │
                          │ 4. 广播新FAM    │
                          │ 5. 解压参数     │
                          └─────────┬──────┘
                                    │
                        ┌───────────▼────────────┐
                        │  验证 (Val Round t)     │
                        │                        │
                        │ for each client:       │
                        │   acc = test(client)   │
                        │   if acc > best:       │
                        │     save_checkpoint()  │
                        │                        │
                        └────────────────────────┘
                                    │
                        if t < num_rounds:
                                    │
                        ┌───────────▼────────────┐
                        │  Fed Round t+1         │
                        └────────────────────────┘
                                    │
                                    └─→ [返回到本地训练]
```

### 5.2 批处理数据流

```
┌────────────────────────────────────────────────────────────┐
│           一个批次的完整处理流程                              │
└────────────────────────────────────────────────────────────┘

输入批次: (32×3×224×224) [32张图像]

        ↓

[CLIP 图像编码器 - 冻制·无梯度]
    - 预处理 (缩放, 中心裁剪, 归一化)
    - 卷积特征提取 (ResNet或ViT)
    - 全局平均池化或取特殊token
    - L2 归一化
        ↓
    图像特征: (32×512)

        ↓

[FAM 注意力模块 - 梯度计算]
    - Linear1: (32×512) → (32×512)
    - InstanceNorm1d: 独立归一化每个样本
    - ReLU6: 激活
    - Linear2: (32×512) → (32×512)
    - Softmax: 生成注意力权重
    - 加权求和: (32×512) ⊙ (32×512)
        ↓
    加权特征: (32×512)

        ↓

[MLP 分类头 - 梯度计算]
    Layer 1:
      - Linear: (32×512) @ (512×512)ᵀ + bias → (32×512)
      - BatchNorm1d: 批内统计
      - ReLU: 激活
    
    Layer 2:
      - Linear: (32×512) @ (512×512)ᵀ + bias → (32×512)
      - BatchNorm1d
      - ReLU
    
    Layer 3:
      - Linear: (32×512) @ (512×512)ᵀ + bias → (32×512)
      - BatchNorm1d
      - ReLU
    
    Output Layer:
      - Linear: (32×512) @ (512×num_classes)ᵀ + bias → (32×num_classes)
        ↓
    Logits: (32×num_classes)

        ↓

[损失计算]
    - CE损失: CrossEntropyLoss(logits, labels) → scalar
    - LMMD损失: class-weighted MMD
    - 对抗损失: 通过Discriminator
    - 总损失 = L_ce + λ₁·L_lmmd + λ₂·L_adv
        ↓
    Loss: scalar (反向传播的基点)

        ↓

[反向传播]
    梯度流向:
    - FAM 参数: ∂Loss/∂FAM (梯度 ≠ 0)
    - MLP 参数: ∂Loss/∂MLP (梯度 ≠ 0)
    - CLIP 参数: ∂Loss/∂CLIP (梯度 = 0, 冻结)

        ↓

[优化步骤]
    使用 AdamW 优化器:
    - 学习率 lr_fam = 5e-5   (对FAM)
    - 学习率 lr_mlp = 1e-3   (对MLP)
    - β₁ = 0.9, β₂ = 0.98
    - weight_decay = 0.02
    
    更新:
    m_t = β₁·m_{t-1} + (1-β₁)·g_t
    v_t = β₂·v_{t-1} + (1-β₂)·g_t²
    θ_t = θ_{t-1} - lr·m_t/(√v_t + ε) - wd·θ_{t-1}
        ↓
    参数更新完成
```

---

## 六、模型配置示例

### 6.1 ViT-B/32 配置 (推荐)

```python
args.net = 'ViT-B/32'

CLIP配置:
├─ 视觉编码器
│  ├─ 补丁大小: 32×32
│  ├─ 图像分辨率: 224×224
│  ├─ 补丁数: 49 (7×7)
│  ├─ 特征维度: 768
│  └─ 转换器层数: 12
│
├─ 特征池化后维度: 512 (标准化)
└─ 参数总数: 338M

FAM配置:
├─ 输入维度: 512
├─ 隐层维度: 512
├─ 输出维度: 512
└─ 参数总数: 525K (0.16% vs CLIP)

MLP配置:
├─ 隐层大小: 512 (x3)
├─ 输出维度: num_classes
└─ 参数总数: 824K (若num_classes=65)

总参数:
├─ CLIP: 338M (冻结)
├─ FAM: 525K (可训)
├─ MLP: 824K (可训)
└─ 合计可训: 1.3M (0.38%)
```

### 6.2 ResNet50 配置 (备选)

```python
args.net = 'RN50'

CLIP配置:
├─ 视觉编码器: ResNet50
├─ 残差层数: 50
├─ 特征维度: 512
└─ 参数总数: 102M

FAM配置:
├─ 相同配置
└─ 参数总数: 525K

MLP配置:
├─ 相同配置
└─ 参数总数: 824K
```

---

## 七、内存占用分析

### 运行时内存占用

```
推理 (Inference):

CLIP模型权重:
├─ ViT-B/32: 338M × 4字节 = 1.35 GB
├─ 激活值缓存: ~200 MB
└─ 小计: 1.6 GB

FAM模块:
├─ 权重: 525K × 4字节 = 2.1 MB
├─ 激活值: (batch×512) × 4字节
│  ├─ 批大小=32: 64 KB
│  ├─ 批大小=256: 512 KB
│  └─ 批大小=1024: 2 MB
└─ 小计: ~3 MB (batch=32)

MLP模块:
├─ 权重: 824K × 4字节 = 3.3 MB
├─ 激活值: ~2 MB (batch=32)
└─ 小计: ~5 MB

推理总计: ~1.6 GB


训练 (Training):

CLIP模型:
├─ 权重: 1.35 GB
├─ 梯度: 0 (冻结)
└─ 小计: 1.35 GB

FAM + MLP:
├─ 权重: 5.4 MB
├─ 梯度: 5.4 MB
├─ 优化器状态 (AdamW): 10.8 MB
├─ 激活值梯度缓存: ~50 MB (batch=32)
└─ 小计: ~72 MB

损失函数:
├─ Discriminator: ~4 MB
├─ 临时张量: ~20 MB
└─ 小计: ~24 MB

训练总计: ~1.45 GB (batch=32)

with gradient_checkpointing:
└─ 训练总计: ~1.35 GB (节省激活缓存)
```

---

## 八、计算复杂度

### 时间复杂度

```
单个样本推理:

CLIP编码: O(H×W×C) = O(224×224×3×L)
         L是模型深度, 主要是卷积和变压器
         ~ 20-30 FLOPs (前向传播)

FAM计算: 2×线性(512×512) + norm + activation
        ~ 1M FLOPs

MLP计算: 3×线性(512×512) + 1×线性(512×num_classes)
        ~ 2-3M FLOPs

总计: ~25-35M FLOPs per 224×224 image


批处理 (batch_size=32):

CLIP编码: 32×25M = 800M FLOPs
FAM: 32×1M = 32M FLOPs
MLP: 32×2.5M = 80M FLOPs

总计: ~910M FLOPs

在V100 GPU上:
├─ 理论峰值: 130 TFLOPS (FP32) / 260 TFLOPS (混合精度)
├─ 实际吞吐: 70-100 TFLOPS (考虑内存带宽)
├─ 推理延迟: ~10-15 ms per batch
└─ 吞吐: ~2000-3000 images/sec
```

### 空间复杂度

```
模型大小: O(CLIP参数 + FAM参数 + MLP参数)
        = O(338M + 525K + 824K)
        ≈ 338M 参数

可训参数大小: O(525K + 824K) = O(1.3M)

批处理激活: O(batch_size × feature_dim × layers)
          = O(32 × 512 × 15) ≈ 250MB
```

---

## 总结

该架构设计的核心特点:

1. **轻量级**: 仅添加1.3M可训参数到338M CLIP
2. **高效**: 通过量化通信减少90%通信开销
3. **有效**: 多损失函数融合域适应和对抗学习
4. **灵活**: 支持多种聚合和微调策略
5. **可扩展**: 易于适配不同数据集和客户端设置

关键创新是FAM模块,它以极少参数代价实现了有效的特征自适应,同时保留了预训练CLIP的强大泛化能力。
