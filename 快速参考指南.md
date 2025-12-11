# 快速参考指南

## 快速启动

### 运行主方法 (Ours)
```bash
python Ours.py
```

### 运行其他方法
```bash
python DN-FedAVG-1.py      # 标准FedAVG
python DN-FedCLIP-1.py     # 仅FAM
python LP++.py             # 线性探针
python LoRA_OF-2.py        # LoRA
python DN-Prompt.py        # 提示学习
```

### 指定参数运行
```bash
python Ours.py --dataset BraTS --net ViT-B/32 --iters 50 --batch 32
python Ours.py --dataset Prostate --lr 5e-5 --lr_mlp 1e-3 --test_envs 0
```

---

## 关键参数速查

| 参数 | 默认 | 范围 | 说明 |
|------|------|------|------|
| `--dataset` | OfficeHome | 见下表 | 数据集 |
| `--batch` | 32 | 16-128 | 批大小 |
| `--iters` | 50 | 10-200 | 联邦轮数 |
| `--wk_iters` | 1 | 1-5 | 本地轮数 |
| `--net` | ViT-B/32 | 见下表 | CLIP模型 |
| `--lr` | 5e-4 | 1e-5~1e-3 | FAM学习率 |
| `--lr_mlp` | 1e-4 | 1e-4~1e-2 | MLP学习率 |
| `--seed` | 0 | 0-999 | 随机种子 |
| `--aggmode` | att | att/avg | 聚合方式 |
| `--test_envs` | [3] | 0-N | 测试客户端 |
| `--weight_decay` | 0.02 | 0-0.1 | L2正则化 |

---

## 支持的数据集

### 医学数据集
```
BraTS           # 脑肿瘤 (2类) ✓ 推荐
Prostate        # 前列腺 (2类) ✓
brain_ICH       # 颅内出血 (5类) ✓
ISIC2019        # 皮肤病变 (6类)
```

### 通用数据集
```
OfficeHome      # 办公用品 (65类) ✓ 推荐
DomainNet       # 绘图转照片 (345类)
miniDomainNet   # 小DomainNet (65类)
Food101         # 食物分类 (101类)
```

---

## CLIP 模型对比

| 模型 | 参数 | 速度 | 精度 | 推荐 |
|------|------|------|------|------|
| RN50 | 102M | 快 | 中 | - |
| RN101 | 170M | 中 | 中 | - |
| ViT-B/32 | 338M | 快 | 高 | ✓ |
| ViT-B/16 | 338M | 中 | 高 | ✓ |
| ViT-L/14 | 427M | 慢 | 最高 | - |

```python
# 使用方式
--net ViT-B/32      # 推荐,平衡精度和速度
--net RN50          # 资源有限时选用
--net ViT-L/14      # 需要最高精度
```

---

## 方法对比速查

### Ours (推荐)
- **特点**: FAM + MLP
- **参数**: 1.3M可训 / 338M冻结
- **通信**: 极低 (量化)
- **命令**: `python Ours.py --aggmode att --method attn_mlp`

### FedAVG
- **特点**: 更新全部参数
- **参数**: 338M可训
- **通信**: 高
- **命令**: `python DN-FedAVG-1.py --aggmode avg`

### FedCLIP
- **特点**: 仅FAM
- **参数**: 525K可训
- **通信**: 低
- **命令**: `python DN-FedCLIP-1.py --aggmode att`

### LP++
- **特点**: 线性分类头
- **参数**: 33K可训
- **通信**: 极低
- **命令**: `python LP++.py`

### LoRA
- **特点**: 低秩分解
- **参数**: 100K可训
- **通信**: 极低
- **命令**: `python LoRA_OF-2.py`

---

## 数据准备

### 数据结构
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
│
├─ OfficeHome/
│  ├─ A/           # Art
│  ├─ C/           # Clipart
│  ├─ P/           # Product
│  └─ R/           # RealWorld
```

### 数据分割
- 60% 训练
- 20% 验证
- 20% 测试

修改比例:
```python
# utils/prepare_data_dg_clip.py, Line 263
l1, l2, l3 = int(l*0.6), int(l*0.2), int(l*0.2)
```

---

## 模型存储路径

### 默认保存位置
```
SavedModel/
├─ OfficeHome/AAAI2026/ACPR/
│  └─ Fed_attnMLP/ViT-B-32/
│     ├─ Client0.pth           # FAM权重
│     ├─ mlp0.pth              # MLP权重
│     └─ Client_quantized_0.pth # 量化模型

results/
├─ /OfficeHome/AAAI2026/ACPR/
│  ├─ ValidationMetrics(*.csv)
│  ├─ TestingMetrics(*.csv)
│  ├─ NetBenefits(*.csv)
│  ├─ TPR(*.csv)
│  └─ FPR(*.csv)
```

### 自定义保存路径
```python
# Ours.py, Line 296-298
Location = './SavedModel/YourDataset/YourMethod/'
method = '/YourMethod/'
dataset = '/YourDataset/'
main(Location, dataset, method)
```

---

## 关键函数查找

### 模型相关
```
ClipModelat          → nets/models.py Line 82
MLP                  → nets/models.py Line 225
FAM初始化            → nets/models.py Line 58-66
```

### 训练相关
```
train()              → utils/training.py Line ?
test()               → utils/testing.py Line ?
Glotest()            → utils/testing.py Line ?
```

### 聚合相关
```
communication()      → utils/aggregation.py Line 14
communication_quantized() → utils/aggregation.py Line ?
compress_model()     → utils/clip_util.py Line ?
decompress_model()   → utils/clip_util.py Line ?
```

### 损失函数相关
```
LMMDLoss             → adaptation.py Line 55
AdversarialLoss      → adaptation.py Line 172
Discriminator        → adaptation.py Line 219
CORAL                → adaptation.py Line 240
FocalLoss            → utils/clip_util.py Line ?
```

---

## 常见问题快速解答

### Q: 如何只在GPU上运行?
```python
# Ours.py, Line 25
device = torch.device('cuda')
```

### Q: 如何改变冻结策略?
```python
# nets/models.py, Line 96
freezepy=False  # 不冻结CLIP

# utils/clip_util.py, Line 28-32
def freeze_param(model):
    for name, param in model.named_parameters():
        param.requires_grad = False  # 改为True解冻
```

### Q: 如何改变聚合策略?
```bash
# 聚合FAM (PEFT方法)
--aggmode att

# 聚合全部 (FedAVG)
--aggmode avg
```

### Q: 如何查看具体的损失值?
```python
# utils/training.py
print(f"Loss CE: {loss_ce.item()}")
print(f"Loss LMMD: {loss_lmmd.item()}")
print(f"Loss Adv: {loss_adv.item()}")
```

### Q: 如何改变MLP结构?
```python
# nets/models.py, Line 225-240
class MLP(nn.Module):
    def __init__(self, input_size=512, hidden_size=512, num_classes=65):
        # 修改hidden_size或层数
        self.linear1 = ...
        self.linear2 = ...
        # 添加或删除隐层
```

### Q: 如何支持新数据集?
```python
# utils/config.py
def img_param_init(args):
    if args.dataset == 'MyDataset':
        args.domains = ['domain1', 'domain2', ...]
        args.num_classes = 10

# utils/prepare_data_dg_clip.py
def prepare_mydataset_data(args, model):
    # 加载数据逻辑
    return train_loaders, val_loaders, test_loaders, ...
```

---

## 性能优化建议

### 内存优化
```bash
--batch 16              # 减小批大小
# 启用梯度检查点 (代码修改)
torch.utils.checkpoint.checkpoint(...)
```

### 速度优化
```bash
--net ViT-B/32          # 选择较小模型
--wk_iters 1            # 减少本地轮数
# 使用混合精度
torch.cuda.amp
```

### 精度优化
```bash
--net ViT-L/14          # 使用更大模型
--iters 100             # 增加通信轮数
--wk_iters 5            # 增加本地轮数
--lr 5e-5               # 降低学习率
```

---

## 输出文件说明

### CSV文件内容

**ValidationMetrics(i).csv**
```
Acc    | BalAcc | F1    | Precision | Recall
--------|--------|-------|-----------|--------
0.8543 | 0.8402 | 0.8451| 0.8520    | 0.8451
0.8612 | 0.8481 | 0.8520| 0.8589    | 0.8520
...
```

**TestingMetrics(i).csv**
```
Acc    | BalAcc | F1    | Precision | Recall
```

**NetBenefits(i).csv** (医学指标)
```
Threshold | Net_Benefit
----------|-------------
0.1       | 0.1234
0.2       | 0.1456
...
```

**TPR(i).csv** / **FPR(i).csv** (ROC曲线)
```
Threshold | TPR   | FPR
----------|-------|------
0.1       | 0.95  | 0.05
0.2       | 0.92  | 0.08
...
```

---

## 日志输出说明

### 训练日志
```
============ Train epoch 0 ============
(每个本地训练轮)

Epoch: 0 | Loss: 1.2345 | Acc: 0.7523
Epoch: 1 | Loss: 1.1234 | Acc: 0.7634
```

### 验证日志
```
Test site-0| Validation Acc: 0.8543 | Bacc: 0.8402
Test site-1| Validation Acc: 0.8612 | Bacc: 0.8481
```

### 测试日志
```
Test site-0| Test Acc: 0.8543 | Bacc: 0.8402 | F1: 0.8451
Test site-3| Test Acc: 0.8412 | Bacc: 0.8234 | F1: 0.8301
```

---

## 代码结构速览

```
项目根目录/
├─ Ours.py              主方法 ✓
├─ DN-FedAVG-1.py       FedAVG基准
├─ DN-FedCLIP-1.py      FedCLIP对比
├─ LP++.py              线性探针
├─ LoRA_OF-2.py         LoRA方法
├─ DN-Prompt.py         提示学习
├─ DN-Zero-Shot-1.py    零样本
│
├─ nets/
│  ├─ models.py         CLIP + FAM + MLP ✓
│  ├─ PromptCLIP.py     提示模块
│  ├─ LinearProbeV2.py  线性分类器
│  ├─ CoCoOpCLIP.py     CoCoOp模块
│  └─ MLPs.py           辅助MLP
│
├─ utils/
│  ├─ training.py       本地训练 ✓
│  ├─ testing.py        测试评估 ✓
│  ├─ aggregation.py    参数聚合 ✓
│  ├─ clip_util.py      CLIP工具 ✓
│  ├─ prepare_data_dg_clip.py  数据加载 ✓
│  ├─ config.py         参数初始化 ✓
│  └─ loss_function.py  损失函数
│
├─ adaptation.py        域适应 ✓
├─ GetLog.py            日志记录
├─ sheduler.py          学习率调度
├─ data/                数据目录
├─ SavedModel/          模型保存
├─ results/             结果输出
└─ README.md            说明文档
```

---

## 精要总结

### 项目核心
- **目标**: 联邦学习 + 医学图像分类
- **方法**: CLIP + FAM + MLP
- **创新**: 轻量级参数高效微调 + 量化通信

### 关键部分
1. **ClipModelat**: CLIP包装 (冻结) + FAM (可训)
2. **MLP**: 分类头 (可训)
3. **train()**: 本地训练 (CE + LMMD + Adv)
4. **communication()**: 参数聚合 (att/avg)
5. **test()**: 评估 (Acc/F1/校准)

### 快速修改
- 改方法: 换`--aggmode` 和 `--method`
- 改模型: 换`--net` 和 `--dataset`
- 改参数: 改学习率 `--lr`, `--lr_mlp`
- 改数据: 修改`prepare_data_dg_clip.py`

### 常用命令
```bash
# 标准运行
python Ours.py

# 改参数
python Ours.py --dataset BraTS --lr 1e-4 --batch 64

# 改方法
python DN-FedAVG-1.py --dataset Prostate

# 改模型
python Ours.py --net ViT-L/14 --dataset OfficeHome
```
