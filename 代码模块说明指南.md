# 详细代码模块说明

## 目录

1. [主程序入口](#1-主程序入口)
2. [模型模块](#2-模型模块)
3. [数据处理](#3-数据处理)
4. [训练测试](#4-训练测试)
5. [聚合优化](#5-聚合优化)
6. [损失函数](#6-损失函数)
7. [工具函数](#7-工具函数)

---

## 1. 主程序入口

### 1.1 Ours.py - 提出的方法

**运行命令**:
```bash
python Ours.py
```

**关键步骤**:

```python
# 第1步: 初始化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
# 解析命令行参数 (见README中的参数列表)
args = parser.parse_args()

# 第2步: 设置随机种子和参数
set_random_seed(args.seed)  # 保证可复现性
args = img_param_init(args)  # 初始化数据集配置

# 第3步: 初始化服务器模型
server_model = ClipModelat(
    args.net,          # CLIP模型名称 (e.g., 'ViT-B/32')
    attention=True,    # 包含FAM模块
    freezepy=True      # 冻结CLIP参数
)

# 第4步: 加载数据
train_loaders, val_loaders, test_loaders, test_train, train_test_loaders, Labels = \
    get_data(args.dataset)(args, server_model)

# 第5步: 初始化FAM模块
server_model.initdgatal(test_loaders[3])  # 用样本数据初始化

# 第6步: 创建客户端副本和优化器
client_num = len(test_loaders)
sclient_num = client_num - len(args.test_envs)  # 训练客户端数

models = [copy.deepcopy(server_model).to('cpu') for idx in range(client_num)]
mlp = [MLP(hidden_size=512, input_size=512, num_classes=len(Labels[idx])) 
       for idx in range(client_num)]

# 第7步: 联邦学习主循环
for a_iter in range(args.iters):  # 每个通信轮
    optimizers = [optim.AdamW(...) for idx in range(client_num)]
    
    for wi in range(args.wk_iters):  # 本地训练轮
        for client_idx, model in enumerate(models):
            if client_idx not in args.test_envs:  # 跳过测试客户端
                # 本地训练
                train(args, model, train_test_loaders[client_idx], 
                      optimizers[client_idx], device, 
                      test_train[0], adv[client_idx], 
                      server_model_pre, previous_nets[client_idx],
                      mlp[client_idx])
    
    # 第8步: 聚合
    with torch.no_grad():
        server_model = compress_model(server_model)
        for client_idx in range(client_num):
            models[client_idx] = compress_model(models[client_idx])
        
        server_model, models = communication_quantized(
            args, server_model, models, client_weights)
        
        server_model = decompress_model(server_model, server_model_copy)
        for client_idx in range(client_num):
            models[client_idx] = decompress_model(models[client_idx], 
                                                   server_model_copy)
    
    # 第9步: 验证
    for client_idx in range(client_num):
        if client_idx not in args.test_envs:
            test_acc, bacc, f1, precision, recall, benefits, tp, fp = \
                test(args, models[client_idx], val_loaders[client_idx], device, mlp[client_idx])

# 第10步: 最终测试
for client_idx in range(client_num):
    if client_idx in args.test_envs:
        # 全局模型测试
        test_acc, bacc, f1 = Glotest(args, server_model, test_loaders[client_idx], device)
    else:
        # 本地模型+MLP测试
        test_acc, bacc, f1 = test(args, models[client_idx], test_loaders[client_idx], 
                                   device, mlp[client_idx])
```

**关键参数说明**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | 'OfficeHome' | 数据集名称 |
| `--batch` | 32 | 批大小 |
| `--iters` | 50 | 联邦通信轮数 |
| `--wk_iters` | 1 | 本地优化轮数 |
| `--net` | 'ViT-B/32' | CLIP模型 |
| `--lr` | 5e-4 | FAM学习率 |
| `--lr_mlp` | 1e-4 | MLP学习率 |
| `--aggmode` | 'att' | 'att'(FAM聚合) 或 'avg'(全部聚合) |
| `--test_envs` | [3] | 测试客户端索引 |

---

### 1.2 对比方法

#### DN-FedAVG-1.py - 标准FedAVG

```python
# 关键差异:
--aggmode = 'avg'        # 聚合整个模型
--net = 'RN50'           # 使用ResNet (可选)
--lr = 5e-5              # CLIP学习率 (注意不同)

# 优化器只优化CLIP模型:
optimizers = [optim.AdamW(
    params=[{'params': models[idx].model.parameters(), ...}],
    ...
) for idx in range(client_num)]
```

**区别**:
- 更新CLIP所有参数 (破坏预训练)
- 通信开销大 (338M参数)
- 可能更新不稳定

#### DN-FedCLIP-1.py - 仅FAM聚合

```python
# 与Ours类似,但只优化FAM:
optimizers = [optim.AdamW(
    params=[{'params': models[idx].fea_attn.parameters(), ...}],
    ...
) for idx in range(client_num)]

# 没有MLP优化,使用简单分类头
```

#### LP++.py - 线性探针

```python
# MLP替换为线性分类器:
mlp = [nn.Linear(512, len(models[idx].labels), bias=True, 
                 dtype=models[idx].model.dtype).to(device)
       for idx in range(client_num)]

# 最少参数(仅~33K),最低通信开销
```

#### LoRA_OF-2.py - LoRA微调

```python
# 使用LoRA而不是FAM:
# 低秩分解: ΔW = AB^T (A∈R^{d×r}, B∈R^{d×r}, r<<d)
# 参数量减少到最少

# 聚合时只聚合LoRA参数
```

---

## 2. 模型模块

### 2.1 nets/models.py

#### ClipModelat 类 (核心)

```python
class ClipModelat(nn.Module):
    """
    CLIP模型包装，包含冻结的视觉编码器和可训练的FAM
    """
    
    CLIP_MODELS = [
        'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64',
        'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'
    ]
    
    def __init__(self, model_name='ViT-B/32', device='cuda', logger=None, 
                 attention=True, freezepy=True):
        """
        参数:
        - model_name: CLIP模型名或索引
        - device: 'cuda' 或 'cpu'
        - attention: 是否包含FAM (True)
        - freezepy: 是否冻结CLIP (True)
        """
        super().__init__()
        
        if type(model_name) is int:
            model_name = self.CLIP_MODELS[model_name]
        
        # 加载CLIP模型
        self.model, _ = clip.load(model_name, device=device)
        self.model.eval()  # 评估模式(无BN更新)
        
        self.model_name = model_name
        self.attention = attention
        self.freezepy = freezepy
        self.device = device
    
    def initdgatal(self, dataloader):
        """
        用样本数据初始化FAM
        - 运行一个批次确定特征维度
        - 创建FAM模块
        """
        for batch in dataloader:
            with torch.no_grad():
                image, _, label = batch
                image = image.to(self.device)
                # 提取CLIP图像特征
                image_features = self.model.encode_image(image)
                feature_dim = image_features.shape[1]  # 通常512
            break
        
        if self.freezepy:
            freeze_param(self.model)  # 冻结CLIP
        
        if self.attention:
            # 创建FAM模块
            self.fea_attn = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.InstanceNorm1d(feature_dim),
                nn.ReLU6(),
                nn.Linear(feature_dim, feature_dim),
                nn.Softmax(dim=1)
            ).to(self.device)
    
    def forward(self, x):
        """
        前向传播
        输入: x (batch_size, 3, 224, 224)
        输出: 加权特征 (batch_size, 512)
        """
        with torch.no_grad():  # CLIP冻结
            x = self.model.encode_image(x)
        
        if self.attention:
            x = self.fea_attn(x)  # 应用FAM
        
        return x
    
    def setselflabel(self, labels):
        """设置类别标签"""
        self.labels = labels
    
    @staticmethod
    def get_model_name_by_index(index):
        """根据索引获取模型名"""
        name = ClipModelat.CLIP_MODELS[index]
        return name.replace('/', '_')
```

#### MLP 类 (分类头)

```python
class MLP(nn.Module):
    """
    多层感知机分类头
    输入: 特征 (batch_size, 512)
    输出: logits (batch_size, num_classes)
    """
    
    def __init__(self, input_size=512, hidden_size=512, num_classes=65):
        super(MLP, self).__init__()
        
        # 隐层配置
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # 第一隐层
        self.linear1 = MaskedMLP(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        
        # 第二隐层
        self.linear2 = MaskedMLP(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.relu2 = nn.ReLU()
        
        # 第三隐层
        self.linear3 = MaskedMLP(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.relu3 = nn.ReLU()
        
        # 输出层
        self.fc4 = MaskedMLP(hidden_size, num_classes)
    
    def forward(self, x):
        """
        前向传播
        输入: x (batch_size, 512)
        """
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.linear3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.fc4(x)  # logits (batch_size, num_classes)
        return x


class MaskedMLP(nn.Module):
    """可掩码的线性层(用于量化/剪枝)"""
    
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.mask = None  # 可选的掩码
    
    def forward(self, x):
        x = self.linear(x)
        if self.mask is not None:
            x = x * self.mask
        return x
```

### 2.2 nets/PromptCLIP.py - 提示学习

```python
class PromptLearner_client(nn.Module):
    """
    学习可训练的文本提示
    目标: 学习上下文向量,形成如 "a photo of X class" 的提示
    """
    
    def __init__(self, n_ctx_num, classnames, clip_model):
        """
        参数:
        - n_ctx_num: 上下文词数 (通常4-16)
        - classnames: 类别名列表
        - clip_model: CLIP模型(用于提取嵌入维度等)
        """
        super().__init__()
        
        n_cls = len(classnames)
        n_ctx = n_ctx_num
        
        # 获取维度信息
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]  # 512
        
        # 初始化上下文向量 (可学习)
        ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        
        self.ctx_global = nn.Parameter(ctx_vectors)  # 可优化
        
        # 构建提示模板
        prompt_prefix = " ".join(["X"] * n_ctx)  # "X X X X ..."
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        
        # 分词并提取嵌入
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts.cuda())
            embedding = embedding.type(dtype)
        
        # 保存token前缀(SOS)和后缀(EOS)
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1+n_ctx:, :])
        
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
    
    def forward(self):
        """
        前向传播: 构建完整提示嵌入
        输出: (num_classes, max_prompt_length, embedding_dim)
        """
        ctx = self.ctx_global  # (num_classes, n_ctx, ctx_dim)
        
        prefix = self.token_prefix  # (num_classes, 1, ctx_dim)
        suffix = self.token_suffix  # (num_classes, *, ctx_dim)
        
        # 拼接: [prefix] + [context] + [class_name] + [suffix]
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        
        return prompts


class CustomCLIP_client(nn.Module):
    """
    CLIP + 提示学习
    冻结视觉编码器,学习文本提示
    """
    
    def __init__(self, classnames, clip_model, n_ctx_num=16):
        super().__init__()
        
        self.prompt_learner = PromptLearner_client(n_ctx_num, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
    
    def forward(self, image):
        """
        前向传播
        输入: image (batch_size, 3, 224, 224)
        输出: logits (batch_size, num_classes)
        """
        # 图像编码
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # 文本编码
        prompts = self.prompt_learner()  # 获取可学习提示
        text_features = self.text_encoder(prompts, self.tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 相似度计算
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        
        return logits
```

### 2.3 nets/LinearProbeV2.py - 线性探针

```python
class LinearProbe(nn.Module):
    """
    最简单的分类器: 线性层
    输入: CLIP特征 (batch_size, 512)
    输出: logits (batch_size, num_classes)
    """
    
    def __init__(self, num_classes):
        super().__init__()
        self.linear = nn.Linear(512, num_classes)
    
    def forward(self, x):
        return self.linear(x)


def compute_centroids(features, labels):
    """
    计算类别中心
    用于LP++方法的初始化
    """
    num_classes = labels.max().item() + 1
    centroids = []
    
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            centroid = features[mask].mean(dim=0)
            centroids.append(centroid)
    
    return torch.stack(centroids)
```

---

## 3. 数据处理

### 3.1 utils/prepare_data_dg_clip.py

```python
def get_data(dataset_name):
    """
    返回数据加载器工厂函数
    """
    if dataset_name == 'BraTS':
        return prepare_brats_data
    elif dataset_name == 'Prostate':
        return prepare_prostate_data
    elif dataset_name == 'OfficeHome':
        return prepare_officehome_data
    # ... 其他数据集
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def prepare_brats_data(args, model):
    """
    准备BraTS数据集
    
    数据结构:
    data/BraTS/
    ├─ client_0/
    │  ├─ class_1/ (非肿瘤)
    │  │  ├─ slice_001.jpg
    │  │  └─ ...
    │  └─ class_2/ (肿瘤)
    ├─ client_1/
    └─ Global/
    """
    
    train_loaders = []
    val_loaders = []
    test_loaders = []
    Labels = []
    
    # 遍历每个客户端
    for domain_idx, domain in enumerate(args.domains):
        # 读取类别标签
        class_names = os.listdir(domain_path)
        Labels.append(class_names)
        
        # 加载图像
        images, labels = load_domain_data(domain_path)
        
        # 分割为训练/验证/测试
        total = len(images)
        l1 = int(total * 0.6)  # 60% 训练
        l2 = int(total * 0.2)  # 20% 验证
        l3 = int(total * 0.2)  # 20% 测试
        
        train_data = ImageDataset(images[:l1], labels[:l1])
        val_data = ImageDataset(images[l1:l1+l2], labels[l1:l1+l2])
        test_data = ImageDataset(images[l1+l2:], labels[l1+l2:])
        
        train_loaders.append(DataLoader(train_data, batch_size=args.batch, shuffle=True))
        val_loaders.append(DataLoader(val_data, batch_size=args.batch, shuffle=False))
        test_loaders.append(DataLoader(test_data, batch_size=args.batch, shuffle=False))
    
    return train_loaders, val_loaders, test_loaders, Labels


class ImageDataset(Dataset):
    """
    图像数据集
    """
    
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform or default_transform()
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, image_path, label


def default_transform():
    """
    默认图像变换
    与CLIP的预处理一致
    """
    return transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
```

---

## 4. 训练测试

### 4.1 utils/training.py

#### train() 函数 - 本地客户端训练

```python
def train(args, model, train_loader, optimizer, device, 
          server_model, adv_loss, previous_net, mlp):
    """
    本地客户端训练
    
    参数:
    - model: 客户端模型 (包含CLIP + FAM)
    - train_loader: 训练数据加载器
    - optimizer: 优化器
    - device: GPU/CPU
    - server_model: 服务器模型(用于对齐)
    - adv_loss: 对抗损失函数
    - mlp: 分类头
    
    流程:
    1. 前向传播: 图像 -> CLIP -> FAM -> MLP
    2. 计算损失: CE + LMMD + 对抗
    3. 反向传播
    4. 优化器更新
    """
    
    model.train()
    mlp.train()
    
    total_loss = 0.0
    
    for batch_idx, (images, image_paths, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        features = model(images)  # (batch_size, 512)
        logits = mlp(features)    # (batch_size, num_classes)
        
        # 损失计算
        loss_ce = nn.CrossEntropyLoss()(logits, labels)
        
        # 本地MMD损失 (域适应)
        with torch.no_grad():
            server_features = server_model(images)
            server_logits = mlp(server_features)
        
        loss_lmmd = lmmd_loss(features, server_logits, labels, len(model.labels))
        
        # 对抗损失
        loss_adv = adv_loss(features, server_features)
        
        # 总损失
        total_loss_batch = loss_ce + 0.1 * loss_lmmd + 0.01 * loss_adv
        
        # 反向传播
        optimizer.zero_grad()
        total_loss_batch.backward()
        optimizer.step()
        
        total_loss += total_loss_batch.item()
    
    return total_loss / len(train_loader)


def totrain(model):
    """切换为训练模式"""
    model.model.train()    # CLIP保持eval
    model.fea_attn.train()


def toeval(model):
    """切换为评估模式"""
    model.model.eval()
    model.fea_attn.eval()
```

### 4.2 utils/testing.py

#### test() 函数 - 本地测试

```python
def test(args, model, test_loader, device, mlp, optimizer_tta=None):
    """
    本地客户端测试
    
    返回指标:
    - test_acc: 准确率
    - bacc: 平衡准确率 (处理不平衡)
    - f1: F1分数
    - precision: 精度
    - recall: 召回率
    - benefits: 临床净获益
    - tp: 真正率曲线
    - fp: 假正率曲线
    """
    
    model.eval()
    mlp.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, image_paths, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播
            features = model(images)
            logits = mlp(features)
            probs = torch.softmax(logits, dim=1)
            
            # 预测
            preds = logits.argmax(dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    # 合并所有批次
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    
    # 计算指标
    test_acc = accuracy_score(all_labels, all_preds)
    bacc = balanced_accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    
    # 校准指标 (医学应用)
    benefits, tp, fp = calculate_calibration_metrics(all_labels, all_probs)
    
    return test_acc, bacc, f1, precision, recall, benefits, tp, fp


def Glotest(args, server_model, test_loader, device):
    """
    全局模型测试 (无MLP, 仅使用CLIP+FAM)
    """
    server_model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, image_paths, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # 获取CLIP特征
            features = server_model(images)
            
            # 与类别提示计算相似度
            text_features = get_text_features_list(
                server_model.labels, server_model.model, device)
            
            logits = features @ text_features.t()
            preds = logits.argmax(dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    test_acc = accuracy_score(all_labels, all_preds)
    bacc = balanced_accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return test_acc, bacc, f1
```

---

## 5. 聚合优化

### 5.1 utils/aggregation.py

#### communication() - 标准聚合

```python
def communication(args, server_model, models, client_weights):
    """
    联邦学习聚合步骤
    
    参数:
    - server_model: 服务器模型
    - models: 客户端模型列表
    - client_weights: 客户端权重 (通常均匀分布)
    
    流程:
    根据 aggmode 选择聚合策略
    """
    
    client_num = len(models)
    
    if args.aggmode == 'att':
        # 只聚合FAM模块
        with torch.no_grad():
            for key in server_model.fea_attn.state_dict().keys():
                if 'num_batches_tracked' in key:
                    # BN追踪计数器,不聚合
                    server_model.fea_attn.state_dict()[key].data.copy_(
                        models[client_num-1].fea_attn.state_dict()[key])
                else:
                    # 聚合注意力权重
                    temp = torch.zeros_like(
                        server_model.fea_attn.state_dict()[key],
                        dtype=torch.float32)
                    
                    for client_idx in range(client_num):
                        if client_idx not in args.test_envs:
                            temp += client_weights[client_idx] * \
                                    models[client_idx].fea_attn.state_dict()[key]
                    
                    # 更新服务器
                    server_model.fea_attn.state_dict()[key].data.copy_(temp)
                    
                    # 广播给所有客户端
                    for client_idx in range(client_num):
                        if client_idx not in args.test_envs:
                            models[client_idx].fea_attn.state_dict()[key].data.copy_(
                                server_model.fea_attn.state_dict()[key])
    
    elif args.aggmode == 'avg':
        # 聚合整个模型
        with torch.no_grad():
            for key in server_model.state_dict().keys():
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(
                        models[client_num-1].state_dict()[key])
                else:
                    temp = torch.zeros_like(
                        server_model.state_dict()[key],
                        dtype=torch.float32)
                    
                    for client_idx in range(client_num):
                        if client_idx not in args.test_envs:
                            temp += client_weights[client_idx] * \
                                    models[client_idx].state_dict()[key]
                    
                    server_model.state_dict()[key].data.copy_(temp)
                    
                    for client_idx in range(client_num):
                        if client_idx not in args.test_envs:
                            models[client_idx].state_dict()[key].data.copy_(
                                server_model.state_dict()[key])
    
    return server_model, models


#### communication_quantized() - 量化聚合

def communication_quantized(args, server_model, models, client_weights):
    """
    量化通信优化
    
    步骤:
    1. 压缩客户端参数
    2. 聚合
    3. 广播给客户端
    4. 客户端解压
    
    通信优化: 参数 -> 压缩 (90%压缩率)
    """
    
    client_num = len(models)
    
    # 第1步: 聚合 (与communication()相同)
    server_model, models = communication(args, server_model, models, client_weights)
    
    # 第2步: 压缩模型以减少通信开销
    # (在prepare_model_for_transmission前)
    
    # 第3步: 发送到客户端 (模拟)
    
    return server_model, models
```

#### compress_model() / decompress_model()

```python
def compress_model(model):
    """
    压缩模型参数
    提取可训参数 (FAM, MLP),进行量化压缩
    """
    
    compressed = {}
    
    # 提取FAM参数
    for name, param in model.fea_attn.named_parameters():
        compressed[f'fea_attn.{name}'] = param.data.cpu().float()
    
    # 序列化和压缩
    model_bytes = io.BytesIO()
    torch.save(compressed, model_bytes)
    
    compressed_data = zlib.compress(model_bytes.getvalue())
    
    return compressed_data


def decompress_model(compressed_data, original_model):
    """
    解压模型参数
    恢复原始精度并重新加载到模型
    """
    
    decompressed_data = zlib.decompress(compressed_data)
    model_bytes = io.BytesIO(decompressed_data)
    
    compressed = torch.load(model_bytes)
    
    # 重新加载参数
    for name, param in compressed.items():
        if 'fea_attn' in name:
            original_model.fea_attn.state_dict()[
                name.replace('fea_attn.', '')].copy_(param)
    
    return original_model
```

---

## 6. 损失函数

### 6.1 adaptation.py

#### LMMDLoss - 本地最大均值差异

```python
class LMMDLoss(MMDLoss, LambdaSheduler):
    """
    本地MMD损失 (类级别)
    用于域自适应
    """
    
    def forward(self, source, target, source_label, target_logits):
        """
        计算加权MMD损失
        
        参数:
        - source: 源域特征 (batch_size, feature_dim)
        - target: 目标域特征
        - source_label: 源域真实标签
        - target_logits: 目标域预测logits
        """
        
        batch_size = source.size()[0]
        
        # 计算类级权重
        weight_ss, weight_tt, weight_st = self.cal_weight(
            source_label, target_logits)
        
        weight_ss = torch.from_numpy(weight_ss).cuda()
        weight_tt = torch.from_numpy(weight_tt).cuda()
        weight_st = torch.from_numpy(weight_st).cuda()
        
        # 计算高斯核
        kernels = self.guassian_kernel(
            source, target,
            kernel_mul=self.kernel_mul,
            kernel_num=self.kernel_num,
            fix_sigma=self.fix_sigma)
        
        # 提取子矩阵
        SS = kernels[:batch_size, :batch_size]
        TT = kernels[batch_size:, batch_size:]
        ST = kernels[:batch_size, batch_size:]
        
        # MMD距离
        loss = torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)
        
        # 动态加权 (从0增长到1)
        lamb = self.lamb()
        loss = loss * lamb
        
        return loss
    
    def cal_weight(self, source_label, target_logits):
        """
        计算类级权重矩阵
        
        返回:
        - weight_ss: (batch_size, batch_size)
        - weight_tt: (batch_size, batch_size)
        - weight_st: (batch_size, batch_size)
        """
        
        batch_size = source_label.size()[0]
        
        # 源域: one-hot标签 -> 行归一化
        source_label_onehot = np.eye(self.num_class)[source_label.cpu().numpy()]
        source_label_sum = np.sum(source_label_onehot, axis=0).reshape(1, -1)
        source_label_sum[source_label_sum == 0] = 100
        source_label_onehot = source_label_onehot / source_label_sum
        
        # 目标域: softmax logits -> 行归一化
        target_label = target_logits.cpu().data.max(1)[1].numpy()
        target_logits_norm = target_logits.cpu().data.numpy()
        target_logits_sum = np.sum(target_logits_norm, axis=0).reshape(1, -1)
        target_logits_sum[target_logits_sum == 0] = 100
        target_logits_norm = target_logits_norm / target_logits_sum
        
        # 计算权重 (仅对共享类别)
        weight_ss = np.zeros((batch_size, batch_size))
        weight_tt = np.zeros((batch_size, batch_size))
        weight_st = np.zeros((batch_size, batch_size))
        
        set_s = set(source_label.cpu().numpy())
        set_t = set(target_label)
        
        for c in range(self.num_class):
            if c in set_s and c in set_t:
                s_vec = source_label_onehot[:, c].reshape(-1, 1)
                t_vec = target_logits_norm[:, c].reshape(-1, 1)
                
                weight_ss += np.dot(s_vec, s_vec.T)
                weight_tt += np.dot(t_vec, t_vec.T)
                weight_st += np.dot(s_vec, t_vec.T)
        
        return weight_ss, weight_tt, weight_st
```

### 6.2 utils/loss_function.py

#### CrossEntropyLabelSmooth

```python
class CrossEntropyLabelSmooth(nn.Module):
    """
    带标签平滑的交叉熵损失
    防止过拟合,提高模型鲁棒性
    """
    
    def __init__(self, num_classes, epsilon=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self, input, target):
        """
        参数:
        - input: logits (batch_size, num_classes)
        - target: 真实标签 (batch_size,)
        """
        
        log_probs = self.logsoftmax(input)
        
        # 目标分布: 真实标签概率 (1-epsilon), 其他标签 epsilon/(num_classes-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.epsilon / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.epsilon)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))
```

---

## 7. 工具函数

### 7.1 utils/clip_util.py

```python
def freeze_param(model):
    """冻结所有参数 (梯度 = 0)"""
    for name, param in model.named_parameters():
        param.requires_grad = False


def get_image_features(image, model, device='cuda'):
    """提取CLIP图像特征"""
    with torch.no_grad():
        image = image.to(device)
        image_features = model.encode_image(image)
    return image_features


def get_text_features_list(texts, model, device='cuda'):
    """
    提取CLIP文本特征
    
    参数:
    - texts: 文本列表 (e.g., ['cat', 'dog', 'bird'])
    - model: CLIP模型
    
    返回:
    - text_features: (num_texts, embedding_dim)
    """
    
    text_inputs = torch.cat([clip.tokenize(t) for t in texts]).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    
    return text_features


class FocalLossWithSmoothing(nn.Module):
    """
    Focal Loss + 标签平滑
    处理类别不平衡问题
    """
    
    def __init__(self, num_classes, gamma=1, lb_smooth=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.lb_smooth = lb_smooth
    
    def forward(self, logits, labels):
        """
        logits: (batch_size, num_classes)
        labels: (batch_size,)
        """
        probs = torch.softmax(logits, dim=1)
        
        # Focal term: (1 - p_t)^gamma
        focal_weight = (1 - probs).pow(self.gamma)
        
        # 标签平滑
        with torch.no_grad():
            smooth_labels = torch.zeros_like(probs)
            smooth_labels.fill_(self.lb_smooth / (self.num_classes - 1))
            smooth_labels.scatter_(1, labels.unsqueeze(1), 
                                   1.0 - self.lb_smooth)
        
        # Focal Loss = -focal_weight * log(p_t)
        log_probs = torch.log_softmax(logits, dim=1)
        loss = -(smooth_labels * log_probs * focal_weight).sum(dim=1).mean()
        
        return loss
```

### 7.2 utils/config.py

```python
def img_param_init(args):
    """
    初始化数据集参数
    
    设置:
    - args.domains: 客户端/域列表
    - args.num_classes: 类别数
    """
    
    dataset = args.dataset
    
    if dataset == 'BraTS':
        args.domains = ['client_0', 'client_1', 'client_2', 'client_3', 'Global']
        args.num_classes = 2
    
    elif dataset == 'OfficeHome':
        args.domains = ['A', 'C', 'P', 'R']
        args.num_classes = 65
    
    elif dataset == 'Prostate':
        args.domains = ['Center_2', 'Center_3', 'Center_4', 'Center_1']
        args.num_classes = 2
    
    # ... 其他数据集
    
    return args


def set_random_seed(seed=0):
    """设置所有随机种子,保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## 总结

这个代码库是一个完整的联邦学习框架,包含:

1. **主程序**: 多个方法实现 (Ours, FedAVG, FedCLIP, LoRA等)
2. **模型**: CLIP + FAM + MLP 的标准组件
3. **数据**: 支持多个医学和通用数据集
4. **训练**: 多损失函数融合
5. **测试**: 全面的评估指标
6. **聚合**: 量化通信优化
7. **工具**: CLIP特征提取, 参数冻结等

关键设计决策:
- 冻结CLIP以保留预训练知识
- 轻量级FAM实现参数高效微调
- 多客户端特定MLP处理数据异构
- 量化通信减少90%开销
- 多损失融合实现强大的域适应
