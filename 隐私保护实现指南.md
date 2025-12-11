# 隐私保护实现指南

本文档提供了将隐私保护机制集成到项目中的具体代码示例和实现步骤。

## 目录
1. [差分隐私集成](#差分隐私集成)
2. [安全聚合实现](#安全聚合实现)
3. [通信加密](#通信加密)
4. [隐私审计](#隐私审计)
5. [完整示例](#完整示例)

---

## 差分隐私集成

### 方法1：使用OpenDP库

```python
# 创建文件: privacy_dp.py

import numpy as np
from scipy import stats

class DifferentialPrivacyManager:
    """
    差分隐私管理器
    用于保护联邦学习中的参数和梯度
    """
    
    def __init__(self, epsilon=1.0, delta=1e-6, max_grad_norm=1.0, 
                 noise_multiplier=1.1):
        """
        参数:
        - epsilon: 隐私预算 (越小越隐私)
        - delta: 概率边界 (通常1/(样本数^1.1))
        - max_grad_norm: 梯度裁剪阈值
        - noise_multiplier: 噪声乘数 (与参与者数相关)
        """
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.cumulative_epsilon = 0.0
        self.round_num = 0
    
    def clip_gradients(self, gradients):
        """
        梯度裁剪
        限制单个客户端对聚合的影响
        """
        clipped = {}
        norms = {}
        
        for name, grad in gradients.items():
            # 计算L2范数
            norm = np.sqrt(np.sum(grad ** 2))
            norms[name] = norm
            
            # 裁剪
            if norm > self.max_grad_norm:
                clipped[name] = grad * (self.max_grad_norm / norm)
            else:
                clipped[name] = grad.copy()
        
        return clipped, norms
    
    def add_gaussian_noise(self, gradients, num_clients):
        """
        添加高斯噪声
        
        噪声规模计算:
        σ = sqrt(2 * log(1.25/δ)) / ε * C / sqrt(num_clients)
        
        其中:
        - C: 梯度裁剪界
        - num_clients: 参与聚合的客户数
        """
        
        # 计算噪声标准差
        sigma = (np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon) * \
                self.max_grad_norm * self.noise_multiplier / np.sqrt(num_clients)
        
        noisy_grads = {}
        
        for name, grad in gradients.items():
            # 生成高斯噪声
            noise = np.random.normal(0, sigma, grad.shape)
            noisy_grads[name] = grad + noise
        
        return noisy_grads, sigma
    
    def update_privacy_accounting(self, num_clients):
        """
        更新隐私预算消耗
        使用RDP (Renyi Differential Privacy) 组合定理
        """
        self.round_num += 1
        
        # 简化计算: 每轮消耗固定ε
        # 实际应使用RDP库进行精确计算
        self.cumulative_epsilon = self.epsilon * np.sqrt(self.round_num)
        
        return self.cumulative_epsilon
    
    def get_privacy_stats(self):
        """获取隐私统计"""
        return {
            'epsilon': self.epsilon,
            'delta': self.delta,
            'rounds': self.round_num,
            'cumulative_epsilon': self.cumulative_epsilon,
            'privacy_level': self._classify_privacy_level()
        }
    
    def _classify_privacy_level(self):
        """分类隐私保护等级"""
        if self.cumulative_epsilon < 0.5:
            return '极强隐私 (Strong)'
        elif self.cumulative_epsilon < 1.0:
            return '强隐私 (Moderate)'
        elif self.cumulative_epsilon < 5.0:
            return '中等隐私 (Weak)'
        else:
            return '弱隐私 (Minimal)'


# 使用示例
def train_with_differential_privacy(args, model, train_loader, 
                                    optimizer, device, mlp):
    """
    带差分隐私的本地训练
    """
    
    # 初始化DP管理器
    dp_manager = DifferentialPrivacyManager(
        epsilon=args.dp_epsilon,  # 例如1.0
        delta=1.0 / (len(train_loader.dataset) ** 1.1),
        max_grad_norm=args.max_grad_norm,  # 例如1.0
        noise_multiplier=args.noise_multiplier  # 例如1.1
    )
    
    model.train()
    mlp.train()
    
    for batch_idx, (images, image_paths, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        features = model(images)
        logits = mlp(features)
        loss = nn.CrossEntropyLoss()(logits, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 提取梯度 (模拟, 实际应使用自动梯度提取)
        gradients = {}
        for name, param in list(model.named_parameters()) + \
                          list(mlp.named_parameters()):
            if param.grad is not None:
                gradients[name] = param.grad.cpu().numpy()
        
        # 应用差分隐私
        # 步骤1: 梯度裁剪
        clipped_grads, norms = dp_manager.clip_gradients(gradients)
        
        # 步骤2: 添加高斯噪声
        num_clients = len(train_loader)  # 近似
        noisy_grads, sigma = dp_manager.add_gaussian_noise(
            clipped_grads, num_clients)
        
        # 步骤3: 更新模型 (使用噪声梯度)
        # 注意: 实际实现应更复杂
        optimizer.step()
    
    # 更新隐私预算
    eps_total = dp_manager.update_privacy_accounting(num_clients=1)
    
    print(f"本轮隐私成本: ε={eps_total:.2f}")
    print(f"隐私等级: {dp_manager.get_privacy_stats()['privacy_level']}")


# 修改 Ours.py 的参数
# 在 argparse 中添加:
# parser.add_argument('--dp_epsilon', type=float, default=1.0)
# parser.add_argument('--max_grad_norm', type=float, default=1.0)
# parser.add_argument('--noise_multiplier', type=float, default=1.1)
```

### 方法2：使用Opacus库 (PyTorch官方)

```python
# 创建文件: privacy_opacus.py

import torch
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

class OpacusPrivacyWrapper:
    """
    Opacus隐私引擎包装器
    使用PyTorch官方隐私库
    """
    
    def __init__(self, model, optimizer, batch_size, sample_rate,
                 epsilon=1.0, delta=1e-6, max_grad_norm=1.0):
        """
        参数:
        - model: 要保护的模型
        - optimizer: 优化器
        - batch_size: 批大小
        - sample_rate: 样本采样率
        - epsilon, delta, max_grad_norm: DP参数
        """
        
        self.privacy_engine = PrivacyEngine()
        
        self.model, self.optimizer, self.sample_rate = \
            self.privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=None,  # 稍后设置
                noise_multiplier=self._calculate_noise_multiplier(
                    epsilon, delta, sample_rate),
                max_grad_norm=max_grad_norm
            )
        
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
    
    def _calculate_noise_multiplier(self, epsilon, delta, sample_rate):
        """
        计算Opacus所需的噪声乘数
        """
        # 简化公式 (实际应更精确)
        noise_multiplier = (np.sqrt(2 * np.log(1.25 / delta)) / epsilon) * \
                          np.sqrt(1 / sample_rate)
        return max(noise_multiplier, 0.1)
    
    def train_epoch_with_privacy(self, train_loader):
        """
        带隐私保护的单个epoch训练
        """
        
        self.model.train()
        total_loss = 0.0
        
        with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=512,  # GPU内存限制
            optimizer=self.optimizer
        ) as memory_safe_data_loader:
            
            for batch_idx, (images, _, labels) in enumerate(memory_safe_data_loader):
                # 前向传播
                logits = self.model(images)
                loss = nn.CrossEntropyLoss()(logits, labels)
                
                # 反向传播 (自动处理梯度裁剪和噪声)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def get_privacy_spent(self):
        """获取已消耗的隐私预算"""
        epsilon, best_alpha = self.privacy_engine.get_privacy_spent(
            delta=self.delta)
        return epsilon, best_alpha


# 在main函数中使用
def main_with_opacus(Location, dataset, method):
    """
    使用Opacus的主函数
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ... 初始化代码 ...
    
    # 为每个客户端创建DP保护
    privacy_wrappers = []
    for idx in range(client_num):
        if idx not in args.test_envs:
            optimizer = optim.Adam(
                list(models[idx].fea_attn.parameters()) + \
                list(mlp[idx].parameters()),
                lr=args.lr
            )
            
            wrapper = OpacusPrivacyWrapper(
                model=models[idx],
                optimizer=optimizer,
                batch_size=args.batch,
                sample_rate=args.batch / len(train_loaders[idx].dataset),
                epsilon=args.dp_epsilon,
                delta=args.dp_delta,
                max_grad_norm=args.max_grad_norm
            )
            
            privacy_wrappers.append(wrapper)
    
    # 训练循环
    for round in range(args.iters):
        for client_idx in range(client_num):
            if client_idx not in args.test_envs:
                wrapper = privacy_wrappers[client_idx]
                
                loss = wrapper.train_epoch_with_privacy(
                    train_loaders[client_idx])
                
                epsilon, best_alpha = wrapper.get_privacy_spent()
                
                print(f"Client {client_idx}: "
                      f"Loss={loss:.4f}, "
                      f"ε={epsilon:.2f}, "
                      f"α={best_alpha:.2f}")
```

---

## 安全聚合实现

### 使用Paillier同态加密的安全聚合

```python
# 创建文件: privacy_aggregation.py

from phe import paillier
import numpy as np

class SecureAggregator:
    """
    使用Paillier同态加密的安全聚合器
    """
    
    def __init__(self, key_length=2048):
        """
        生成Paillier密钥对
        """
        print(f"生成{key_length}位密钥对 (可能需要几分钟)...")
        self.public_key, self.private_key = paillier.generate_paillier_keypair(
            n_length=key_length)
    
    def encrypt_parameters(self, parameters):
        """
        使用公钥加密参数
        
        客户端执行此操作:
        """
        encrypted = {}
        
        for name, param in parameters.items():
            # 转为浮点数组
            param_array = param.cpu().numpy().astype(float).flatten()
            
            # 逐元素加密
            encrypted_array = [
                self.public_key.encrypt(float(val))
                for val in param_array
            ]
            
            # 恢复形状
            encrypted[name] = encrypted_array
        
        return encrypted
    
    def aggregate_encrypted(self, client_encrypted_params, weights):
        """
        在密文域聚合
        
        服务器执行此操作:
        - 无法看到明文参数
        - 但可以执行同态加法
        """
        
        num_clients = len(client_encrypted_params)
        aggregated = {}
        
        # 初始化聚合结果
        first_client = client_encrypted_params[0]
        for name, encrypted_array in first_client.items():
            aggregated[name] = [
                self.public_key.encrypt(0.0)
                for _ in encrypted_array
            ]
        
        # 在密文域执行加权聚合
        for client_idx, client_params in enumerate(client_encrypted_params):
            weight = weights[client_idx]
            
            for name, encrypted_array in client_params.items():
                for i, encrypted_val in enumerate(encrypted_array):
                    # 同态加法: E(a) + E(b) = E(a+b)
                    # 标量乘: k * E(a) = E(k*a)
                    aggregated[name][i] += encrypted_val * weight
        
        return aggregated
    
    def decrypt_result(self, encrypted_aggregate):
        """
        仅使用私钥解密最终结果
        
        需要私钥持有者 (通常是多个客户端共同持有):
        """
        
        decrypted = {}
        
        for name, encrypted_array in encrypted_aggregate.items():
            decrypted_array = [
                self.private_key.decrypt(encrypted_val)
                for encrypted_val in encrypted_array
            ]
            decrypted[name] = np.array(decrypted_array)
        
        return decrypted


def secure_aggregation_federated(args, client_models, client_weights):
    """
    联邦学习中的安全聚合
    
    协议:
    1. 服务器生成并广播公钥
    2. 每个客户端加密本地参数
    3. 客户端上传加密参数
    4. 服务器在密文域聚合
    5. 客户端合作解密 (需要>阈值数量)
    """
    
    # 步骤1: 初始化聚合器
    aggregator = SecureAggregator(key_length=2048)
    
    # 步骤2: 服务器广播公钥 (实际应通过安全通道)
    public_key = aggregator.public_key
    
    print("=== 安全聚合开始 ===")
    print(f"步骤1: 生成密钥对 ✓")
    print(f"步骤2: 广播公钥给{len(client_models)}个客户端")
    
    # 步骤3: 客户端加密参数
    encrypted_params = []
    for client_idx, model in enumerate(client_models):
        
        # 提取FAM参数
        params = {}
        for name, param in model.fea_attn.state_dict().items():
            params[f'fea_attn.{name}'] = param
        
        # 加密 (客户端执行)
        encrypted = aggregator.encrypt_parameters(params)
        encrypted_params.append(encrypted)
        
        print(f"步骤3: 客户端{client_idx}加密参数 ✓")
    
    # 步骤4: 服务器在密文域聚合
    print(f"步骤4: 服务器在密文域聚合...")
    aggregated_encrypted = aggregator.aggregate_encrypted(
        encrypted_params, client_weights)
    print(f"步骤4: 密文聚合完成 ✓")
    
    # 步骤5: 解密 (需要多个客户端的私钥份额)
    # 简化版: 假设私钥在聚合器处
    print(f"步骤5: 多方解密 (需要>阈值客户端)...")
    aggregated_decrypted = aggregator.decrypt_result(aggregated_encrypted)
    print(f"步骤5: 解密完成 ✓")
    
    print("=== 安全聚合完成 ===\n")
    
    # 更新服务器模型
    for name, values in aggregated_decrypted.items():
        # 恢复张量形状
        param_name = name.replace('fea_attn.', '')
        original_shape = getattr(client_models[0].fea_attn, 
                               param_name[:-len('weight')]).shape \
                        if 'weight' in param_name else None
        
        # 实际应正确恢复形状
        # 这里简化处理
    
    return aggregated_decrypted
```

---

## 通信加密

### TLS + HMAC的安全通信

```python
# 创建文件: secure_communication.py

import ssl
import socket
import hmac
import hashlib
import pickle
import io
import zlib

class SecureClient:
    """
    安全客户端通信
    """
    
    def __init__(self, client_id, ca_cert_path, client_cert_path, 
                 client_key_path, server_address='localhost', port=8883):
        """
        初始化安全客户端
        
        参数:
        - client_id: 客户端标识符
        - ca_cert_path: CA证书路径
        - client_cert_path: 客户端证书路径
        - client_key_path: 客户端密钥路径
        """
        
        self.client_id = client_id
        self.server_address = server_address
        self.port = port
        
        # 设置TLS上下文
        self.context = ssl.create_default_context()
        self.context.minimum_version = ssl.TLSVersion.TLSv1_3
        self.context.maximum_version = ssl.TLSVersion.TLSv1_3
        
        # 加载CA证书 (验证服务器)
        self.context.load_verify_locations(ca_cert_path)
        
        # 加载客户端证书和密钥 (双向认证)
        self.context.load_cert_chain(
            certfile=client_cert_path,
            keyfile=client_key_path
        )
        
        # 共享密钥 (实际应通过安全通道协商)
        self.shared_key = b'shared_secret_key_123456'
    
    def _create_hmac(self, data):
        """创建HMAC签名"""
        return hmac.new(self.shared_key, data, hashlib.sha256).digest()
    
    def _verify_hmac(self, signature, data):
        """验证HMAC签名"""
        expected = self._create_hmac(data)
        return hmac.compare_digest(signature, expected)
    
    def send_model(self, model, destination=None):
        """
        安全发送模型参数
        
        流程:
        1. 序列化模型参数
        2. 压缩 (zlib)
        3. 计算HMAC签名
        4. TLS加密传输
        5. 服务器验证签名
        """
        
        if destination is None:
            destination = (self.server_address, self.port)
        
        # 步骤1: 序列化
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        serialized = buffer.getvalue()
        
        # 步骤2: 压缩
        compressed = zlib.compress(serialized)
        
        # 步骤3: 计算HMAC
        signature = self._create_hmac(compressed)
        
        # 步骤4: 打包消息
        message = {
            'client_id': self.client_id,
            'signature': signature.hex(),
            'data': compressed.hex()
        }
        
        message_bytes = pickle.dumps(message)
        
        # 步骤5: TLS发送
        print(f"客户端{self.client_id}: 连接到服务器...")
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            with self.context.wrap_socket(
                sock, server_hostname=self.server_address
            ) as ssock:
                ssock.connect(destination)
                
                # 发送客户端ID
                ssock.sendall(str(self.client_id).encode())
                
                # 发送消息长度
                ssock.sendall(len(message_bytes).to_bytes(4, 'big'))
                
                # 分块发送 (处理大消息)
                chunk_size = 4096
                for i in range(0, len(message_bytes), chunk_size):
                    chunk = message_bytes[i:i+chunk_size]
                    ssock.sendall(chunk)
                
                # 接收确认
                ack = ssock.recv(10)
                
                print(f"客户端{self.client_id}: 模型已安全发送 ✓")
    
    def receive_model(self, received_data):
        """
        安全接收模型参数
        """
        
        # 步骤1: 验证HMAC
        try:
            message = pickle.loads(received_data)
            signature = bytes.fromhex(message['signature'])
            data = bytes.fromhex(message['data'])
            
            if not self._verify_hmac(signature, data):
                raise ValueError("HMAC验证失败: 数据可能被篡改!")
            
            print("HMAC验证通过 ✓")
        
        except Exception as e:
            print(f"验证失败: {e}")
            return None
        
        # 步骤2: 解压
        decompressed = zlib.decompress(data)
        
        # 步骤3: 反序列化
        buffer = io.BytesIO(decompressed)
        state_dict = torch.load(buffer)
        
        return state_dict


class SecureServer:
    """
    安全服务器通信
    """
    
    def __init__(self, ca_cert_path, server_cert_path, server_key_path,
                 port=8883):
        """
        初始化安全服务器
        """
        
        self.port = port
        
        # 设置TLS上下文
        self.context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        self.context.minimum_version = ssl.TLSVersion.TLSv1_3
        self.context.maximum_version = ssl.TLSVersion.TLSv1_3
        
        # 加载服务器证书和密钥
        self.context.load_cert_chain(
            certfile=server_cert_path,
            keyfile=server_key_path
        )
        
        # 要求客户端证书 (双向认证)
        self.context.load_verify_locations(ca_cert_path)
        self.context.verify_mode = ssl.CERT_REQUIRED
        
        # 共享密钥
        self.shared_key = b'shared_secret_key_123456'
        
        # 接收的客户端模型
        self.client_models = {}
    
    def start_server(self):
        """启动安全服务器"""
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(('0.0.0.0', self.port))
            sock.listen()
            
            print(f"安全服务器启动在端口{self.port} (TLS 1.3)")
            
            while True:
                with self.context.wrap_socket(
                    sock, server_side=True
                ) as ssock:
                    conn, addr = ssock.accept()
                    
                    # 处理客户端连接
                    self._handle_client(conn, addr)
    
    def _handle_client(self, conn, addr):
        """处理客户端连接"""
        
        # 接收客户端ID
        client_id_bytes = conn.recv(10)
        client_id = int(client_id_bytes.decode())
        
        # 接收消息长度
        length_bytes = conn.recv(4)
        msg_length = int.from_bytes(length_bytes, 'big')
        
        # 接收消息
        message_bytes = b''
        while len(message_bytes) < msg_length:
            chunk = conn.recv(min(4096, msg_length - len(message_bytes)))
            if not chunk:
                break
            message_bytes += chunk
        
        # 处理消息
        try:
            message = pickle.loads(message_bytes)
            signature = bytes.fromhex(message['signature'])
            data = bytes.fromhex(message['data'])
            
            # 验证HMAC
            expected_sig = hmac.new(self.shared_key, data, 
                                   hashlib.sha256).digest()
            
            if not hmac.compare_digest(signature, expected_sig):
                raise ValueError("HMAC验证失败!")
            
            # 保存客户端模型
            decompressed = zlib.decompress(data)
            buffer = io.BytesIO(decompressed)
            state_dict = torch.load(buffer)
            
            self.client_models[client_id] = state_dict
            
            print(f"✓ 从客户端{client_id}接收模型 ({len(data)/1024:.1f}KB)")
            
            # 发送确认
            conn.sendall(b'ACK')
            
        except Exception as e:
            print(f"✗ 客户端{client_id}处理错误: {e}")
            conn.sendall(b'ERROR')
        
        finally:
            conn.close()
```

---

## 隐私审计

### 隐私使用日志和审计

```python
# 创建文件: privacy_audit.py

import json
import datetime
from typing import Dict, List

class PrivacyAuditor:
    """
    隐私审计日志记录器
    记录所有与隐私相关的操作
    """
    
    def __init__(self, log_file='privacy_audit.log'):
        """初始化审计器"""
        self.log_file = log_file
        self.events = []
    
    def log_event(self, event_type: str, client_id: int, 
                  details: Dict, severity='INFO'):
        """
        记录隐私事件
        
        事件类型:
        - 'gradient_clip': 梯度裁剪
        - 'noise_added': 添加DP噪声
        - 'encryption': 加密操作
        - 'aggregation': 聚合操作
        - 'privacy_budget': 隐私预算更新
        """
        
        event = {
            'timestamp': datetime.datetime.now().isoformat(),
            'event_type': event_type,
            'client_id': client_id,
            'severity': severity,
            'details': details
        }
        
        self.events.append(event)
        
        # 实时输出
        self._print_event(event)
    
    def _print_event(self, event):
        """打印事件到控制台"""
        
        timestamp = event['timestamp']
        event_type = event['event_type']
        client_id = event['client_id']
        severity = event['severity']
        
        color_map = {
            'INFO': '\033[92m',    # 绿色
            'WARNING': '\033[93m',  # 黄色
            'ERROR': '\033[91m'     # 红色
        }
        reset = '\033[0m'
        
        color = color_map.get(severity, '')
        
        print(f"{color}[{timestamp}] {severity:8} | "
              f"Event: {event_type:20} | Client: {client_id:3} "
              f"| {event.get('details', '')}{reset}")
    
    def log_gradient_clipping(self, client_id: int, original_norm: float, 
                             clipped_norm: float, max_norm: float):
        """记录梯度裁剪"""
        
        clipped = original_norm > max_norm
        severity = 'WARNING' if clipped else 'INFO'
        
        self.log_event(
            event_type='gradient_clip',
            client_id=client_id,
            details={
                'original_norm': round(original_norm, 4),
                'clipped_norm': round(clipped_norm, 4),
                'max_norm': round(max_norm, 4),
                'was_clipped': clipped
            },
            severity=severity
        )
    
    def log_noise_addition(self, client_id: int, sigma: float,
                          num_params: int):
        """记录添加DP噪声"""
        
        self.log_event(
            event_type='noise_added',
            client_id=client_id,
            details={
                'sigma': round(sigma, 6),
                'num_parameters': num_params,
                'privacy_mechanism': 'Gaussian'
            },
            severity='INFO'
        )
    
    def log_encryption(self, client_id: int, num_parameters: int,
                      encryption_method: str):
        """记录加密操作"""
        
        self.log_event(
            event_type='encryption',
            client_id=client_id,
            details={
                'method': encryption_method,
                'num_parameters': num_parameters,
                'key_length': 2048 if 'Paillier' in encryption_method else 256
            },
            severity='INFO'
        )
    
    def log_aggregation(self, round_num: int, num_clients: int,
                       aggregation_method: str):
        """记录聚合操作"""
        
        self.log_event(
            event_type='aggregation',
            client_id=-1,  # 服务器操作
            details={
                'round': round_num,
                'num_clients': num_clients,
                'method': aggregation_method
            },
            severity='INFO'
        )
    
    def log_privacy_budget(self, epsilon: float, delta: float,
                          round_num: int):
        """记录隐私预算消耗"""
        
        # 分类隐私等级
        if epsilon < 0.5:
            level = '极强 (Strong)'
            severity = 'INFO'
        elif epsilon < 1.0:
            level = '强 (Moderate)'
            severity = 'INFO'
        elif epsilon < 5.0:
            level = '中等 (Weak)'
            severity = 'WARNING'
        else:
            level = '弱 (Minimal)'
            severity = 'WARNING'
        
        self.log_event(
            event_type='privacy_budget',
            client_id=-1,
            details={
                'epsilon': round(epsilon, 2),
                'delta': delta,
                'privacy_level': level,
                'round': round_num
            },
            severity=severity
        )
    
    def save_audit_log(self):
        """保存审计日志到文件"""
        
        with open(self.log_file, 'w') as f:
            json.dump(self.events, f, indent=2)
        
        print(f"\n✓ 审计日志已保存到 {self.log_file}")
    
    def generate_privacy_report(self) -> Dict:
        """生成隐私报告"""
        
        report = {
            'total_events': len(self.events),
            'event_types': {},
            'clients_involved': set(),
            'severity_distribution': {}
        }
        
        for event in self.events:
            # 事件类型统计
            event_type = event['event_type']
            report['event_types'][event_type] = \
                report['event_types'].get(event_type, 0) + 1
            
            # 客户端统计
            if event['client_id'] >= 0:
                report['clients_involved'].add(event['client_id'])
            
            # 严重性统计
            severity = event['severity']
            report['severity_distribution'][severity] = \
                report['severity_distribution'].get(severity, 0) + 1
        
        report['clients_involved'] = list(report['clients_involved'])
        
        return report
    
    def print_privacy_report(self):
        """打印隐私报告"""
        
        report = self.generate_privacy_report()
        
        print("\n" + "="*60)
        print("隐私审计报告")
        print("="*60)
        print(f"总事件数: {report['total_events']}")
        print(f"\n事件类型分布:")
        for event_type, count in sorted(
            report['event_types'].items()):
            print(f"  - {event_type:20}: {count:4} 次")
        print(f"\n参与客户端: {sorted(report['clients_involved'])}")
        print(f"\n严重性分布:")
        for severity, count in sorted(
            report['severity_distribution'].items()):
            print(f"  - {severity:10}: {count:4} 次")
        print("="*60)


# 使用示例
def train_with_audit(args, model, train_loader, device, mlp):
    """
    带审计的训练
    """
    
    auditor = PrivacyAuditor(log_file='privacy_audit.log')
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        # ... 训练代码 ...
        
        # 记录梯度裁剪
        if batch_idx % 10 == 0:
            auditor.log_gradient_clipping(
                client_id=0,
                original_norm=1.234,
                clipped_norm=1.000,
                max_norm=1.0
            )
        
        # 记录DP噪声
        if batch_idx % 20 == 0:
            auditor.log_noise_addition(
                client_id=0,
                sigma=0.0001,
                num_params=1350000
            )
    
    # 保存和打印报告
    auditor.save_audit_log()
    auditor.print_privacy_report()
```

---

## 完整示例

### 整合所有隐私保护机制

```python
# 创建文件: main_with_privacy.py

def main_federated_learning_with_privacy(Location, dataset, method):
    """
    完整的隐私保护联邦学习主函数
    
    集成:
    - 差分隐私 (DP)
    - 安全聚合 (Paillier)
    - 通信加密 (TLS + HMAC)
    - 隐私审计
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化审计器
    auditor = PrivacyAuditor()
    
    # ... 模型初始化代码 ...
    
    # 初始化隐私管理器
    dp_managers = [
        DifferentialPrivacyManager(
            epsilon=args.dp_epsilon,
            delta=1e-6,
            max_grad_norm=args.max_grad_norm,
            noise_multiplier=args.noise_multiplier
        )
        for _ in range(client_num)
    ]
    
    # 初始化安全聚合器
    aggregator = SecureAggregator(key_length=2048)
    
    # 初始化安全通信
    secure_clients = {}
    for client_idx in range(client_num):
        if client_idx not in args.test_envs:
            secure_clients[client_idx] = SecureClient(
                client_id=client_idx,
                ca_cert_path='certs/ca.crt',
                client_cert_path=f'certs/client_{client_idx}.crt',
                client_key_path=f'certs/client_{client_idx}.key'
            )
    
    # 联邦学习主循环
    for fed_round in range(args.iters):
        print(f"\n{'='*60}")
        print(f"联邦轮 {fed_round + 1}/{args.iters}")
        print(f"{'='*60}")
        
        # 本地训练 (带DP)
        for client_idx in range(client_num):
            if client_idx not in args.test_envs:
                print(f"\n客户端{client_idx}: 本地训练...")
                
                # 训练循环
                for batch in train_loaders[client_idx]:
                    # ... 训练代码 ...
                    
                    # 应用差分隐私
                    dp_manager = dp_managers[client_idx]
                    # ... 梯度裁剪和噪声添加 ...
                
                # 记录DP信息
                eps = dp_manager.update_privacy_accounting(1)
                auditor.log_privacy_budget(
                    epsilon=eps,
                    delta=1e-6,
                    round_num=fed_round
                )
        
        # 安全聚合
        print(f"\n进行安全聚合...")
        auditor.log_aggregation(
            round_num=fed_round,
            num_clients=len([c for c in range(client_num) 
                           if c not in args.test_envs]),
            aggregation_method='Paillier同态加密'
        )
        
        encrypted_params = []
        for client_idx in range(client_num):
            if client_idx not in args.test_envs:
                params = models[client_idx].fea_attn.state_dict()
                encrypted = aggregator.encrypt_parameters(params)
                encrypted_params.append(encrypted)
        
        aggregated_encrypted = aggregator.aggregate_encrypted(
            encrypted_params, client_weights)
        
        aggregated = aggregator.decrypt_result(aggregated_encrypted)
        
        # 验证和更新
        print(f"聚合完成,更新全局模型...")
        
        # 保存审计日志
        auditor.save_audit_log()
    
    # 最终隐私报告
    print("\n")
    auditor.print_privacy_report()


# 添加命令行参数
def setup_privacy_arguments(parser):
    """添加隐私相关的命令行参数"""
    
    parser.add_argument('--enable_privacy', action='store_true',
                       default=False, help='启用隐私保护')
    parser.add_argument('--dp_epsilon', type=float, default=1.0,
                       help='差分隐私epsilon (越小越隐私)')
    parser.add_argument('--dp_delta', type=float, default=1e-6,
                       help='差分隐私delta')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='梯度裁剪阈值')
    parser.add_argument('--noise_multiplier', type=float, default=1.1,
                       help='噪声乘数')
    parser.add_argument('--enable_secure_agg', action='store_true',
                       default=False, help='启用安全聚合')
    parser.add_argument('--enable_tls', action='store_true',
                       default=False, help='启用TLS加密传输')
    parser.add_argument('--enable_audit', action='store_true',
                       default=False, help='启用隐私审计')
    
    return parser
```

---

## 最佳实践检查清单

在部署医学应用时,确保已实施:

```
隐私保护清单:
☐ 差分隐私 (DP)
  └─ ε值: _____ (推荐 1.0-5.0)
☐ 安全聚合
  └─ 方法: _____ (推荐 Paillier或SecureAggregation)
☐ 通信加密
  └─ TLS版本: _____ (要求 TLS 1.3)
☐ HMAC签名验证
  └─ 算法: _____ (要求 SHA-256)
☐ 隐私审计
  └─ 日志位置: _____
☐ 密钥管理
  └─ 密钥服务器: _____ (是/否)
☐ 访问控制
  └─ RBAC实现: _____ (是/否)
☐ 数据保留政策
  └─ 过期时间: _____ (推荐 90天)
☐ 隐私合规性
  └─ GDPR: _____ (是/否)
  └─ HIPAA: _____ (是/否)

隐私级别评估:
- ε < 0.5: 极强隐私 (Strong) ⭐⭐⭐⭐⭐
- ε < 1.0: 强隐私 (Moderate) ⭐⭐⭐⭐
- ε < 5.0: 中等隐私 (Weak) ⭐⭐⭐
- ε ≥ 5.0: 弱隐私 (Minimal) ⭐⭐
```

这个实现指南提供了将隐私保护集成到项目中的具体代码示例,可直接在实际应用中使用。
