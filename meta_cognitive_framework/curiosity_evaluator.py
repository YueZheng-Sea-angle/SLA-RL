"""
好奇心评价器 (Curiosity Evaluator)
==================================
这是元认知框架的核心组件，负责评估底层算法从数据中学习的"潜在价值"。

核心思想：
- 输入：数据批次 (s, a, r, s') 和底层网络的参数梯度 g
- 输出：学习价值权重 w ∈ [0, 2]
- 训练目标：拟合优势函数的提升 Target_w = clip(A_new - A_old, 0, 2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CuriosityEvaluator(nn.Module):
    """
    好奇心评价器
    
    评估一个数据批次对于底层算法学习的价值
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=1e-3, device='cpu'):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
            lr: 学习率
            device: 运行设备
        """
        super().__init__()
        self.device = device
        
        # 状态-动作编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        # 梯度信息编码器（接收梯度的统计信息）
        # 输入：[grad_mean, grad_std, grad_norm]
        self.gradient_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.ReLU()
        )
        
        # 价值评估网络
        combined_dim = hidden_dim // 2 + hidden_dim // 4 + hidden_dim // 4
        self.value_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 输出 [0, 1]，后续乘以2得到 [0, 2]
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(device)
        
    def forward(self, states, actions, gradient_stats):
        """
        前向传播
        
        Args:
            states: 状态批次 [batch_size, state_dim]
            actions: 动作批次 [batch_size, action_dim]
            gradient_stats: 梯度统计信息 [batch_size, 3] (mean, std, norm)
            
        Returns:
            weights: 学习价值权重 [batch_size]，范围 [0, 2]
        """
        # 编码状态和动作
        state_features = self.state_encoder(states)
        action_features = self.action_encoder(actions)
        gradient_features = self.gradient_encoder(gradient_stats)
        
        # 组合特征
        combined = torch.cat([state_features, action_features, gradient_features], dim=-1)
        
        # 计算权重 (0 到 2)
        weights = self.value_net(combined).squeeze(-1) * 2.0
        
        return weights
    
    def compute_gradient_stats(self, gradients):
        """
        计算梯度的统计信息
        
        Args:
            gradients: 梯度列表或张量
            
        Returns:
            stats: [mean, std, norm] 形状 [batch_size, 3]
        """
        if isinstance(gradients, list):
            # 如果是梯度列表，展平并连接
            flat_grads = torch.cat([g.flatten() for g in gradients if g is not None])
        else:
            flat_grads = gradients.flatten()
        
        # 计算统计量
        grad_mean = flat_grads.mean().item()
        grad_std = flat_grads.std().item()
        grad_norm = torch.norm(flat_grads).item()
        
        return torch.tensor([grad_mean, grad_std, grad_norm], device=self.device)
    
    def update(self, states, actions, gradient_stats, target_weights):
        """
        更新好奇心评价器
        
        Args:
            states: 状态批次
            actions: 动作批次
            gradient_stats: 梯度统计信息
            target_weights: 目标权重（基于优势函数提升）
            
        Returns:
            loss: 损失值
        """
        # 预测权重
        pred_weights = self.forward(states, actions, gradient_stats)
        
        # MSE损失
        loss = F.mse_loss(pred_weights, target_weights)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate_batch_value(self, states, actions, base_gradients):
        """
        评估一个批次的学习价值
        
        Args:
            states: 状态批次
            actions: 动作批次
            base_gradients: 底层算法计算出的梯度
            
        Returns:
            weights: 每个样本的学习价值权重 [0, 2]
        """
        with torch.no_grad():
            # 为批次中的每个样本计算梯度统计
            batch_size = states.shape[0]
            gradient_stats_batch = []
            
            for i in range(batch_size):
                # 这里简化处理：使用整个批次的梯度统计
                # 实际应用中可以为每个样本单独计算
                stats = self.compute_gradient_stats(base_gradients)
                gradient_stats_batch.append(stats)
            
            gradient_stats_batch = torch.stack(gradient_stats_batch)
            
            # 计算权重
            weights = self.forward(states, actions, gradient_stats_batch)
            
        return weights


class SimplifiedCuriosityEvaluator(nn.Module):
    """
    简化版好奇心评价器
    
    只使用状态-动作信息，不使用梯度信息（更易于实现）
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=1e-3, device='cpu'):
        super().__init__()
        self.device = device
        
        # 简单的MLP网络
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出 [0, 1]
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(device)
    
    def forward(self, states, actions):
        """前向传播"""
        # 如果actions是离散的（整数），需要转换为one-hot
        if actions.dim() == 1 or actions.shape[-1] == 1:
            # 假设是离散动作索引
            actions = actions.long().squeeze(-1)
            # 这里简化处理，直接使用索引
            actions = actions.float().unsqueeze(-1)
        
        x = torch.cat([states, actions], dim=-1)
        weights = self.net(x).squeeze(-1) * 2.0  # [0, 2]
        return weights
    
    def update(self, states, actions, target_weights):
        """更新评价器"""
        pred_weights = self.forward(states, actions)
        loss = F.mse_loss(pred_weights, target_weights)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate_batch_value(self, states, actions):
        """评估批次价值"""
        with torch.no_grad():
            weights = self.forward(states, actions)
        return weights

