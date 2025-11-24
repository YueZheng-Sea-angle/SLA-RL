"""
修复版基础RL算法
================
解决Base DQN的稳定性问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """经验回放池"""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """Q网络（带改进的初始化）"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 改进的权重初始化
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, state):
        return self.net(state)


class StableDQN:
    """
    稳定版DQN
    
    修复点：
    1. ✓ 梯度裁剪
    2. ✓ Huber Loss (SmoothL1Loss)
    3. ✓ 目标网络软更新选项
    4. ✓ Double DQN
    5. ✓ 更保守的学习率
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=3e-4,  # 降低学习率
                 gamma=0.99, epsilon=1.0, epsilon_decay=0.995, 
                 epsilon_min=0.01, tau=0.005, use_double_dqn=True,
                 device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.tau = tau  # 软更新系数
        self.use_double_dqn = use_double_dqn
        self.device = device
        
        # Q网络
        self.q_net = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_q_net = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        # 使用更保守的优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()
        
        self.update_count = 0
        self.target_update_freq = 10  # 硬更新频率（如果不用软更新）
        
        # 用于追踪训练统计
        self.loss_history = deque(maxlen=1000)
        
    def select_action(self, state, eval_mode=False):
        """选择动作（ε-贪心策略）"""
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state)
            return q_values.argmax(dim=1).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储转移"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def compute_loss(self, batch):
        """
        计算DQN损失（使用Huber Loss）
        
        Returns:
            loss: 损失值
            td_error: TD误差（用于元认知）
        """
        states, actions, rewards, next_states, dones = batch
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 当前Q值
        q_values = self.q_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 目标Q值
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: 用当前网络选动作，目标网络评估
                next_actions = self.q_net(next_states).argmax(1)
                next_q_values = self.target_q_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # 标准DQN
                next_q_values = self.target_q_net(next_states).max(1)[0]
            
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # 计算TD误差（绝对值，用于元认知）
        td_error = torch.abs(q_values - target_q_values).mean().item()
        
        # 使用Huber Loss (SmoothL1Loss) - 对异常值更鲁棒
        loss = F.smooth_l1_loss(q_values, target_q_values)
        
        return loss, td_error
    
    def update(self, batch_size=64, weight=None):
        """
        更新Q网络（加入梯度裁剪和稳定性措施）
        
        Args:
            batch_size: 批次大小
            weight: 元认知权重（可选）
            
        Returns:
            loss: 损失值
            td_error: TD误差
        """
        if len(self.replay_buffer) < batch_size:
            return 0.0, 0.0
        
        # 采样批次
        batch = self.replay_buffer.sample(batch_size)
        loss, td_error = self.compute_loss(batch)
        
        # 如果有权重，应用权重
        if weight is not None:
            loss = loss * weight
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 🔧 关键修复：梯度裁剪！！！
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        
        # 更新目标网络（软更新）
        self.update_count += 1
        if self.tau > 0:
            # 软更新：每次都更新一点点
            for param, target_param in zip(self.q_net.parameters(), self.target_q_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:
            # 硬更新：每N步完全复制
            if self.update_count % self.target_update_freq == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        # 衰减epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # 记录损失
        self.loss_history.append(loss.item())
        
        return loss.item(), td_error
    
    def get_q_values(self, states):
        """获取Q值（用于计算优势函数）"""
        with torch.no_grad():
            if not isinstance(states, torch.Tensor):
                states = torch.FloatTensor(states).to(self.device)
            q_values = self.q_net(states)
        return q_values
    
    def get_avg_loss(self):
        """获取平均损失（用于监控）"""
        if len(self.loss_history) > 0:
            return np.mean(list(self.loss_history))
        return 0.0

