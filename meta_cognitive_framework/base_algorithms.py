"""
基础RL算法实现
=============
包含DQN和SAC的简化实现，作为元认知框架的底层算法
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
    """Q网络"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        return self.net(state)


class DQN:
    """
    Deep Q-Network (DQN)
    
    基础DQN实现，可以被元认知框架包装
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=1e-3, 
                 gamma=0.99, epsilon=1.0, epsilon_decay=0.995, 
                 epsilon_min=0.01, device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.device = device
        
        # Q网络
        self.q_net = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_q_net = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()
        
        self.update_count = 0
        self.target_update_freq = 10
    
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
        计算DQN损失
        
        Returns:
            loss: 损失值
            loss_tensor: 损失张量（用于计算梯度）
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
            next_q_values = self.target_q_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # TD误差
        loss = F.mse_loss(q_values, target_q_values)
        
        return loss, (states, actions, rewards, next_states, dones)
    
    def update(self, batch_size=64, weight=None):
        """
        更新Q网络
        
        Args:
            batch_size: 批次大小
            weight: 元认知权重（可选，由元认知框架提供）
            
        Returns:
            loss: 损失值
            gradients: 梯度（供元认知评价器使用）
        """
        if len(self.replay_buffer) < batch_size:
            return 0.0, None
        
        # 采样批次
        batch = self.replay_buffer.sample(batch_size)
        loss, batch_tensors = self.compute_loss(batch)
        
        # 如果有权重，应用权重
        if weight is not None:
            loss = loss * weight
        
        # 计算梯度
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)  # 保留计算图供元认知使用
        
        # 保存梯度信息
        gradients = [p.grad.clone() if p.grad is not None else None 
                    for p in self.q_net.parameters()]
        
        # 更新参数
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        
        # 更新目标网络
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        # 衰减epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item(), gradients
    
    def get_q_values(self, states):
        """获取Q值（用于计算优势函数）"""
        with torch.no_grad():
            if not isinstance(states, torch.Tensor):
                states = torch.FloatTensor(states).to(self.device)
            q_values = self.q_net(states)
        return q_values


class ActorNetwork(nn.Module):
    """Actor网络（用于SAC）"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)
        log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob


class SAC:
    """
    Soft Actor-Critic (SAC)
    
    基础SAC实现（连续动作空间），可以被元认知框架包装
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=3e-4,
                 gamma=0.99, tau=0.005, alpha=0.2, device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = device
        
        # Actor
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        # Critics
        self.critic1 = QNetwork(state_dim + action_dim, 1, hidden_dim).to(device)
        self.critic2 = QNetwork(state_dim + action_dim, 1, hidden_dim).to(device)
        self.critic1_target = QNetwork(state_dim + action_dim, 1, hidden_dim).to(device)
        self.critic2_target = QNetwork(state_dim + action_dim, 1, hidden_dim).to(device)
        
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr)
        
        self.replay_buffer = ReplayBuffer()
    
    def select_action(self, state, eval_mode=False):
        """选择动作"""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if eval_mode:
                action, _ = self.actor.forward(state)
                action = torch.tanh(action)
            else:
                action, _ = self.actor.sample(state)
            return action.cpu().numpy()[0]
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储转移"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self, batch_size=64, weight=None):
        """更新SAC"""
        if len(self.replay_buffer) < batch_size:
            return 0.0, None
        
        # 采样
        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = batch
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 更新Critic（简化版）
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            next_q1 = self.critic1_target(torch.cat([next_states, next_actions], 1))
            next_q2 = self.critic2_target(torch.cat([next_states, next_actions], 1))
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q
        
        current_q1 = self.critic1(torch.cat([states, actions], 1))
        current_q2 = self.critic2(torch.cat([states, actions], 1))
        
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # 应用权重
        if weight is not None:
            critic_loss = critic_loss * weight
        
        # 更新Critic
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        
        # 保存梯度
        gradients = [p.grad.clone() if p.grad is not None else None 
                    for p in list(self.critic1.parameters()) + list(self.critic2.parameters())]
        
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()
        
        # 软更新目标网络
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return critic_loss.item(), gradients
    
    def get_value(self, states, actions):
        """获取状态-动作价值"""
        with torch.no_grad():
            if not isinstance(states, torch.Tensor):
                states = torch.FloatTensor(states).to(self.device)
            if not isinstance(actions, torch.Tensor):
                actions = torch.FloatTensor(actions).to(self.device)
            q1 = self.critic1(torch.cat([states, actions], -1))
            q2 = self.critic2(torch.cat([states, actions], -1))
            return torch.min(q1, q2)

