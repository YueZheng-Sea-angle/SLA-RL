"""
元认知框架包装器 v2
==================
改进版本，解决训练不稳定的问题
"""

import torch
import numpy as np
from curiosity_evaluator import SimplifiedCuriosityEvaluator


class ImprovedMetaWrapper:
    """
    改进的元认知包装器
    
    主要改进：
    1. 添加warmup期，让base算法先学习
    2. 渐进式引入元认知权重
    3. 更稳健的优势估计
    4. 权重clip和平滑
    """
    
    def __init__(self, base_algorithm, state_dim, action_dim, 
                 meta_lr=1e-3, warmup_steps=5000, device='cpu'):
        """
        Args:
            warmup_steps: 预热步数，期间不使用元认知
        """
        self.base_algorithm = base_algorithm
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.warmup_steps = warmup_steps
        
        # 好奇心评价器
        self.curiosity_evaluator = SimplifiedCuriosityEvaluator(
            state_dim=state_dim,
            action_dim=1 if not hasattr(base_algorithm, 'actor') else action_dim,
            hidden_dim=64,
            lr=meta_lr,
            device=device
        )
        
        # 训练统计
        self.total_updates = 0
        self.episode_rewards = []
        self.value_history = {}
        
        # 性能追踪（用于计算目标权重）
        self.performance_window = []
        self.baseline_performance = -1000  # 初始基线
        
    def select_action(self, state, eval_mode=False):
        return self.base_algorithm.select_action(state, eval_mode)
    
    def store_transition(self, state, action, reward, next_state, done):
        self.base_algorithm.store_transition(state, action, reward, next_state, done)
    
    def add_episode_reward(self, reward):
        """记录回合奖励"""
        self.episode_rewards.append(reward)
        self.performance_window.append(reward)
        
        # 保持窗口大小
        if len(self.performance_window) > 20:
            self.performance_window.pop(0)
        if len(self.episode_rewards) > 100:
            self.episode_rewards.pop(0)
    
    def _compute_target_weight(self):
        """
        计算目标权重（基于性能提升）
        
        改进：使用滑动窗口的性能提升
        """
        if len(self.performance_window) < 5:
            return 1.0  # 初期返回中性权重
        
        # 当前性能（最近5个episode的均值）
        current_perf = np.mean(self.performance_window[-5:])
        
        # 计算相对于基线的提升
        if self.baseline_performance == -1000:
            self.baseline_performance = current_perf
            return 1.0
        
        improvement = current_perf - self.baseline_performance
        
        # 归一化到[0, 2]
        # improvement > 0 时，权重 > 1（鼓励学习）
        # improvement < 0 时，权重 < 1（减少学习）
        target_w = 1.0 + np.tanh(improvement / 50.0)  # 使用tanh平滑
        target_w = np.clip(target_w, 0.3, 1.7)  # 限制范围，避免极端值
        
        # 缓慢更新基线
        self.baseline_performance = 0.95 * self.baseline_performance + 0.05 * current_perf
        
        return target_w
    
    def update(self, batch_size=64):
        """改进的更新流程"""
        if len(self.base_algorithm.replay_buffer) < batch_size:
            return {'base_loss': 0, 'meta_loss': 0, 'weight': 1.0, 'in_warmup': True}
        
        self.total_updates += 1
        
        # Warmup期：只更新base算法
        if self.total_updates < self.warmup_steps:
            base_loss, _ = self.base_algorithm.update(batch_size)
            return {
                'base_loss': base_loss,
                'meta_loss': 0,
                'weight': 1.0,
                'in_warmup': True
            }
        
        # 渐进式引入元认知（warmup后逐渐增加影响）
        progress = min(1.0, (self.total_updates - self.warmup_steps) / 2000)
        
        # 1. 正常更新base算法
        base_loss, _ = self.base_algorithm.update(batch_size)
        
        # 2. 评估批次并训练元评价器
        meta_loss = 0
        avg_weight = 1.0
        
        if len(self.episode_rewards) > 10:  # 有足够的性能数据
            # 采样一个批次用于元训练
            batch = self.base_algorithm.replay_buffer.sample(batch_size)
            states = torch.FloatTensor(batch[0]).to(self.device)
            actions = torch.FloatTensor(batch[1]).to(self.device)
            if actions.dim() == 1:
                actions = actions.unsqueeze(1)
            
            # 计算目标权重
            target_w = self._compute_target_weight()
            target_weights = torch.full((batch_size,), target_w, device=self.device)
            
            # 更新元评价器
            meta_loss = self.curiosity_evaluator.update(states, actions, target_weights)
            
            # 获取当前批次的预测权重
            pred_weights = self.curiosity_evaluator.evaluate_batch_value(states, actions)
            avg_weight = pred_weights.mean().item()
            
            # 渐进式应用权重（避免突变）
            avg_weight = 1.0 * (1 - progress) + avg_weight * progress
        
        return {
            'base_loss': base_loss,
            'meta_loss': meta_loss,
            'weight': avg_weight,
            'target_weight': target_w if len(self.episode_rewards) > 10 else 1.0,
            'in_warmup': False,
            'progress': progress
        }
    
    def get_stats(self):
        """获取统计信息"""
        return {
            'total_updates': self.total_updates,
            'in_warmup': self.total_updates < self.warmup_steps,
            'num_episodes': len(self.episode_rewards),
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'baseline': self.baseline_performance
        }


class MinimalMetaWrapper:
    """
    最小化元认知包装器
    
    极简实现，专注于核心概念验证
    策略：暂时禁用元认知，先让base算法正常学习
    """
    
    def __init__(self, base_algorithm, state_dim, action_dim, device='cpu'):
        self.base_algorithm = base_algorithm
        self.device = device
        self.episode_rewards = []
        
        # 暂时禁用元认知
        self.meta_enabled = False
        
    def select_action(self, state, eval_mode=False):
        return self.base_algorithm.select_action(state, eval_mode)
    
    def store_transition(self, state, action, reward, next_state, done):
        self.base_algorithm.store_transition(state, action, reward, next_state, done)
    
    def add_episode_reward(self, reward):
        self.episode_rewards.append(reward)
        
    def update(self, batch_size=64):
        """简单转发到base算法"""
        if len(self.base_algorithm.replay_buffer) < batch_size:
            return {'base_loss': 0, 'meta_loss': 0}
        
        base_loss, _ = self.base_algorithm.update(batch_size)
        
        return {
            'base_loss': base_loss,
            'meta_loss': 0  # 元认知被禁用
        }

