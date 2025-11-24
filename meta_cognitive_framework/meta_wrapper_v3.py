"""
元认知框架包装器 v3 - 终极修复版
================================
核心修复："以惊奇度论英雄"而非"以结果论英雄"
"""

import torch
import numpy as np
from curiosity_evaluator import SimplifiedCuriosityEvaluator


class SurpriseDrivenMetaWrapper:
    """
    惊奇驱动的元认知包装器
    
    核心理念转变：
    - 旧：Performance ↓ → w ↓ (晴天送伞，雨天收伞) ❌
    - 新：TD Error ↑ → w ↑ (信息量大的样本优先学习) ✓
    
    灵感来源：Prioritized Experience Replay (PER)
    """
    
    def __init__(self, base_algorithm, state_dim, action_dim, 
                 meta_lr=1e-3, warmup_steps=2000, device='cpu'):
        """
        Args:
            warmup_steps: 预热步数（让base算法先稳定）
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
        
        # 🔧 新增：追踪TD Error和Loss
        self.td_error_history = []
        self.running_avg_td_error = 1.0  # 运行平均TD误差
        self.running_avg_loss = 1.0      # 运行平均损失
        
        # 性能追踪（辅助信息）
        self.performance_window = []
        
    def select_action(self, state, eval_mode=False):
        return self.base_algorithm.select_action(state, eval_mode)
    
    def store_transition(self, state, action, reward, next_state, done):
        self.base_algorithm.store_transition(state, action, reward, next_state, done)
    
    def add_episode_reward(self, reward):
        """记录回合奖励"""
        self.episode_rewards.append(reward)
        self.performance_window.append(reward)
        
        if len(self.performance_window) > 20:
            self.performance_window.pop(0)
        if len(self.episode_rewards) > 100:
            self.episode_rewards.pop(0)
    
    def _compute_target_weight_v3(self, td_error, current_performance):
        """
        🎯 核心修复：基于"惊奇度"而非"结果"
        
        新逻辑：
        1. 高TD Error = 高信息量 = 应该多学习
        2. 但要防止异常值破坏训练
        3. 结合性能趋势作为辅助
        
        Args:
            td_error: 当前批次的TD误差
            current_performance: 最近的性能（可选）
            
        Returns:
            target_w: 目标权重 [0.5, 2.0]
        """
        # 归一化TD Error
        normalized_td = td_error / (self.running_avg_td_error + 1e-6)
        
        # 更新运行平均（缓慢更新）
        self.running_avg_td_error = 0.99 * self.running_avg_td_error + 0.01 * td_error
        
        # 🔧 方案 A: PER风格 - 简单有效
        # TD Error 越大，权重越高
        # 使用tanh平滑，避免极端值
        surprise_bonus = 0.5 * np.tanh(normalized_td - 1.0)
        target_w = 1.0 + surprise_bonus
        
        # 限制范围
        target_w = np.clip(target_w, 0.5, 2.0)
        
        return target_w
    
    def _compute_target_weight_v3_advanced(self, td_error, batch_loss, current_performance):
        """
        🎯 高级版本：混合逻辑
        
        考虑三个因素：
        1. TD Error (信息量)
        2. Loss (学习难度)
        3. Performance (结果质量)
        """
        # 1. 归一化TD Error
        normalized_td = td_error / (self.running_avg_td_error + 1e-6)
        self.running_avg_td_error = 0.99 * self.running_avg_td_error + 0.01 * td_error
        
        # 2. 归一化Loss
        normalized_loss = batch_loss / (self.running_avg_loss + 1e-6)
        self.running_avg_loss = 0.99 * self.running_avg_loss + 0.01 * batch_loss
        
        # 3. 计算性能趋势
        if len(self.performance_window) >= 10:
            recent_perf = np.mean(self.performance_window[-5:])
            older_perf = np.mean(self.performance_window[-10:-5])
            perf_trend = (recent_perf - older_perf) / (abs(older_perf) + 1)
        else:
            perf_trend = 0
        
        # 🎯 混合决策逻辑
        # 
        # 情况1: 高TD Error + 性能还行 → 高价值样本 (High Leverage)
        #   - 这些样本信息量大，且不是纯噪音
        #   - w = 1.5 - 2.0
        #
        # 情况2: 高TD Error + 性能很差 → 可能太难或噪音
        #   - 先不急着学，或者适度学习
        #   - w = 0.8 - 1.2
        #
        # 情况3: 低TD Error → 已经学会了
        #   - 降低权重，节省计算/防止过拟合
        #   - w = 0.5 - 0.8
        
        # 基础权重：基于TD Error
        base_w = 1.0 + 0.5 * np.tanh(normalized_td - 1.0)
        
        # 调整：基于性能趋势
        if normalized_td > 1.5:  # 高TD Error
            if perf_trend > 0:  # 性能在上升
                # 高杠杆样本！加强学习
                adjustment = 0.3
            else:  # 性能在下降
                # 可能是难样本或噪音，适度学习
                adjustment = -0.1
        else:  # 低TD Error
            # 已经学会的样本，降低权重
            adjustment = -0.2
        
        target_w = base_w + adjustment
        
        # 限制范围
        target_w = np.clip(target_w, 0.5, 2.0)
        
        return target_w
    
    def update(self, batch_size=64):
        """
        🚀 元认知增强的更新流程
        
        Returns:
            dict: 更新统计信息
        """
        if len(self.base_algorithm.replay_buffer) < batch_size:
            return {
                'base_loss': 0, 
                'meta_loss': 0, 
                'weight': 1.0, 
                'td_error': 0,
                'in_warmup': True
            }
        
        self.total_updates += 1
        
        # Warmup期：只更新base算法
        if self.total_updates < self.warmup_steps:
            base_loss, td_error = self.base_algorithm.update(batch_size)
            return {
                'base_loss': base_loss,
                'meta_loss': 0,
                'weight': 1.0,
                'td_error': td_error,
                'in_warmup': True
            }
        
        # 渐进式引入元认知
        progress = min(1.0, (self.total_updates - self.warmup_steps) / 2000)
        
        # 1. 第一次更新：获取TD Error（不应用权重）
        base_loss_1, td_error = self.base_algorithm.update(batch_size, weight=None)
        
        # 2. 计算目标权重（基于TD Error）
        current_perf = np.mean(self.performance_window[-5:]) if len(self.performance_window) >= 5 else 0
        
        # 使用简单版本（PER风格）
        target_w = self._compute_target_weight_v3(td_error, current_perf)
        
        # 3. 训练元评价器
        meta_loss = 0
        pred_weight = 1.0
        
        if len(self.episode_rewards) > 5:  # 有足够数据
            # 采样批次
            batch = self.base_algorithm.replay_buffer.sample(batch_size)
            states = torch.FloatTensor(batch[0]).to(self.device)
            actions = torch.FloatTensor(batch[1]).to(self.device)
            if actions.dim() == 1:
                actions = actions.unsqueeze(1)
            
            # 目标权重（所有样本用同一个权重，简化版）
            target_weights = torch.full((batch_size,), target_w, device=self.device)
            
            # 更新元评价器
            meta_loss = self.curiosity_evaluator.update(states, actions, target_weights)
            
            # 获取预测权重
            pred_weights = self.curiosity_evaluator.evaluate_batch_value(states, actions)
            pred_weight = pred_weights.mean().item()
            
            # 渐进式应用
            pred_weight = 1.0 * (1 - progress) + pred_weight * progress
        
        # 记录TD Error
        self.td_error_history.append(td_error)
        if len(self.td_error_history) > 1000:
            self.td_error_history.pop(0)
        
        return {
            'base_loss': base_loss_1,
            'meta_loss': meta_loss,
            'weight': pred_weight,
            'target_weight': target_w,
            'td_error': td_error,
            'normalized_td': td_error / (self.running_avg_td_error + 1e-6),
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
            'avg_td_error': np.mean(self.td_error_history) if self.td_error_history else 0,
            'running_avg_td': self.running_avg_td_error
        }

