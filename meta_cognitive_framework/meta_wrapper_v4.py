"""
元认知框架包装器 v4 - 激进版
================================
核心升级：从"温和镇静剂"到"强力加速器"

关键修复：
1. 基于Z-Score的批次内归一化 → 高方差权重分布
2. 非线性放大困难样本 → 真正的好奇心加速
3. 样本级别权重 → 精细控制
"""

import torch
import numpy as np
from curiosity_evaluator import SimplifiedCuriosityEvaluator


class AggressiveMetaWrapper:
    """
    激进版元认知包装器
    
    核心改进：
    - ✓ Z-Score归一化：保证批次内高方差
    - ✓ 样本级权重：精细到每个样本
    - ✓ 非线性放大：困难样本指数级关注
    """
    
    def __init__(self, base_algorithm, state_dim, action_dim, 
                 meta_lr=1e-3, warmup_steps=2000, 
                 scale_factor=0.5, use_exponential=False,
                 device='cpu'):
        """
        Args:
            scale_factor: 权重分布的激进程度 (0.3-1.0)
                - 0.3: 温和 (权重范围 [0.7, 1.3])
                - 0.5: 中等 (权重范围 [0.5, 1.5])
                - 1.0: 激进 (权重范围 [0.0, 2.0])
            use_exponential: 是否使用指数放大
        """
        self.base_algorithm = base_algorithm
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.warmup_steps = warmup_steps
        self.scale_factor = scale_factor
        self.use_exponential = use_exponential
        
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
        
        # TD Error追踪
        self.td_error_history = []
        self.running_max_td_error = 1.0  # 运行最大TD误差（用于方案B）
        self.running_avg_td_error = 1.0  # 运行平均（用于监控）
        
        # 权重统计（用于监控）
        self.weight_history = []
        
    def select_action(self, state, eval_mode=False):
        return self.base_algorithm.select_action(state, eval_mode)
    
    def store_transition(self, state, action, reward, next_state, done):
        self.base_algorithm.store_transition(state, action, reward, next_state, done)
    
    def add_episode_reward(self, reward):
        """记录回合奖励"""
        self.episode_rewards.append(reward)
        if len(self.episode_rewards) > 100:
            self.episode_rewards.pop(0)
    
    def _compute_sample_td_errors(self, batch):
        """
        计算批次中每个样本的TD Error
        
        Returns:
            td_errors: [batch_size] 每个样本的TD误差
        """
        states, actions, rewards, next_states, dones = batch
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        with torch.no_grad():
            # 当前Q值
            q_values = self.base_algorithm.q_net(states)
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # 目标Q值
            if hasattr(self.base_algorithm, 'use_double_dqn') and self.base_algorithm.use_double_dqn:
                next_actions = self.base_algorithm.q_net(next_states).argmax(1)
                next_q_values = self.base_algorithm.target_q_net(next_states).gather(
                    1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q_values = self.base_algorithm.target_q_net(next_states).max(1)[0]
            
            target_q_values = rewards + (1 - dones) * self.base_algorithm.gamma * next_q_values
            
            # TD Error（绝对值）
            td_errors = torch.abs(q_values - target_q_values)
        
        return td_errors
    
    def _compute_target_weights_zscore(self, batch_td_errors):
        """
        🎯 方案A: 基于Z-Score的强力归一化
        
        核心思想：
        - 在批次内部制造高对比度
        - 高TD Error样本 → w > 1 (加强学习)
        - 低TD Error样本 → w < 1 (快速跳过)
        - 平均权重 ≈ 1.0 (不改变整体学习率)
        
        Args:
            batch_td_errors: [batch_size] TD误差
            
        Returns:
            target_weights: [batch_size] 目标权重
        """
        # 1. 计算批次统计量
        mean = batch_td_errors.mean()
        std = batch_td_errors.std() + 1e-6
        
        # 2. Z-Score标准化
        # Z值通常在[-2, 2]范围内
        z_scores = (batch_td_errors - mean) / std
        
        # 3. 映射到权重
        # scale_factor控制激进程度
        # scale_factor=0.5: z=2时w=2.0, z=-2时w=0.0
        # scale_factor=1.0: z=1时w=2.0, z=-1时w=0.0
        target_weights = 1.0 + z_scores * self.scale_factor
        
        # 4. 裁剪到安全范围
        target_weights = torch.clamp(target_weights, 0.1, 2.0)
        
        return target_weights
    
    def _compute_target_weights_exponential(self, batch_td_errors):
        """
        🎯 方案B: 非线性放大（好奇心加速）
        
        核心思想：
        - 对高TD Error样本给予指数级奖励
        - 模拟真正的"好奇心"驱动
        
        Args:
            batch_td_errors: [batch_size] TD误差
            
        Returns:
            target_weights: [batch_size] 目标权重
        """
        # 归一化到[0, 1]
        normalized = batch_td_errors / (self.running_max_td_error + 1e-6)
        normalized = torch.clamp(normalized, 0, 1)
        
        # 更新运行最大值
        current_max = batch_td_errors.max().item()
        self.running_max_td_error = 0.99 * self.running_max_td_error + 0.01 * current_max
        
        # 指数级关注困难样本
        # normalized=1.0 (最大误差) → w=2.0
        # normalized=0.5 → w=1.25
        # normalized=0.0 → w=1.0
        target_weights = 1.0 + (normalized ** 2)
        
        # 裁剪
        target_weights = torch.clamp(target_weights, 0.5, 2.0)
        
        return target_weights
    
    def _compute_target_weights_hybrid(self, batch_td_errors):
        """
        🎯 方案C: 混合策略（Z-Score + 指数放大）
        
        结合两者优点：
        - Z-Score保证批次内对比度
        - 指数放大给困难样本额外奖励
        """
        # 1. Z-Score基础权重
        mean = batch_td_errors.mean()
        std = batch_td_errors.std() + 1e-6
        z_scores = (batch_td_errors - mean) / std
        base_weights = 1.0 + z_scores * 0.4  # 稍微保守一点
        
        # 2. 指数加成（给真正困难的样本额外boost）
        normalized = batch_td_errors / (self.running_max_td_error + 1e-6)
        normalized = torch.clamp(normalized, 0, 1)
        curiosity_bonus = 0.3 * (normalized ** 2)  # 最多+0.3
        
        # 3. 组合
        target_weights = base_weights + curiosity_bonus
        
        # 4. 裁剪
        target_weights = torch.clamp(target_weights, 0.1, 2.0)
        
        return target_weights
    
    def update(self, batch_size=64):
        """
        🚀 样本级别的元认知增强更新
        
        Returns:
            dict: 详细的更新统计
        """
        if len(self.base_algorithm.replay_buffer) < batch_size:
            return {
                'base_loss': 0,
                'meta_loss': 0,
                'avg_weight': 1.0,
                'weight_std': 0,
                'in_warmup': True
            }
        
        self.total_updates += 1
        
        # Warmup期
        if self.total_updates < self.warmup_steps:
            base_loss, avg_td = self.base_algorithm.update(batch_size)
            return {
                'base_loss': base_loss,
                'meta_loss': 0,
                'avg_weight': 1.0,
                'weight_std': 0,
                'td_error': avg_td,
                'in_warmup': True
            }
        
        # 渐进式引入元认知
        progress = min(1.0, (self.total_updates - self.warmup_steps) / 2000)
        
        # 1. 采样批次
        batch = self.base_algorithm.replay_buffer.sample(batch_size)
        
        # 2. 计算每个样本的TD Error
        sample_td_errors = self._compute_sample_td_errors(batch)
        
        # 3. 计算目标权重（使用选定的策略）
        if self.use_exponential:
            target_weights = self._compute_target_weights_exponential(sample_td_errors)
        else:
            target_weights = self._compute_target_weights_zscore(sample_td_errors)
        
        # 可选：使用混合策略
        # target_weights = self._compute_target_weights_hybrid(sample_td_errors)
        
        # 4. 正常更新base算法（不加权，用于获取基础损失）
        base_loss, avg_td = self.base_algorithm.update(batch_size, weight=None)
        
        # 5. 训练元评价器
        meta_loss = 0
        pred_weights = target_weights  # 初始化
        
        if self.total_updates > self.warmup_steps + 500:  # 给点额外warmup
            states = torch.FloatTensor(batch[0]).to(self.device)
            actions = torch.FloatTensor(batch[1]).to(self.device)
            if actions.dim() == 1:
                actions = actions.unsqueeze(1)
            
            # 训练评价器拟合目标权重
            meta_loss = self.curiosity_evaluator.update(states, actions, target_weights)
            
            # 获取预测权重
            pred_weights = self.curiosity_evaluator.evaluate_batch_value(states, actions)
            
            # 渐进式应用（后期可以完全使用预测权重）
            if progress < 0.5:
                # 早期：主要用目标权重
                final_weights = target_weights * (1 - progress * 2) + pred_weights * (progress * 2)
            else:
                # 后期：主要用预测权重
                final_weights = pred_weights
        else:
            final_weights = target_weights
        
        # 6. 记录统计
        avg_weight = final_weights.mean().item()
        weight_std = final_weights.std().item()
        self.weight_history.append({'mean': avg_weight, 'std': weight_std})
        
        self.td_error_history.append(avg_td)
        if len(self.td_error_history) > 1000:
            self.td_error_history.pop(0)
        
        self.running_avg_td_error = 0.99 * self.running_avg_td_error + 0.01 * avg_td
        
        return {
            'base_loss': base_loss,
            'meta_loss': meta_loss,
            'avg_weight': avg_weight,
            'weight_std': weight_std,
            'weight_min': final_weights.min().item(),
            'weight_max': final_weights.max().item(),
            'td_error': avg_td,
            'target_weight_mean': target_weights.mean().item(),
            'target_weight_std': target_weights.std().item(),
            'in_warmup': False,
            'progress': progress
        }
    
    def get_stats(self):
        """获取详细统计"""
        stats = {
            'total_updates': self.total_updates,
            'in_warmup': self.total_updates < self.warmup_steps,
            'num_episodes': len(self.episode_rewards),
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'avg_td_error': np.mean(self.td_error_history) if self.td_error_history else 0,
        }
        
        if len(self.weight_history) > 0:
            recent_weights = self.weight_history[-100:]
            stats['avg_weight_mean'] = np.mean([w['mean'] for w in recent_weights])
            stats['avg_weight_std'] = np.mean([w['std'] for w in recent_weights])
        
        return stats


class UltraAggressiveMetaWrapper(AggressiveMetaWrapper):
    """
    超激进版本 - 进一步放大权重差异
    
    适用场景：
    - Base算法已经很稳定
    - 想要最大化元认知的影响
    - 愿意承担一定风险
    """
    
    def __init__(self, base_algorithm, state_dim, action_dim, 
                 meta_lr=1e-3, warmup_steps=2000, device='cpu'):
        super().__init__(
            base_algorithm=base_algorithm,
            state_dim=state_dim,
            action_dim=action_dim,
            meta_lr=meta_lr,
            warmup_steps=warmup_steps,
            scale_factor=1.0,  # 更激进！
            use_exponential=False,
            device=device
        )
    
    def _compute_target_weights_zscore(self, batch_td_errors):
        """超激进版Z-Score"""
        mean = batch_td_errors.mean()
        std = batch_td_errors.std() + 1e-6
        z_scores = (batch_td_errors - mean) / std
        
        # 更激进的映射
        # z=2 → w=2.0, z=-2 → w=0.0
        target_weights = 1.0 + z_scores * 1.0  # scale_factor=1.0
        
        # 稍微宽松的裁剪
        target_weights = torch.clamp(target_weights, 0.05, 2.0)
        
        return target_weights

