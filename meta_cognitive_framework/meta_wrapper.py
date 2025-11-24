"""
元认知框架包装器
================
将好奇心评价器作为元优化器，包装任何基础RL算法
"""

import torch
import numpy as np
from curiosity_evaluator import SimplifiedCuriosityEvaluator


class MetaCognitiveWrapper:
    """
    元认知框架包装器
    
    将好奇心作为元认知能力，引导底层RL算法的学习过程
    
    工作流程：
    1. 从回放池采样一个批次
    2. 计算底层算法的损失和梯度
    3. 用好奇心评价器评估这个批次的学习价值 w
    4. 用权重 w 对损失进行加权：L_total = w * L_base
    5. 更新好奇心评价器：让 w 拟合优势函数的提升
    """
    
    def __init__(self, base_algorithm, state_dim, action_dim, 
                 meta_lr=1e-3, device='cpu'):
        """
        Args:
            base_algorithm: 底层RL算法（DQN, SAC等）
            state_dim: 状态维度
            action_dim: 动作维度
            meta_lr: 元学习器学习率
            device: 运行设备
        """
        self.base_algorithm = base_algorithm
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # 好奇心评价器（使用简化版）
        self.curiosity_evaluator = SimplifiedCuriosityEvaluator(
            state_dim=state_dim,
            action_dim=action_dim if hasattr(base_algorithm, 'actor') else 1,  # 离散动作用1维
            hidden_dim=128,
            lr=meta_lr,
            device=device
        )
        
        # 用于计算优势函数提升的历史
        self.value_history = {}  # {state_key: old_value}
        self.advantage_buffer = []  # 存储优势提升用于训练
        
    def select_action(self, state, eval_mode=False):
        """选择动作（直接调用底层算法）"""
        return self.base_algorithm.select_action(state, eval_mode)
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储转移（直接调用底层算法）"""
        self.base_algorithm.store_transition(state, action, reward, next_state, done)
    
    def _compute_advantage_improvement(self, states, actions):
        """
        计算优势函数的提升
        
        Target_w = clip(A_new - A_old, 0, 2)
        
        这里简化处理：使用Q值或V值的变化作为优势提升的代理
        """
        batch_size = states.shape[0]
        improvements = []
        
        # 获取当前值估计
        if hasattr(self.base_algorithm, 'get_q_values'):
            # DQN: 使用Q值
            current_values = self.base_algorithm.get_q_values(states)
            if actions.dim() > 1:
                actions_idx = actions.argmax(dim=1)
            else:
                actions_idx = actions.long()
            current_values = current_values.gather(1, actions_idx.unsqueeze(1)).squeeze(1)
        elif hasattr(self.base_algorithm, 'get_value'):
            # SAC: 使用V值
            current_values = self.base_algorithm.get_value(states, actions).squeeze(1)
        else:
            # 默认：返回中性权重
            return torch.ones(batch_size, device=self.device)
        
        # 对于每个状态，计算与历史值的差异
        for i in range(batch_size):
            state_key = hash(states[i].cpu().numpy().tobytes())
            
            if state_key in self.value_history:
                old_value = self.value_history[state_key]
                improvement = current_values[i].item() - old_value
                # Clip到[0, 2]
                improvement = max(0, min(2, improvement + 1.0))  # +1使其中心在1.0
            else:
                improvement = 1.0  # 新状态，中性权重
            
            improvements.append(improvement)
            # 更新历史值
            self.value_history[state_key] = current_values[i].item()
        
        return torch.tensor(improvements, device=self.device)
    
    def update(self, batch_size=64):
        """
        元认知增强的更新
        
        Returns:
            dict: 更新统计信息
        """
        if len(self.base_algorithm.replay_buffer) < batch_size:
            return {'base_loss': 0, 'meta_loss': 0, 'avg_weight': 1.0}
        
        # 1. 采样批次
        batch = self.base_algorithm.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = batch
        
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        if actions_tensor.dim() == 1:
            actions_tensor = actions_tensor.unsqueeze(1)
        
        # 2. 第一次前向：计算底层损失和梯度（但不更新）
        # 这里为了简化，我们让底层算法先进行一次常规更新
        base_loss, gradients = self.base_algorithm.update(batch_size, weight=None)
        
        # 3. 用好奇心评价器评估批次价值
        batch_weights = self.curiosity_evaluator.evaluate_batch_value(
            states_tensor, actions_tensor
        )
        avg_weight = batch_weights.mean().item()
        
        # 4. 使用加权损失进行第二次更新
        # 注意：这里简化了流程，实际上应该在一次更新中完成
        # 但为了清晰展示概念，我们分两步
        if avg_weight > 0.5:  # 只有当权重较高时才进行加权更新
            weighted_loss, _ = self.base_algorithm.update(batch_size, weight=avg_weight)
        else:
            weighted_loss = base_loss
        
        # 5. 更新好奇心评价器
        # 计算目标权重（基于优势提升）
        target_weights = self._compute_advantage_improvement(states_tensor, actions_tensor)
        
        meta_loss = self.curiosity_evaluator.update(
            states_tensor, actions_tensor, target_weights
        )
        
        # 清理历史（避免内存溢出）
        if len(self.value_history) > 10000:
            # 随机删除一半
            keys = list(self.value_history.keys())
            for key in keys[:len(keys)//2]:
                del self.value_history[key]
        
        return {
            'base_loss': base_loss,
            'weighted_loss': weighted_loss,
            'meta_loss': meta_loss,
            'avg_weight': avg_weight,
            'weight_std': batch_weights.std().item()
        }
    
    def get_stats(self):
        """获取统计信息"""
        return {
            'value_history_size': len(self.value_history),
            'epsilon': getattr(self.base_algorithm, 'epsilon', None)
        }


class SimpleMetaWrapper:
    """
    简化版元认知包装器
    
    更加直观的实现，适合快速验证概念
    """
    
    def __init__(self, base_algorithm, state_dim, action_dim, device='cpu'):
        self.base_algorithm = base_algorithm
        self.device = device
        
        # 简化的评价器：只评估状态新颖性
        self.curiosity_evaluator = SimplifiedCuriosityEvaluator(
            state_dim=state_dim,
            action_dim=1 if not hasattr(base_algorithm, 'actor') else action_dim,
            hidden_dim=64,
            lr=1e-3,
            device=device
        )
        
        # 性能追踪
        self.episode_rewards = []
        self.recent_performance = 0
        
    def select_action(self, state, eval_mode=False):
        return self.base_algorithm.select_action(state, eval_mode)
    
    def store_transition(self, state, action, reward, next_state, done):
        self.base_algorithm.store_transition(state, action, reward, next_state, done)
    
    def update(self, batch_size=64):
        """简化的更新流程"""
        if len(self.base_algorithm.replay_buffer) < batch_size:
            return {'base_loss': 0, 'meta_loss': 0}
        
        # 常规更新
        base_loss, _ = self.base_algorithm.update(batch_size)
        
        # 定期更新好奇心评价器（基于性能提升）
        meta_loss = 0
        if len(self.episode_rewards) > 10:
            recent_perf = np.mean(self.episode_rewards[-10:])
            improvement = max(0, min(2, (recent_perf - self.recent_performance) / 100 + 1.0))
            self.recent_performance = recent_perf
            
            # 简化训练：用性能提升作为全局权重目标
            batch = self.base_algorithm.replay_buffer.sample(batch_size)
            states = torch.FloatTensor(batch[0]).to(self.device)
            actions = torch.FloatTensor(batch[1]).to(self.device)
            if actions.dim() == 1:
                actions = actions.unsqueeze(1)
            
            target = torch.full((batch_size,), improvement, device=self.device)
            meta_loss = self.curiosity_evaluator.update(states, actions, target)
        
        return {'base_loss': base_loss, 'meta_loss': meta_loss}
    
    def add_episode_reward(self, reward):
        """添加回合奖励"""
        self.episode_rewards.append(reward)
        if len(self.episode_rewards) > 100:
            self.episode_rewards.pop(0)

