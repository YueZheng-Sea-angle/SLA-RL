"""
双策略PPO算法实现
核心创新：探索-利用-稳定性统一框架
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
from typing import Tuple, Dict, Optional


class ActorCriticNetwork(nn.Module):
    """Actor-Critic网络，支持离散和连续动作空间"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, 
                 continuous: bool = False):
        super(ActorCriticNetwork, self).__init__()
        self.continuous = continuous
        
        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        
        # Actor头
        if continuous:
            # 连续动作空间：输出均值和标准差
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        else:
            # 离散动作空间：输出概率分布
            self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic头
        self.critic = nn.Linear(hidden_dim, 1)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """正交初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        """
        前向传播
        返回: (动作分布, 状态价值)
        """
        features = self.shared(state)
        
        # Actor输出
        if self.continuous:
            mean = self.actor_mean(features)
            std = torch.exp(self.actor_logstd).expand_as(mean)
            dist = Normal(mean, std)
        else:
            logits = self.actor(features)
            dist = Categorical(logits=logits)
        
        # Critic输出
        value = self.critic(features)
        
        return dist, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """获取动作"""
        dist, value = self.forward(state)
        
        if deterministic:
            if self.continuous:
                action = dist.mean
            else:
                action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        if self.continuous:
            log_prob = log_prob.sum(dim=-1)
        
        return action, log_prob, value
    
    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor):
        """评估动作"""
        dist, value = self.forward(state)
        
        log_prob = dist.log_prob(action)
        if self.continuous:
            log_prob = log_prob.sum(dim=-1)
        
        entropy = dist.entropy()
        if self.continuous:
            entropy = entropy.sum(dim=-1)
        
        return log_prob, value, entropy


class CuriosityModule(nn.Module):
    """
    可控性好奇心模块
    使用前向动力学模型和逆向动力学模型
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 continuous: bool = False):
        super(CuriosityModule, self).__init__()
        self.continuous = continuous
        self.action_dim = action_dim
        
        # 特征编码器
        self.feature_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # 前向动力学模型: (s_t, a_t) -> s_{t+1}
        action_input_dim = action_dim if continuous else action_dim
        self.forward_model = nn.Sequential(
            nn.Linear(hidden_dim // 2 + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # 逆向动力学模型: (s_t, s_{t+1}) -> a_t
        self.inverse_model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor, 
                next_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算好奇心奖励
        返回: (好奇心奖励, 前向损失, 逆向损失)
        """
        # 编码状态特征
        phi_state = self.feature_encoder(state)
        phi_next_state = self.feature_encoder(next_state)
        
        # 处理动作（如果是离散的，需要one-hot编码）
        if not self.continuous:
            # 离散动作空间：转换为one-hot编码
            action_input = F.one_hot(action.long(), num_classes=self.action_dim)
            action_input = action_input.float()
        else:
            # 连续动作空间：直接使用
            action_input = action
        
        # 前向模型预测下一状态特征
        forward_input = torch.cat([phi_state, action_input], dim=-1)
        pred_next_state = self.forward_model(forward_input)
        
        # 前向预测误差作为好奇心（可控性）
        forward_loss = F.mse_loss(pred_next_state, phi_next_state.detach(), reduction='none').mean(dim=-1)
        curiosity_reward = forward_loss  # 预测误差越大，好奇心越强
        
        # 逆向模型预测动作
        inverse_input = torch.cat([phi_state, phi_next_state], dim=-1)
        pred_action = self.inverse_model(inverse_input)
        
        # 逆向损失
        if self.continuous:
            inverse_loss = F.mse_loss(pred_action, action)
        else:
            inverse_loss = F.cross_entropy(pred_action, action.long())
        
        return curiosity_reward, forward_loss.mean(), inverse_loss


class DualPolicyPPO:
    """
    双策略PPO算法
    创新点：探索-利用-稳定性统一框架
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 continuous: bool = False,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 intrinsic_coef: float = 0.1,
                 kl_coef: float = 0.01,
                 kl_threshold: float = 0.3,
                 update_opponent_interval: int = 10,
                 hidden_dim: int = 256,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.intrinsic_coef = intrinsic_coef
        self.kl_coef = kl_coef
        self.kl_threshold = kl_threshold
        self.update_opponent_interval = update_opponent_interval
        self.device = device
        
        # 主策略网络
        self.actor_critic = ActorCriticNetwork(
            state_dim, action_dim, hidden_dim, continuous
        ).to(device)
        
        # 对手策略网络（保守的锚点）
        self.opponent_actor_critic = ActorCriticNetwork(
            state_dim, action_dim, hidden_dim, continuous
        ).to(device)
        
        # 初始化对手策略为主策略的副本
        self.opponent_actor_critic.load_state_dict(self.actor_critic.state_dict())
        
        # 好奇心模块
        self.curiosity = CuriosityModule(
            state_dim, action_dim, hidden_dim, continuous
        ).to(device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            list(self.actor_critic.parameters()) + list(self.curiosity.parameters()),
            lr=lr
        )
        
        # 训练统计
        self.update_count = 0
        self.performance_history = []
        
    def compute_kl_divergence(self, states: torch.Tensor) -> torch.Tensor:
        """
        计算主策略和对手策略之间的KL散度
        """
        with torch.no_grad():
            dist_opponent, _ = self.opponent_actor_critic(states)
        
        dist_actor, _ = self.actor_critic(states)
        
        # 计算KL(π_actor || π_opponent)
        if self.continuous:
            kl = torch.distributions.kl_divergence(dist_actor, dist_opponent).sum(dim=-1)
        else:
            kl = torch.distributions.kl_divergence(dist_actor, dist_opponent)
        
        return kl
    
    def compute_intrinsic_reward(self, 
                                  states: torch.Tensor,
                                  actions: torch.Tensor,
                                  next_states: torch.Tensor) -> torch.Tensor:
        """
        计算改进的内在奖励
        r_intrinsic = Curiosity(s, a) * [1 - KL(π_actor || π_opponent)]
        """
        # 好奇心奖励
        curiosity_reward, _, _ = self.curiosity(states, actions, next_states)
        
        # KL散度（归一化到[0, 1]）
        kl_div = self.compute_kl_divergence(states)
        kl_penalty = torch.clamp(kl_div / self.kl_threshold, 0, 1)
        
        # 组合内在奖励
        intrinsic_reward = curiosity_reward * (1 - kl_penalty)
        
        return intrinsic_reward.detach()
    
    def compute_gae(self, 
                    rewards: torch.Tensor,
                    values: torch.Tensor,
                    dones: torch.Tensor,
                    next_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算广义优势估计(GAE)
        """
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values
        
        return advantages, returns
    
    def update(self, memory: Dict[str, torch.Tensor], 
               batch_size: int = 64, 
               n_epochs: int = 10) -> Dict[str, float]:
        """
        更新策略和价值网络
        """
        # 准备数据
        states = memory['states']
        actions = memory['actions']
        old_log_probs = memory['log_probs']
        rewards = memory['rewards']
        dones = memory['dones']
        values = memory['values']
        next_states = memory['next_states']
        
        # 计算内在奖励
        with torch.no_grad():
            intrinsic_rewards = self.compute_intrinsic_reward(states, actions, next_states)
            total_rewards = rewards + self.intrinsic_coef * intrinsic_rewards
        
        # 计算GAE
        with torch.no_grad():
            _, _, next_values = self.actor_critic.get_action(next_states)
            next_values = next_values.squeeze()
            advantages, returns = self.compute_gae(total_rewards, values, dones, next_values)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 训练多个epoch
        n_samples = len(states)
        indices = np.arange(n_samples)
        
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        total_curiosity_loss = 0
        
        for epoch in range(n_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_next_states = next_states[batch_indices]
                
                # 评估当前策略
                log_probs, state_values, entropy = self.actor_critic.evaluate_actions(
                    batch_states, batch_actions
                )
                state_values = state_values.squeeze()
                
                # PPO损失
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = F.mse_loss(state_values, batch_returns)
                
                # KL散度损失（与对手策略）
                kl_div = self.compute_kl_divergence(batch_states)
                kl_loss = kl_div.mean()
                
                # 好奇心损失
                _, forward_loss, inverse_loss = self.curiosity(
                    batch_states, batch_actions, batch_next_states
                )
                curiosity_loss = forward_loss + inverse_loss
                
                # 总损失
                loss = (policy_loss + 
                       self.value_coef * value_loss - 
                       self.entropy_coef * entropy.mean() +
                       self.kl_coef * kl_loss +
                       curiosity_loss)
                
                # 更新网络
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor_critic.parameters()) + list(self.curiosity.parameters()),
                    max_norm=0.5
                )
                self.optimizer.step()
                
                # 记录统计
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_kl += kl_loss.item()
                total_curiosity_loss += curiosity_loss.item()
        
        n_updates = (n_samples // batch_size) * n_epochs
        
        self.update_count += 1
        
        # 返回训练统计
        stats = {
            'loss': total_loss / n_updates,
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'kl_divergence': total_kl / n_updates,
            'curiosity_loss': total_curiosity_loss / n_updates,
            'intrinsic_reward_mean': intrinsic_rewards.mean().item(),
            'intrinsic_reward_std': intrinsic_rewards.std().item(),
        }
        
        return stats
    
    def update_opponent_policy(self, current_performance: float):
        """
        更新对手策略（当主策略性能稳定提升时）
        """
        self.performance_history.append(current_performance)
        
        # 每隔一定间隔检查是否需要更新
        if self.update_count % self.update_opponent_interval == 0:
            if len(self.performance_history) >= self.update_opponent_interval:
                recent_perf = self.performance_history[-self.update_opponent_interval:]
                # 检查性能是否稳定提升（简单的增长趋势检查）
                if all(recent_perf[i] <= recent_perf[i+1] for i in range(len(recent_perf)-1)):
                    # 更新对手策略
                    self.opponent_actor_critic.load_state_dict(self.actor_critic.state_dict())
                    print(f"✓ 对手策略已更新 (更新次数: {self.update_count})")
                    return True
        
        return False
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'actor_critic': self.actor_critic.state_dict(),
            'opponent_actor_critic': self.opponent_actor_critic.state_dict(),
            'curiosity': self.curiosity.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'update_count': self.update_count,
            'performance_history': self.performance_history,
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        self.opponent_actor_critic.load_state_dict(checkpoint['opponent_actor_critic'])
        self.curiosity.load_state_dict(checkpoint['curiosity'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.update_count = checkpoint['update_count']
        self.performance_history = checkpoint['performance_history']

