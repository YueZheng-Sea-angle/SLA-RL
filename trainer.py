"""
训练器：处理环境交互和训练循环
"""

import torch
import numpy as np
import gym
from typing import Dict, List, Optional
from dual_policy_ppo import DualPolicyPPO
from collections import deque
import time


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.next_states = []
    
    def add(self, state, action, log_prob, reward, done, value, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.next_states.append(next_state)
    
    def get(self) -> Dict[str, torch.Tensor]:
        """返回tensor形式的数据"""
        return {
            'states': torch.FloatTensor(np.array(self.states)),
            'actions': torch.FloatTensor(np.array(self.actions)) if len(self.actions) > 0 and isinstance(self.actions[0], (list, np.ndarray)) else torch.LongTensor(np.array(self.actions)),
            'log_probs': torch.FloatTensor(np.array(self.log_probs)),
            'rewards': torch.FloatTensor(np.array(self.rewards)),
            'dones': torch.FloatTensor(np.array(self.dones)),
            'values': torch.FloatTensor(np.array(self.values)),
            'next_states': torch.FloatTensor(np.array(self.next_states)),
        }
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.next_states.clear()
    
    def __len__(self):
        return len(self.states)


class Trainer:
    """
    双策略PPO训练器
    """
    
    def __init__(self,
                 env_name: str,
                 agent: DualPolicyPPO,
                 max_episodes: int = 1000,
                 max_steps_per_episode: int = 1000,
                 update_frequency: int = 2048,
                 eval_frequency: int = 10,
                 eval_episodes: int = 5,
                 save_frequency: int = 50,
                 log_frequency: int = 1,
                 save_dir: str = './checkpoints'):
        
        self.env_name = env_name
        self.agent = agent
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.update_frequency = update_frequency
        self.eval_frequency = eval_frequency
        self.eval_episodes = eval_episodes
        self.save_frequency = save_frequency
        self.log_frequency = log_frequency
        self.save_dir = save_dir
        
        # 创建保存目录
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # 创建环境
        self.env = gym.make(env_name)
        self.eval_env = gym.make(env_name)
        
        # 缓冲区
        self.buffer = ReplayBuffer()
        
        # 训练统计
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        
        # 最佳性能跟踪
        self.best_eval_reward = -float('inf')
        
    def collect_experience(self, n_steps: int) -> Dict[str, float]:
        """
        收集经验
        返回: 统计信息字典
        """
        episode_reward = 0
        episode_length = 0
        episodes_finished = 0
        
        # 重置环境（兼容gym和gymnasium）
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            state, _ = reset_result
        else:
            state = reset_result
        
        for step in range(n_steps):
            # 选择动作
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                action, log_prob, value = self.agent.actor_critic.get_action(state_tensor)
                
                # 转换为numpy
                if self.agent.continuous:
                    action_np = action.cpu().numpy()[0]
                else:
                    action_np = action.cpu().item()
                
                log_prob_np = log_prob.cpu().item()
                value_np = value.cpu().item()
            
            # 执行动作（兼容gym和gymnasium）
            step_result = self.env.step(action_np)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:  # 旧版gym
                next_state, reward, done, info = step_result
            
            # 存储经验
            self.buffer.add(state, action_np, log_prob_np, reward, float(done), value_np, next_state)
            
            episode_reward += reward
            episode_length += 1
            
            # 更新状态
            state = next_state
            
            # 处理回合结束
            if done or episode_length >= self.max_steps_per_episode:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                episodes_finished += 1
                
                # 重置环境（兼容gym和gymnasium）
                reset_result = self.env.reset()
                if isinstance(reset_result, tuple):
                    state, _ = reset_result
                else:
                    state = reset_result
                episode_reward = 0
                episode_length = 0
        
        # 返回统计信息
        stats = {
            'steps_collected': n_steps,
            'episodes_finished': episodes_finished,
        }
        
        if episodes_finished > 0:
            recent_rewards = self.episode_rewards[-episodes_finished:]
            recent_lengths = self.episode_lengths[-episodes_finished:]
            stats.update({
                'mean_episode_reward': np.mean(recent_rewards),
                'mean_episode_length': np.mean(recent_lengths),
                'max_episode_reward': np.max(recent_rewards),
                'min_episode_reward': np.min(recent_rewards),
            })
        
        return stats
    
    def evaluate(self) -> float:
        """
        评估当前策略
        返回: 平均奖励
        """
        eval_rewards = []
        
        for _ in range(self.eval_episodes):
            # 重置环境（兼容gym和gymnasium）
            reset_result = self.eval_env.reset()
            if isinstance(reset_result, tuple):
                state, _ = reset_result
            else:
                state = reset_result
            episode_reward = 0
            done = False
            steps = 0
            
            while not done and steps < self.max_steps_per_episode:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                    action, _, _ = self.agent.actor_critic.get_action(state_tensor, deterministic=True)
                    
                    if self.agent.continuous:
                        action_np = action.cpu().numpy()[0]
                    else:
                        action_np = action.cpu().item()
                
                # 执行动作（兼容gym和gymnasium）
                step_result = self.eval_env.step(action_np)
                if len(step_result) == 5:
                    state, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:  # 旧版gym
                    state, reward, done, _ = step_result
                episode_reward += reward
                steps += 1
            
            eval_rewards.append(episode_reward)
        
        mean_reward = np.mean(eval_rewards)
        self.eval_rewards.append(mean_reward)
        
        return mean_reward
    
    def train(self) -> Dict[str, List]:
        """
        训练主循环
        返回: 训练历史
        """
        print("=" * 80)
        print("开始训练双策略PPO算法")
        print("=" * 80)
        print(f"环境: {self.env_name}")
        print(f"最大回合数: {self.max_episodes}")
        print(f"更新频率: {self.update_frequency} 步")
        print(f"设备: {self.agent.device}")
        print("=" * 80)
        
        total_steps = 0
        start_time = time.time()
        
        training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'eval_rewards': [],
            'training_stats': [],
        }
        
        episode = 0
        last_eval_episode = 0  # 上次评估时的回合数
        last_log_episode = 0   # 上次日志输出时的回合数
        
        while episode < self.max_episodes:
            # 收集经验
            collect_stats = self.collect_experience(self.update_frequency)
            total_steps += collect_stats['steps_collected']
            episode += collect_stats.get('episodes_finished', 0)
            
            # 更新策略
            if len(self.buffer) >= self.update_frequency:
                memory = self.buffer.get()
                # 转移到正确的设备
                memory = {k: v.to(self.agent.device) for k, v in memory.items()}
                
                update_stats = self.agent.update(memory)
                training_history['training_stats'].append(update_stats)
                
                # 清空缓冲区
                self.buffer.clear()
            
            # 评估 - 改进逻辑：检查是否已经达到下一个评估点
            if episode > 0 and episode - last_eval_episode >= self.eval_frequency:
                eval_reward = self.evaluate()
                training_history['eval_rewards'].append(eval_reward)
                last_eval_episode = episode  # 更新上次评估回合数
                
                # 更新对手策略
                self.agent.update_opponent_policy(eval_reward)
                
                # 保存最佳模型
                if eval_reward > self.best_eval_reward:
                    self.best_eval_reward = eval_reward
                    self.agent.save(f"{self.save_dir}/best_model.pth")
                    print(f"★ 新的最佳模型！评估奖励: {eval_reward:.2f}")
            
            # 日志 - 改进逻辑
            if episode > 0 and episode - last_log_episode >= self.log_frequency and len(self.episode_rewards) > 0:
                last_log_episode = episode  # 更新上次日志回合数
                elapsed_time = time.time() - start_time
                recent_rewards = self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
                
                print(f"\n回合 {episode}/{self.max_episodes} | 总步数: {total_steps}")
                print(f"  平均奖励 (最近10回合): {np.mean(recent_rewards):.2f} ± {np.std(recent_rewards):.2f}")
                print(f"  最大奖励: {np.max(recent_rewards):.2f}")
                
                if len(training_history['training_stats']) > 0:
                    latest_stats = training_history['training_stats'][-1]
                    print(f"  策略损失: {latest_stats['policy_loss']:.4f}")
                    print(f"  价值损失: {latest_stats['value_loss']:.4f}")
                    print(f"  KL散度: {latest_stats['kl_divergence']:.4f}")
                    print(f"  内在奖励: {latest_stats['intrinsic_reward_mean']:.4f} ± {latest_stats['intrinsic_reward_std']:.4f}")
                
                if len(self.eval_rewards) > 0:
                    print(f"  最近评估奖励: {self.eval_rewards[-1]:.2f}")
                
                print(f"  用时: {elapsed_time:.1f}秒")
            
            # 定期保存
            if episode % self.save_frequency == 0 and episode > 0:
                self.agent.save(f"{self.save_dir}/checkpoint_{episode}.pth")
                print(f"✓ 已保存检查点: checkpoint_{episode}.pth")
        
        # 训练结束
        print("\n" + "=" * 80)
        print("训练完成！")
        print(f"总训练时间: {(time.time() - start_time) / 60:.2f} 分钟")
        print(f"最佳评估奖励: {self.best_eval_reward:.2f}")
        print("=" * 80)
        
        # 保存最终模型
        self.agent.save(f"{self.save_dir}/final_model.pth")
        
        # 保存训练历史
        training_history['episode_rewards'] = self.episode_rewards
        training_history['episode_lengths'] = self.episode_lengths
        
        return training_history
    
    def close(self):
        """关闭环境"""
        self.env.close()
        self.eval_env.close()

