"""
改进版测试脚本
============
使用改进的元认知包装器
"""

import torch
import numpy as np
import gym
import matplotlib.pyplot as plt
import os
import sys

# 确保能导入模块
sys.path.insert(0, os.path.dirname(__file__))

from base_algorithms import DQN
from meta_wrapper_v2 import ImprovedMetaWrapper
import warnings
warnings.filterwarnings('ignore')


def run_experiment(env_name, agent, agent_name, max_episodes=300, 
                   eval_frequency=30, device='cpu', verbose=True):
    """运行实验"""
    if verbose:
        print(f"\n{'='*70}")
        print(f"算法: {agent_name}")
        print(f"环境: {env_name}")
        print(f"{'='*70}")
    
    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    
    episode_rewards = []
    eval_rewards = []
    losses = []
    meta_losses = []
    weights = []
    
    for episode in range(max_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        
        episode_reward = 0
        episode_loss = 0
        episode_meta_loss = 0
        episode_weights = []
        steps = 0
        done = False
        
        while not done and steps < 1000:
            # 选择动作
            action = agent.select_action(state, eval_mode=False)
            
            # 执行动作
            step_result = env.step(action)
            if len(step_result) == 4:
                next_state, reward, done, info = step_result
            else:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            
            # 存储转移
            agent.store_transition(state, action, reward, next_state, done)
            
            # 更新
            if hasattr(agent, 'replay_buffer'):
                buffer = agent.replay_buffer
            else:
                buffer = agent.base_algorithm.replay_buffer
                
            if len(buffer) > 128:
                update_result = agent.update(batch_size=64)
                if isinstance(update_result, dict):
                    episode_loss += update_result.get('base_loss', 0)
                    episode_meta_loss += update_result.get('meta_loss', 0)
                    episode_weights.append(update_result.get('weight', 1.0))
            
            episode_reward += reward
            state = next_state
            steps += 1
        
        episode_rewards.append(episode_reward)
        losses.append(episode_loss / max(steps, 1))
        meta_losses.append(episode_meta_loss / max(steps, 1))
        weights.append(np.mean(episode_weights) if episode_weights else 1.0)
        
        # 记录回合奖励
        if hasattr(agent, 'add_episode_reward'):
            agent.add_episode_reward(episode_reward)
        
        # 评估
        if (episode + 1) % eval_frequency == 0:
            eval_reward = evaluate(eval_env, agent, n_episodes=5)
            eval_rewards.append(eval_reward)
            
            if verbose:
                recent_reward = np.mean(episode_rewards[-10:])
                recent_weight = np.mean(weights[-10:])
                
                # 获取状态信息
                status = ""
                if hasattr(agent, 'get_stats'):
                    stats = agent.get_stats()
                    if stats.get('in_warmup', False):
                        status = " [WARMUP]"
                
                print(f"Ep {episode+1:3d} | "
                      f"Train: {recent_reward:7.2f} | "
                      f"Eval: {eval_reward:7.2f} | "
                      f"Loss: {losses[-1]:6.4f} | "
                      f"Weight: {recent_weight:.3f}{status}")
    
    env.close()
    eval_env.close()
    
    return {
        'episode_rewards': episode_rewards,
        'eval_rewards': eval_rewards,
        'losses': losses,
        'meta_losses': meta_losses,
        'weights': weights
    }


def evaluate(env, agent, n_episodes=5):
    """评估智能体"""
    total_reward = 0
    for _ in range(n_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        
        done = False
        steps = 0
        while not done and steps < 1000:
            action = agent.select_action(state, eval_mode=True)
            step_result = env.step(action)
            if len(step_result) == 4:
                state, reward, done, _ = step_result
            else:
                state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            total_reward += reward
            steps += 1
    
    return total_reward / n_episodes


def plot_results(all_results, save_dir='./meta_cognitive_framework/results'):
    """绘制改进版结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    n_envs = len(all_results)
    fig = plt.figure(figsize=(24, 5 * n_envs))
    
    colors = {'Base': 'gray', 'Meta': 'red', 'Improved Meta': 'green'}
    
    for env_idx, (env_name, env_results) in enumerate(all_results.items()):
        # 1. 训练奖励
        ax1 = plt.subplot(n_envs, 5, env_idx * 5 + 1)
        for algo_name, history in env_results.items():
            rewards = history['episode_rewards']
            if len(rewards) > 0:
                window = min(20, len(rewards) // 10) if len(rewards) > 10 else 1
                if window > 1:
                    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    x = range(window-1, len(rewards))
                    color = colors.get('Meta' if 'Meta' in algo_name else 'Base', 'blue')
                    if 'Improved' in algo_name:
                        color = colors['Improved Meta']
                    linestyle = '-' if 'Meta' in algo_name else '--'
                    ax1.plot(x, smoothed, label=algo_name, color=color, 
                            linestyle=linestyle, linewidth=2, alpha=0.8)
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward (Smoothed)')
        ax1.set_title(f'{env_name} - Training Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 评估性能
        ax2 = plt.subplot(n_envs, 5, env_idx * 5 + 2)
        for algo_name, history in env_results.items():
            if len(history['eval_rewards']) > 0:
                color = colors.get('Meta' if 'Meta' in algo_name else 'Base', 'blue')
                if 'Improved' in algo_name:
                    color = colors['Improved Meta']
                linestyle = '-' if 'Meta' in algo_name else '--'
                ax2.plot(history['eval_rewards'], marker='o', label=algo_name,
                        color=color, linestyle=linestyle, linewidth=2, alpha=0.8)
        
        ax2.set_xlabel('Evaluation Steps')
        ax2.set_ylabel('Avg Reward')
        ax2.set_title(f'{env_name} - Evaluation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 训练损失
        ax3 = plt.subplot(n_envs, 5, env_idx * 5 + 3)
        for algo_name, history in env_results.items():
            losses = history['losses']
            if len(losses) > 0 and max(losses) > 0:
                window = min(50, len(losses) // 10) if len(losses) > 10 else 1
                if window > 1:
                    smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
                    x = range(window-1, len(losses))
                    color = colors.get('Meta' if 'Meta' in algo_name else 'Base', 'blue')
                    if 'Improved' in algo_name:
                        color = colors['Improved Meta']
                    linestyle = '-' if 'Meta' in algo_name else '--'
                    ax3.plot(x, smoothed, label=algo_name, color=color,
                            linestyle=linestyle, linewidth=2, alpha=0.8)
        
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Loss (Smoothed)')
        ax3.set_title(f'{env_name} - Training Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 元损失
        ax4 = plt.subplot(n_envs, 5, env_idx * 5 + 4)
        for algo_name, history in env_results.items():
            if 'Meta' in algo_name and len(history['meta_losses']) > 0:
                meta_losses = [l for l in history['meta_losses'] if l > 0]
                if meta_losses:
                    window = min(50, len(meta_losses) // 10) if len(meta_losses) > 10 else 1
                    if window > 1:
                        smoothed = np.convolve(meta_losses, np.ones(window)/window, mode='valid')
                        x = range(window-1, len(meta_losses))
                        color = colors.get('Improved Meta' if 'Improved' in algo_name else 'Meta', 'red')
                        ax4.plot(x, smoothed, label=algo_name, color=color, linewidth=2, alpha=0.8)
        
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Meta Loss')
        ax4.set_title(f'{env_name} - Meta-Cognitive Loss')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 权重变化（新增）
        ax5 = plt.subplot(n_envs, 5, env_idx * 5 + 5)
        for algo_name, history in env_results.items():
            if 'Meta' in algo_name and 'weights' in history:
                weights = history['weights']
                if len(weights) > 0:
                    window = min(50, len(weights) // 10) if len(weights) > 10 else 1
                    if window > 1:
                        smoothed = np.convolve(weights, np.ones(window)/window, mode='valid')
                        x = range(window-1, len(weights))
                        color = colors.get('Improved Meta' if 'Improved' in algo_name else 'Meta', 'red')
                        ax5.plot(x, smoothed, label=algo_name, color=color, linewidth=2, alpha=0.8)
        
        ax5.axhline(y=1.0, color='black', linestyle=':', alpha=0.5, label='Neutral')
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Weight')
        ax5.set_title(f'{env_name} - Meta Weights')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f'{save_dir}/improved_meta_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 结果图已保存: {save_path}")
    plt.close()


def main():
    print("\n" + "="*80)
    print("元认知框架实验 - 改进版")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}\n")
    
    test_envs = ['CartPole-v1']  # 先测试一个环境
    all_results = {}
    
    for env_name in test_envs:
        print(f"\n{'#'*80}")
        print(f"环境: {env_name}")
        print(f"{'#'*80}")
        
        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else env.action_space.shape[0]
        env.close()
        
        all_results[env_name] = {}
        
        # 1. 基础DQN
        print(f"\n{'-'*70}")
        print("1. 基础 DQN")
        print(f"{'-'*70}")
        
        base_dqn = DQN(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=128,
            lr=1e-3,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            device=device
        )
        
        history_base = run_experiment(
            env_name, base_dqn, 'Base DQN', 
            max_episodes=300, eval_frequency=30, device=device
        )
        all_results[env_name]['Base DQN'] = history_base
        
        # 2. 改进版元认知DQN
        print(f"\n{'-'*70}")
        print("2. 改进版元认知 DQN")
        print(f"{'-'*70}")
        
        improved_base = DQN(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=128,
            lr=1e-3,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            device=device
        )
        
        improved_meta = ImprovedMetaWrapper(
            base_algorithm=improved_base,
            state_dim=state_dim,
            action_dim=action_dim,
            meta_lr=1e-3,
            warmup_steps=3000,  # 预热3000步
            device=device
        )
        
        history_improved = run_experiment(
            env_name, improved_meta, 'Improved Meta-Cognitive DQN',
            max_episodes=300, eval_frequency=30, device=device
        )
        all_results[env_name]['Improved Meta-Cognitive DQN'] = history_improved
    
    # 绘制结果
    if all_results:
        plot_results(all_results)
    
    print("\n" + "="*80)
    print("实验完成！")
    print("="*80)


if __name__ == '__main__':
    main()

