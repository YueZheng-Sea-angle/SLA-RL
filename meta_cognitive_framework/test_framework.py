"""
元认知框架测试脚本
================
测试好奇心作为元优化器对不同基础算法的增强效果
"""

import torch
import numpy as np
import gym
import matplotlib.pyplot as plt
import os
from base_algorithms import DQN, SAC
from meta_wrapper import MetaCognitiveWrapper, SimpleMetaWrapper


def run_experiment(env_name, agent, agent_name, max_episodes=500, 
                   eval_frequency=50, device='cpu'):
    """
    运行实验
    
    Args:
        env_name: 环境名称
        agent: 智能体
        agent_name: 算法名称
        max_episodes: 最大回合数
        eval_frequency: 评估频率
        device: 设备
        
    Returns:
        dict: 训练历史
    """
    print(f"\n{'='*70}")
    print(f"测试: {agent_name}")
    print(f"环境: {env_name}")
    print(f"{'='*70}")
    
    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    
    episode_rewards = []
    eval_rewards = []
    losses = []
    meta_losses = []
    
    for episode in range(max_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        
        episode_reward = 0
        episode_loss = 0
        episode_meta_loss = 0
        steps = 0
        done = False
        
        while not done:
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
            
            # 更新智能体
            if hasattr(agent, 'replay_buffer') and len(agent.replay_buffer) > 64:
                update_result = agent.update(batch_size=64)
                if isinstance(update_result, dict):
                    episode_loss += update_result.get('base_loss', 0)
                    episode_meta_loss += update_result.get('meta_loss', 0)
                else:
                    episode_loss += update_result[0] if update_result[0] else 0
            
            episode_reward += reward
            state = next_state
            steps += 1
            
            if steps > 1000:  # 防止无限循环
                break
        
        episode_rewards.append(episode_reward)
        losses.append(episode_loss / max(steps, 1))
        meta_losses.append(episode_meta_loss / max(steps, 1))
        
        # 记录回合奖励（用于简化版包装器）
        if hasattr(agent, 'add_episode_reward'):
            agent.add_episode_reward(episode_reward)
        
        # 评估
        if (episode + 1) % eval_frequency == 0:
            eval_reward = evaluate(eval_env, agent, n_episodes=5)
            eval_rewards.append(eval_reward)
            
            print(f"Episode {episode+1}/{max_episodes} | "
                  f"Train Reward: {np.mean(episode_rewards[-10:]):.2f} | "
                  f"Eval Reward: {eval_reward:.2f} | "
                  f"Loss: {losses[-1]:.4f}")
    
    env.close()
    eval_env.close()
    
    return {
        'episode_rewards': episode_rewards,
        'eval_rewards': eval_rewards,
        'losses': losses,
        'meta_losses': meta_losses
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
    """绘制结果对比图"""
    os.makedirs(save_dir, exist_ok=True)
    
    n_envs = len(all_results)
    fig = plt.figure(figsize=(20, 5 * n_envs))
    
    colors = {
        'Base': 'gray',
        'Meta': 'red',
    }
    
    for env_idx, (env_name, env_results) in enumerate(all_results.items()):
        # 1. 训练奖励曲线
        ax1 = plt.subplot(n_envs, 4, env_idx * 4 + 1)
        for algo_name, history in env_results.items():
            rewards = history['episode_rewards']
            if len(rewards) > 0:
                window = min(20, len(rewards) // 10) if len(rewards) > 10 else 1
                if window > 1:
                    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    x = range(window-1, len(rewards))
                    color = colors.get('Meta' if 'Meta' in algo_name else 'Base', 'blue')
                    linestyle = '-' if 'Meta' in algo_name else '--'
                    ax1.plot(x, smoothed, label=algo_name, color=color, 
                            linestyle=linestyle, linewidth=2, alpha=0.8)
        
        ax1.set_xlabel('Episode', fontsize=11)
        ax1.set_ylabel('Reward (Smoothed)', fontsize=11)
        ax1.set_title(f'{env_name} - Training Progress', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. 评估奖励
        ax2 = plt.subplot(n_envs, 4, env_idx * 4 + 2)
        for algo_name, history in env_results.items():
            if len(history['eval_rewards']) > 0:
                color = colors.get('Meta' if 'Meta' in algo_name else 'Base', 'blue')
                linestyle = '-' if 'Meta' in algo_name else '--'
                ax2.plot(history['eval_rewards'], marker='o', label=algo_name,
                        color=color, linestyle=linestyle, linewidth=2, alpha=0.8)
        
        ax2.set_xlabel('Evaluation Steps', fontsize=11)
        ax2.set_ylabel('Avg Reward', fontsize=11)
        ax2.set_title(f'{env_name} - Evaluation Performance', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. 损失曲线
        ax3 = plt.subplot(n_envs, 4, env_idx * 4 + 3)
        for algo_name, history in env_results.items():
            losses = history['losses']
            if len(losses) > 0:
                # 平滑损失
                window = min(50, len(losses) // 10) if len(losses) > 10 else 1
                if window > 1:
                    smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
                    x = range(window-1, len(losses))
                    color = colors.get('Meta' if 'Meta' in algo_name else 'Base', 'blue')
                    linestyle = '-' if 'Meta' in algo_name else '--'
                    ax3.plot(x, smoothed, label=algo_name, color=color,
                            linestyle=linestyle, linewidth=2, alpha=0.8)
        
        ax3.set_xlabel('Episode', fontsize=11)
        ax3.set_ylabel('Loss (Smoothed)', fontsize=11)
        ax3.set_title(f'{env_name} - Training Loss', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # 4. 元损失曲线（仅Meta算法）
        ax4 = plt.subplot(n_envs, 4, env_idx * 4 + 4)
        for algo_name, history in env_results.items():
            if 'Meta' in algo_name and len(history['meta_losses']) > 0:
                meta_losses = history['meta_losses']
                window = min(50, len(meta_losses) // 10) if len(meta_losses) > 10 else 1
                if window > 1:
                    smoothed = np.convolve(meta_losses, np.ones(window)/window, mode='valid')
                    x = range(window-1, len(meta_losses))
                    ax4.plot(x, smoothed, label=algo_name, color='red',
                            linewidth=2, alpha=0.8)
        
        ax4.set_xlabel('Episode', fontsize=11)
        ax4.set_ylabel('Meta Loss', fontsize=11)
        ax4.set_title(f'{env_name} - Meta-Cognitive Loss', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f'{save_dir}/meta_cognitive_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 对比图已保存: {save_path}")
    plt.close()


def main():
    print("\n" + "="*80)
    print("元认知框架实验：好奇心作为多种算法的通用元优化器")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    # 测试环境（选择离散动作空间，适合DQN）
    test_envs = ['CartPole-v1', 'Acrobot-v1']
    
    all_results = {}
    
    for env_name in test_envs:
        print(f"\n\n{'#'*80}")
        print(f"测试环境: {env_name}")
        print(f"{'#'*80}")
        
        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        
        if isinstance(env.action_space, gym.spaces.Discrete):
            action_dim = env.action_space.n
            is_continuous = False
        else:
            action_dim = env.action_space.shape[0]
            is_continuous = True
        
        env.close()
        
        all_results[env_name] = {}
        
        # 测试1: 基础DQN
        if not is_continuous:
            print(f"\n{'='*70}")
            print("1. 基础 DQN（无元认知）")
            print(f"{'='*70}")
            
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
                env_name=env_name,
                agent=base_dqn,
                agent_name='Base DQN',
                max_episodes=300,
                eval_frequency=30,
                device=device
            )
            all_results[env_name]['Base DQN'] = history_base
            
            # 测试2: 元认知增强的DQN
            print(f"\n{'='*70}")
            print("2. 元认知 DQN（好奇心元优化）")
            print(f"{'='*70}")
            
            meta_dqn_base = DQN(
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
            
            meta_dqn = SimpleMetaWrapper(
                base_algorithm=meta_dqn_base,
                state_dim=state_dim,
                action_dim=action_dim,
                device=device
            )
            
            history_meta = run_experiment(
                env_name=env_name,
                agent=meta_dqn,
                agent_name='Meta-Cognitive DQN',
                max_episodes=300,
                eval_frequency=30,
                device=device
            )
            all_results[env_name]['Meta-Cognitive DQN'] = history_meta
    
    # 绘制结果
    if all_results:
        plot_results(all_results)
    
    print("\n" + "="*80)
    print("实验完成！结果保存在 ./meta_cognitive_framework/results/")
    print("="*80)


if __name__ == '__main__':
    main()

