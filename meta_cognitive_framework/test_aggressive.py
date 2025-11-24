"""
激进版元认知测试
===============
验证Z-Score归一化和样本级权重的效果
"""

import torch
import numpy as np
import gym
import matplotlib.pyplot as plt
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))

from base_algorithms_fixed import StableDQN
from meta_wrapper_v4 import AggressiveMetaWrapper, UltraAggressiveMetaWrapper


def run_experiment(env_name, agent, agent_name, max_episodes=300, 
                   eval_frequency=30, device='cpu', verbose=True):
    """运行实验（增强版统计）"""
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
    weights_mean = []
    weights_std = []
    weights_min = []
    weights_max = []
    td_errors = []
    
    for episode in range(max_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        
        episode_reward = 0
        episode_loss = 0
        episode_meta_loss = 0
        episode_weight_stats = []
        episode_td = []
        steps = 0
        done = False
        
        while not done and steps < 1000:
            action = agent.select_action(state, eval_mode=False)
            
            step_result = env.step(action)
            if len(step_result) == 4:
                next_state, reward, done, info = step_result
            else:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            
            agent.store_transition(state, action, reward, next_state, done)
            
            if hasattr(agent, 'replay_buffer'):
                buffer = agent.replay_buffer
            else:
                buffer = agent.base_algorithm.replay_buffer
                
            if len(buffer) > 128:
                update_result = agent.update(batch_size=64)
                if isinstance(update_result, dict):
                    episode_loss += update_result.get('base_loss', 0)
                    episode_meta_loss += update_result.get('meta_loss', 0)
                    
                    # 记录权重统计
                    episode_weight_stats.append({
                        'mean': update_result.get('avg_weight', 1.0),
                        'std': update_result.get('weight_std', 0),
                        'min': update_result.get('weight_min', 1.0),
                        'max': update_result.get('weight_max', 1.0),
                    })
                    episode_td.append(update_result.get('td_error', 0))
                else:
                    episode_loss += update_result[0] if update_result[0] else 0
                    episode_td.append(update_result[1] if len(update_result) > 1 else 0)
            
            episode_reward += reward
            state = next_state
            steps += 1
        
        episode_rewards.append(episode_reward)
        losses.append(episode_loss / max(steps, 1))
        meta_losses.append(episode_meta_loss / max(steps, 1))
        
        # 统计权重
        if episode_weight_stats:
            weights_mean.append(np.mean([w['mean'] for w in episode_weight_stats]))
            weights_std.append(np.mean([w['std'] for w in episode_weight_stats]))
            weights_min.append(np.min([w['min'] for w in episode_weight_stats]))
            weights_max.append(np.max([w['max'] for w in episode_weight_stats]))
        else:
            weights_mean.append(1.0)
            weights_std.append(0.0)
            weights_min.append(1.0)
            weights_max.append(1.0)
        
        td_errors.append(np.mean(episode_td) if episode_td else 0)
        
        if hasattr(agent, 'add_episode_reward'):
            agent.add_episode_reward(episode_reward)
        
        # 评估
        if (episode + 1) % eval_frequency == 0:
            eval_reward = evaluate(eval_env, agent, n_episodes=5)
            eval_rewards.append(eval_reward)
            
            if verbose:
                recent_reward = np.mean(episode_rewards[-10:])
                recent_w_mean = np.mean(weights_mean[-10:])
                recent_w_std = np.mean(weights_std[-10:])
                recent_w_range = f"[{np.mean(weights_min[-10:]):.2f}, {np.mean(weights_max[-10:]):.2f}]"
                
                status = ""
                if hasattr(agent, 'get_stats'):
                    stats = agent.get_stats()
                    if stats.get('in_warmup', False):
                        status = " [WARMUP]"
                
                print(f"Ep {episode+1:3d} | "
                      f"R: {recent_reward:7.2f} | "
                      f"Eval: {eval_reward:7.2f} | "
                      f"W̄: {recent_w_mean:.2f}±{recent_w_std:.2f} | "
                      f"Range: {recent_w_range}{status}")
    
    env.close()
    eval_env.close()
    
    return {
        'episode_rewards': episode_rewards,
        'eval_rewards': eval_rewards,
        'losses': losses,
        'meta_losses': meta_losses,
        'weights_mean': weights_mean,
        'weights_std': weights_std,
        'weights_min': weights_min,
        'weights_max': weights_max,
        'td_errors': td_errors,
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


def plot_aggressive_results(all_results, save_dir='./meta_cognitive_framework/results'):
    """绘制激进版对比图（展示权重分布）"""
    os.makedirs(save_dir, exist_ok=True)
    
    n_envs = len(all_results)
    fig = plt.figure(figsize=(32, 6 * n_envs))
    
    colors = {
        'Stable Base': 'blue',
        'Aggressive': 'red',
        'Ultra Aggressive': 'purple',
    }
    
    for env_idx, (env_name, env_results) in enumerate(all_results.items()):
        # 1. 训练奖励
        ax1 = plt.subplot(n_envs, 7, env_idx * 7 + 1)
        for algo_name, history in env_results.items():
            rewards = history['episode_rewards']
            if len(rewards) > 0:
                window = min(20, len(rewards) // 10) if len(rewards) > 10 else 1
                if window > 1:
                    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    x = range(window-1, len(rewards))
                    
                    if 'Ultra' in algo_name:
                        color = colors['Ultra Aggressive']
                        linestyle = '-'
                    elif 'Aggressive' in algo_name:
                        color = colors['Aggressive']
                        linestyle = '-'
                    else:
                        color = colors['Stable Base']
                        linestyle = '--'
                    
                    ax1.plot(x, smoothed, label=algo_name, color=color, 
                            linestyle=linestyle, linewidth=2.5, alpha=0.9)
        
        ax1.set_xlabel('Episode', fontsize=10)
        ax1.set_ylabel('Reward', fontsize=10)
        ax1.set_title(f'{env_name} - Training', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 2. 评估性能
        ax2 = plt.subplot(n_envs, 7, env_idx * 7 + 2)
        for algo_name, history in env_results.items():
            if len(history['eval_rewards']) > 0:
                if 'Ultra' in algo_name:
                    color = colors['Ultra Aggressive']
                    linestyle = '-'
                elif 'Aggressive' in algo_name:
                    color = colors['Aggressive']
                    linestyle = '-'
                else:
                    color = colors['Stable Base']
                    linestyle = '--'
                
                ax2.plot(history['eval_rewards'], marker='o', label=algo_name,
                        color=color, linestyle=linestyle, linewidth=2.5, 
                        markersize=6, alpha=0.9)
        
        ax2.set_xlabel('Eval Steps', fontsize=10)
        ax2.set_ylabel('Avg Reward', fontsize=10)
        ax2.set_title(f'{env_name} - Evaluation', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # 3. 权重均值（关键图！）
        ax3 = plt.subplot(n_envs, 7, env_idx * 7 + 3)
        for algo_name, history in env_results.items():
            if 'weights_mean' in history and 'Aggressive' in algo_name:
                w_mean = history['weights_mean']
                if len(w_mean) > 0:
                    window = min(50, len(w_mean) // 10) if len(w_mean) > 10 else 1
                    if window > 1:
                        smoothed = np.convolve(w_mean, np.ones(window)/window, mode='valid')
                        x = range(window-1, len(w_mean))
                        
                        color = colors['Ultra Aggressive'] if 'Ultra' in algo_name else colors['Aggressive']
                        ax3.plot(x, smoothed, label=algo_name, color=color, 
                                linewidth=2.5, alpha=0.9)
        
        ax3.axhline(y=1.0, color='black', linestyle=':', alpha=0.5, label='Neutral')
        ax3.set_xlabel('Episode', fontsize=10)
        ax3.set_ylabel('Weight Mean', fontsize=10)
        ax3.set_title(f'{env_name} - Weight Distribution (Mean)', fontsize=11, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0.5, 1.5])
        
        # 4. 权重标准差（高方差是好的！）
        ax4 = plt.subplot(n_envs, 7, env_idx * 7 + 4)
        for algo_name, history in env_results.items():
            if 'weights_std' in history and 'Aggressive' in algo_name:
                w_std = history['weights_std']
                if len(w_std) > 0:
                    window = min(50, len(w_std) // 10) if len(w_std) > 10 else 1
                    if window > 1:
                        smoothed = np.convolve(w_std, np.ones(window)/window, mode='valid')
                        x = range(window-1, len(w_std))
                        
                        color = colors['Ultra Aggressive'] if 'Ultra' in algo_name else colors['Aggressive']
                        ax4.plot(x, smoothed, label=algo_name, color=color, 
                                linewidth=2.5, alpha=0.9)
        
        ax4.set_xlabel('Episode', fontsize=10)
        ax4.set_ylabel('Weight Std', fontsize=10)
        ax4.set_title(f'{env_name} - Weight Variance (Higher=Better!)', 
                     fontsize=11, fontweight='bold', color='darkgreen')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        # 5. 权重范围（Min-Max）
        ax5 = plt.subplot(n_envs, 7, env_idx * 7 + 5)
        for algo_name, history in env_results.items():
            if 'weights_min' in history and 'Aggressive' in algo_name:
                w_min = history['weights_min']
                w_max = history['weights_max']
                if len(w_min) > 0:
                    window = min(50, len(w_min) // 10) if len(w_min) > 10 else 1
                    if window > 1:
                        smoothed_min = np.convolve(w_min, np.ones(window)/window, mode='valid')
                        smoothed_max = np.convolve(w_max, np.ones(window)/window, mode='valid')
                        x = range(window-1, len(w_min))
                        
                        color = colors['Ultra Aggressive'] if 'Ultra' in algo_name else colors['Aggressive']
                        ax5.fill_between(x, smoothed_min, smoothed_max, 
                                        color=color, alpha=0.3)
                        ax5.plot(x, smoothed_max, color=color, linewidth=2, 
                                label=f'{algo_name} (Max)')
                        ax5.plot(x, smoothed_min, color=color, linewidth=2, 
                                linestyle='--', label=f'{algo_name} (Min)')
        
        ax5.axhline(y=1.0, color='black', linestyle=':', alpha=0.5)
        ax5.set_xlabel('Episode', fontsize=10)
        ax5.set_ylabel('Weight Range', fontsize=10)
        ax5.set_title(f'{env_name} - Weight Min/Max', fontsize=11, fontweight='bold')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim([0, 2.2])
        
        # 6. TD Error
        ax6 = plt.subplot(n_envs, 7, env_idx * 7 + 6)
        for algo_name, history in env_results.items():
            if 'td_errors' in history:
                td = [t for t in history['td_errors'] if t > 0]
                if len(td) > 0:
                    window = min(50, len(td) // 10) if len(td) > 10 else 1
                    if window > 1:
                        smoothed = np.convolve(td, np.ones(window)/window, mode='valid')
                        x = range(window-1, len(td))
                        
                        if 'Ultra' in algo_name:
                            color = colors['Ultra Aggressive']
                            linestyle = '-'
                        elif 'Aggressive' in algo_name:
                            color = colors['Aggressive']
                            linestyle = '-'
                        else:
                            color = colors['Stable Base']
                            linestyle = '--'
                        
                        ax6.plot(x, smoothed, label=algo_name, color=color,
                                linestyle=linestyle, linewidth=2.5, alpha=0.9)
        
        ax6.set_xlabel('Episode', fontsize=10)
        ax6.set_ylabel('TD Error', fontsize=10)
        ax6.set_title(f'{env_name} - TD Error', fontsize=11, fontweight='bold')
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3)
        
        # 7. 元损失
        ax7 = plt.subplot(n_envs, 7, env_idx * 7 + 7)
        for algo_name, history in env_results.items():
            if 'Aggressive' in algo_name and len(history['meta_losses']) > 0:
                meta_l = [l for l in history['meta_losses'] if l > 0]
                if meta_l:
                    window = min(50, len(meta_l) // 10) if len(meta_l) > 10 else 1
                    if window > 1:
                        smoothed = np.convolve(meta_l, np.ones(window)/window, mode='valid')
                        x = range(window-1, len(meta_l))
                        
                        color = colors['Ultra Aggressive'] if 'Ultra' in algo_name else colors['Aggressive']
                        ax7.plot(x, smoothed, label=algo_name, color=color, 
                                linewidth=2.5, alpha=0.9)
        
        ax7.set_xlabel('Episode', fontsize=10)
        ax7.set_ylabel('Meta Loss', fontsize=10)
        ax7.set_title(f'{env_name} - Meta Learning', fontsize=11, fontweight='bold')
        ax7.legend(fontsize=9)
        ax7.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f'{save_dir}/aggressive_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 激进版对比图已保存: {save_path}")
    plt.close()


def main():
    print("\n" + "="*80)
    print("🚀 元认知框架 - 激进版测试")
    print("核心：Z-Score归一化 → 高方差权重分布 → 真正的加速效果")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}\n")
    
    test_envs = ['CartPole-v1']
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
        
        # 1. 稳定版Base DQN（对照组）
        print(f"\n{'-'*70}")
        print("1. 稳定版 Base DQN")
        print(f"{'-'*70}")
        
        stable_base = StableDQN(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=128,
            lr=3e-4,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            tau=0.005,
            use_double_dqn=True,
            device=device
        )
        
        history_stable = run_experiment(
            env_name, stable_base, 'Stable Base DQN', 
            max_episodes=300, eval_frequency=30, device=device
        )
        all_results[env_name]['Stable Base DQN'] = history_stable
        
        # 2. 激进版元认知DQN（scale_factor=0.5）
        print(f"\n{'-'*70}")
        print("2. 激进版元认知 DQN (scale=0.5)")
        print(f"{'-'*70}")
        
        aggressive_base = StableDQN(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=128,
            lr=3e-4,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            tau=0.005,
            use_double_dqn=True,
            device=device
        )
        
        aggressive_meta = AggressiveMetaWrapper(
            base_algorithm=aggressive_base,
            state_dim=state_dim,
            action_dim=action_dim,
            meta_lr=1e-3,
            warmup_steps=2000,
            scale_factor=0.5,  # 中等激进
            use_exponential=False,
            device=device
        )
        
        history_aggressive = run_experiment(
            env_name, aggressive_meta, 'Aggressive Meta DQN (0.5)',
            max_episodes=300, eval_frequency=30, device=device
        )
        all_results[env_name]['Aggressive Meta DQN (0.5)'] = history_aggressive
        
        # 3. 超激进版元认知DQN（scale_factor=1.0）
        print(f"\n{'-'*70}")
        print("3. 超激进版元认知 DQN (scale=1.0)")
        print(f"{'-'*70}")
        
        ultra_base = StableDQN(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=128,
            lr=3e-4,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            tau=0.005,
            use_double_dqn=True,
            device=device
        )
        
        ultra_meta = UltraAggressiveMetaWrapper(
            base_algorithm=ultra_base,
            state_dim=state_dim,
            action_dim=action_dim,
            meta_lr=1e-3,
            warmup_steps=2000,
            device=device
        )
        
        history_ultra = run_experiment(
            env_name, ultra_meta, 'Ultra Aggressive Meta DQN (1.0)',
            max_episodes=300, eval_frequency=30, device=device
        )
        all_results[env_name]['Ultra Aggressive Meta DQN (1.0)'] = history_ultra
    
    # 绘制结果
    if all_results:
        plot_aggressive_results(all_results)
    
    print("\n" + "="*80)
    print("🎉 激进版测试完成！")
    print("="*80)
    
    # 打印最终统计
    for env_name, env_results in all_results.items():
        print(f"\n{env_name} 最终结果:")
        for algo_name, history in env_results.items():
            final_reward = np.mean(history['episode_rewards'][-20:])
            max_reward = np.max(history['episode_rewards'])
            
            if 'weights_std' in history:
                avg_w_std = np.mean(history['weights_std'][-50:])
                print(f"  {algo_name:35s}: 最后20ep={final_reward:6.2f}, "
                      f"最高={max_reward:6.2f}, 权重Std={avg_w_std:.3f}")
            else:
                print(f"  {algo_name:35s}: 最后20ep={final_reward:6.2f}, 最高={max_reward:6.2f}")


if __name__ == '__main__':
    main()

