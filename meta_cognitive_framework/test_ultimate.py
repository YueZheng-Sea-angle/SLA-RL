"""
终极测试脚本
============
使用修复后的StableDQN + 惊奇驱动的元认知包装器
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
from meta_wrapper_v3 import SurpriseDrivenMetaWrapper


def run_experiment(env_name, agent, agent_name, max_episodes=300, 
                   eval_frequency=30, device='cpu', verbose=True):
    """运行实验（增强版，记录更多信息）"""
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
    target_weights = []
    td_errors = []
    
    for episode in range(max_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        
        episode_reward = 0
        episode_loss = 0
        episode_meta_loss = 0
        episode_weights = []
        episode_target_weights = []
        episode_td_errors = []
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
                    episode_target_weights.append(update_result.get('target_weight', 1.0))
                    episode_td_errors.append(update_result.get('td_error', 0))
                else:
                    episode_loss += update_result[0] if update_result[0] else 0
                    episode_td_errors.append(update_result[1] if len(update_result) > 1 else 0)
            
            episode_reward += reward
            state = next_state
            steps += 1
        
        episode_rewards.append(episode_reward)
        losses.append(episode_loss / max(steps, 1))
        meta_losses.append(episode_meta_loss / max(steps, 1))
        weights.append(np.mean(episode_weights) if episode_weights else 1.0)
        target_weights.append(np.mean(episode_target_weights) if episode_target_weights else 1.0)
        td_errors.append(np.mean(episode_td_errors) if episode_td_errors else 0)
        
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
                recent_target_w = np.mean(target_weights[-10:])
                recent_td = np.mean(td_errors[-10:])
                
                status = ""
                if hasattr(agent, 'get_stats'):
                    stats = agent.get_stats()
                    if stats.get('in_warmup', False):
                        status = " [WARMUP]"
                
                print(f"Ep {episode+1:3d} | "
                      f"R: {recent_reward:7.2f} | "
                      f"Eval: {eval_reward:7.2f} | "
                      f"Loss: {losses[-1]:6.4f} | "
                      f"W: {recent_weight:.2f} | "
                      f"TW: {recent_target_w:.2f} | "
                      f"TD: {recent_td:.3f}{status}")
    
    env.close()
    eval_env.close()
    
    return {
        'episode_rewards': episode_rewards,
        'eval_rewards': eval_rewards,
        'losses': losses,
        'meta_losses': meta_losses,
        'weights': weights,
        'target_weights': target_weights,
        'td_errors': td_errors
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


def plot_ultimate_results(all_results, save_dir='./meta_cognitive_framework/results'):
    """绘制终极对比图"""
    os.makedirs(save_dir, exist_ok=True)
    
    n_envs = len(all_results)
    fig = plt.figure(figsize=(30, 6 * n_envs))
    
    colors = {
        'Stable Base': 'blue', 
        'Surprise Meta': 'red',
        'Target Weight': 'green'
    }
    
    for env_idx, (env_name, env_results) in enumerate(all_results.items()):
        # 1. 训练奖励
        ax1 = plt.subplot(n_envs, 6, env_idx * 6 + 1)
        for algo_name, history in env_results.items():
            rewards = history['episode_rewards']
            if len(rewards) > 0:
                window = min(20, len(rewards) // 10) if len(rewards) > 10 else 1
                if window > 1:
                    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    x = range(window-1, len(rewards))
                    color = colors.get('Surprise Meta' if 'Surprise' in algo_name else 'Stable Base', 'gray')
                    linestyle = '-' if 'Surprise' in algo_name else '--'
                    ax1.plot(x, smoothed, label=algo_name, color=color, 
                            linestyle=linestyle, linewidth=2.5, alpha=0.9)
        
        ax1.set_xlabel('Episode', fontsize=10)
        ax1.set_ylabel('Reward', fontsize=10)
        ax1.set_title(f'{env_name} - Training', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # 2. 评估性能
        ax2 = plt.subplot(n_envs, 6, env_idx * 6 + 2)
        for algo_name, history in env_results.items():
            if len(history['eval_rewards']) > 0:
                color = colors.get('Surprise Meta' if 'Surprise' in algo_name else 'Stable Base', 'gray')
                linestyle = '-' if 'Surprise' in algo_name else '--'
                ax2.plot(history['eval_rewards'], marker='o', label=algo_name,
                        color=color, linestyle=linestyle, linewidth=2.5, markersize=6, alpha=0.9)
        
        ax2.set_xlabel('Eval Steps', fontsize=10)
        ax2.set_ylabel('Avg Reward', fontsize=10)
        ax2.set_title(f'{env_name} - Evaluation', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # 3. 训练损失（对数尺度）
        ax3 = plt.subplot(n_envs, 6, env_idx * 6 + 3)
        for algo_name, history in env_results.items():
            losses = [l for l in history['losses'] if l > 0]
            if len(losses) > 0:
                window = min(50, len(losses) // 10) if len(losses) > 10 else 1
                if window > 1:
                    smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
                    x = range(window-1, len(losses))
                    color = colors.get('Surprise Meta' if 'Surprise' in algo_name else 'Stable Base', 'gray')
                    linestyle = '-' if 'Surprise' in algo_name else '--'
                    ax3.semilogy(x, smoothed, label=algo_name, color=color,
                                linestyle=linestyle, linewidth=2.5, alpha=0.9)
        
        ax3.set_xlabel('Episode', fontsize=10)
        ax3.set_ylabel('Loss (log scale)', fontsize=10)
        ax3.set_title(f'{env_name} - Training Loss', fontsize=11, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # 4. TD Error
        ax4 = plt.subplot(n_envs, 6, env_idx * 6 + 4)
        for algo_name, history in env_results.items():
            if 'td_errors' in history:
                td_errors = [td for td in history['td_errors'] if td > 0]
                if len(td_errors) > 0:
                    window = min(50, len(td_errors) // 10) if len(td_errors) > 10 else 1
                    if window > 1:
                        smoothed = np.convolve(td_errors, np.ones(window)/window, mode='valid')
                        x = range(window-1, len(td_errors))
                        color = colors.get('Surprise Meta' if 'Surprise' in algo_name else 'Stable Base', 'gray')
                        linestyle = '-' if 'Surprise' in algo_name else '--'
                        ax4.plot(x, smoothed, label=algo_name, color=color,
                                linestyle=linestyle, linewidth=2.5, alpha=0.9)
        
        ax4.set_xlabel('Episode', fontsize=10)
        ax4.set_ylabel('TD Error', fontsize=10)
        ax4.set_title(f'{env_name} - TD Error (Surprise)', fontsize=11, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        # 5. 权重对比（预测 vs 目标）
        ax5 = plt.subplot(n_envs, 6, env_idx * 6 + 5)
        for algo_name, history in env_results.items():
            if 'Surprise' in algo_name and 'weights' in history:
                # 预测权重
                weights = history['weights']
                if len(weights) > 0:
                    window = min(50, len(weights) // 10) if len(weights) > 10 else 1
                    if window > 1:
                        smoothed = np.convolve(weights, np.ones(window)/window, mode='valid')
                        x = range(window-1, len(weights))
                        ax5.plot(x, smoothed, label='Predicted W', color='red', linewidth=2.5, alpha=0.9)
                
                # 目标权重
                if 'target_weights' in history:
                    target_w = history['target_weights']
                    if len(target_w) > 0:
                        window = min(50, len(target_w) // 10) if len(target_w) > 10 else 1
                        if window > 1:
                            smoothed = np.convolve(target_w, np.ones(window)/window, mode='valid')
                            x = range(window-1, len(target_w))
                            ax5.plot(x, smoothed, label='Target W', color='green', 
                                    linewidth=2.5, linestyle='--', alpha=0.7)
        
        ax5.axhline(y=1.0, color='black', linestyle=':', alpha=0.5, label='Neutral')
        ax5.set_xlabel('Episode', fontsize=10)
        ax5.set_ylabel('Weight', fontsize=10)
        ax5.set_title(f'{env_name} - Meta Weights', fontsize=11, fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim([0.3, 2.2])
        
        # 6. 元损失
        ax6 = plt.subplot(n_envs, 6, env_idx * 6 + 6)
        for algo_name, history in env_results.items():
            if 'Surprise' in algo_name and len(history['meta_losses']) > 0:
                meta_losses = [l for l in history['meta_losses'] if l > 0]
                if meta_losses:
                    window = min(50, len(meta_losses) // 10) if len(meta_losses) > 10 else 1
                    if window > 1:
                        smoothed = np.convolve(meta_losses, np.ones(window)/window, mode='valid')
                        x = range(window-1, len(meta_losses))
                        ax6.plot(x, smoothed, label=algo_name, color='red', linewidth=2.5, alpha=0.9)
        
        ax6.set_xlabel('Episode', fontsize=10)
        ax6.set_ylabel('Meta Loss', fontsize=10)
        ax6.set_title(f'{env_name} - Meta Learning', fontsize=11, fontweight='bold')
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f'{save_dir}/ultimate_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 终极对比图已保存: {save_path}")
    plt.close()


def main():
    print("\n" + "="*80)
    print("🎯 元认知框架 - 终极测试")
    print("修复：晴天送伞雨天收伞 → 惊奇驱动学习")
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
        
        # 1. 稳定版Base DQN
        print(f"\n{'-'*70}")
        print("1. 稳定版 Base DQN（梯度裁剪 + Huber Loss + Double DQN）")
        print(f"{'-'*70}")
        
        stable_base = StableDQN(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=128,
            lr=3e-4,  # 降低学习率
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            tau=0.005,  # 软更新
            use_double_dqn=True,
            device=device
        )
        
        history_stable = run_experiment(
            env_name, stable_base, 'Stable Base DQN', 
            max_episodes=300, eval_frequency=30, device=device
        )
        all_results[env_name]['Stable Base DQN'] = history_stable
        
        # 2. 惊奇驱动元认知DQN
        print(f"\n{'-'*70}")
        print("2. 惊奇驱动元认知 DQN（TD Error → Weight）")
        print(f"{'-'*70}")
        
        surprise_base = StableDQN(
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
        
        surprise_meta = SurpriseDrivenMetaWrapper(
            base_algorithm=surprise_base,
            state_dim=state_dim,
            action_dim=action_dim,
            meta_lr=1e-3,
            warmup_steps=2000,  # 2000步预热
            device=device
        )
        
        history_surprise = run_experiment(
            env_name, surprise_meta, 'Surprise-Driven Meta DQN',
            max_episodes=300, eval_frequency=30, device=device
        )
        all_results[env_name]['Surprise-Driven Meta DQN'] = history_surprise
    
    # 绘制结果
    if all_results:
        plot_ultimate_results(all_results)
    
    print("\n" + "="*80)
    print("🎉 终极测试完成！")
    print("="*80)
    
    # 打印最终统计
    for env_name, env_results in all_results.items():
        print(f"\n{env_name} 最终结果:")
        for algo_name, history in env_results.items():
            final_reward = np.mean(history['episode_rewards'][-20:])
            max_reward = np.max(history['episode_rewards'])
            print(f"  {algo_name:30s}: 最后20ep平均={final_reward:7.2f}, 最高={max_reward:7.2f}")


if __name__ == '__main__':
    main()

