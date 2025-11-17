"""
最终基准测试：对比双策略PPO与标准PPO（修复所有问题）
"""

import torch
import numpy as np
import gym
from dual_policy_ppo import DualPolicyPPO
from trainer import Trainer
import matplotlib.pyplot as plt
import os
from datetime import datetime


class StandardPPO(DualPolicyPPO):
    """标准PPO（无好奇心，无双策略）"""
    
    def compute_intrinsic_reward(self, states, actions, next_states):
        return torch.zeros(len(states), device=self.device)
    
    def update(self, memory, batch_size=64, n_epochs=10):
        states = memory['states']
        actions = memory['actions']
        old_log_probs = memory['log_probs']
        rewards = memory['rewards']
        dones = memory['dones']
        values = memory['values']
        next_states = memory['next_states']
        
        with torch.no_grad():
            _, _, next_values = self.actor_critic.get_action(next_states)
            next_values = next_values.squeeze()
            advantages, returns = self.compute_gae(rewards, values, dones, next_values)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        n_samples = len(states)
        indices = np.arange(n_samples)
        
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        
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
                
                log_probs, state_values, entropy = self.actor_critic.evaluate_actions(
                    batch_states, batch_actions
                )
                state_values = state_values.squeeze()
                
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = torch.nn.functional.mse_loss(state_values, batch_returns)
                
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
        
        n_updates = (n_samples // batch_size) * n_epochs
        
        return {
            'loss': total_loss / n_updates,
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': 0,
            'kl_divergence': 0,
            'curiosity_loss': 0,
            'intrinsic_reward_mean': 0,
            'intrinsic_reward_std': 0,
        }


def run_test(env_name, agent_class, config_name, config, seed=42):
    """运行单个测试"""
    print(f"\n{'='*70}")
    print(f"测试: {config_name}")
    print(f"环境: {env_name}")
    print(f"配置: IC={config['intrinsic_coef']}, KL={config['kl_coef']}, Entropy={config['entropy_coef']}")
    print(f"{'='*70}")
    
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 获取环境信息
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
        continuous = False
    else:
        action_dim = env.action_space.shape[0]
        continuous = True
    
    env.close()
    
    # 创建智能体
    agent = agent_class(
        state_dim=state_dim,
        action_dim=action_dim,
        continuous=continuous,
        lr=config['lr'],
        gamma=config['gamma'],
        intrinsic_coef=config['intrinsic_coef'],
        kl_coef=config['kl_coef'],
        entropy_coef=config['entropy_coef'],
        hidden_dim=config['hidden_dim'],
    )
    
    # 创建保存目录（使用安全的路径名）
    safe_name = config_name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
    save_dir = f'./final_test/{env_name}_{safe_name}'
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建训练器
    trainer = Trainer(
        env_name=env_name,
        agent=agent,
        max_episodes=config['max_episodes'],
        update_frequency=config['update_frequency'],
        eval_frequency=config['eval_frequency'],
        save_frequency=1000,
        log_frequency=config['log_frequency'],
        save_dir=save_dir
    )
    
    # 训练
    history = trainer.train()
    trainer.close()
    
    return history


def main():
    print("\n" + "="*80)
    print("最终基准测试：双策略PPO vs 标准PPO")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 测试配置
    test_envs = {
        'CartPole-v1': {
            'max_episodes': 200,
            'update_frequency': 2048,
            'eval_frequency': 20,
            'log_frequency': 20,
        },
        'Acrobot-v1': {
            'max_episodes': 300,
            'update_frequency': 2048,
            'eval_frequency': 30,
            'log_frequency': 30,
        }
    }
    
    # 算法配置
    algorithms = {
        'Standard PPO': {
            'class': StandardPPO,
            'lr': 3e-4,
            'gamma': 0.99,
            'intrinsic_coef': 0.0,
            'kl_coef': 0.0,
            'entropy_coef': 0.01,
            'hidden_dim': 128,
        },
        'Dual PPO (IC=0.01)': {
            'class': DualPolicyPPO,
            'lr': 3e-4,
            'gamma': 0.99,
            'intrinsic_coef': 0.01,
            'kl_coef': 0.001,
            'entropy_coef': 0.02,
            'hidden_dim': 128,
        },
        'Dual PPO (IC=0.05)': {
            'class': DualPolicyPPO,
            'lr': 3e-4,
            'gamma': 0.99,
            'intrinsic_coef': 0.05,
            'kl_coef': 0.0,
            'entropy_coef': 0.02,
            'hidden_dim': 128,
        },
    }
    
    # 存储结果
    all_results = {}
    
    # 运行测试
    for env_name, env_config in test_envs.items():
        print(f"\n\n{'#'*80}")
        print(f"# 环境: {env_name}")
        print(f"{'#'*80}")
        
        all_results[env_name] = {}
        
        for algo_name, algo_config in algorithms.items():
            try:
                config = {**env_config, **algo_config}
                history = run_test(
                    env_name, 
                    algo_config['class'],
                    algo_name,
                    config,
                    seed=42
                )
                all_results[env_name][algo_name] = history
                
                # 打印简要统计
                rewards = history['episode_rewards']
                print(f"\n[OK] {algo_name} 完成")
                print(f"  回合数: {len(rewards)}")
                print(f"  平均奖励: {np.mean(rewards):.2f}")
                print(f"  最大奖励: {np.max(rewards):.2f}")
                if len(history['eval_rewards']) > 0:
                    print(f"  最佳评估: {max(history['eval_rewards']):.2f}")
                
            except Exception as e:
                print(f"\n[X] {algo_name} 失败: {e}")
                import traceback
                traceback.print_exc()
    
    # 绘制对比图
    print("\n" + "="*80)
    print("绘制对比图")
    print("="*80)
    
    for env_name, env_results in all_results.items():
        if not env_results:
            continue
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        # 训练奖励
        ax = axes[0]
        for idx, (algo_name, history) in enumerate(env_results.items()):
            rewards = history['episode_rewards']
            if len(rewards) > 0:
                window = min(20, len(rewards) // 5) if len(rewards) > 5 else 1
                if window > 1:
                    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    x = range(window-1, len(rewards))
                    ax.plot(x, smoothed, linewidth=2, label=algo_name, color=colors[idx % len(colors)])
                else:
                    ax.plot(rewards, linewidth=2, label=algo_name, color=colors[idx % len(colors)])
        
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Reward', fontsize=11)
        ax.set_title(f'{env_name} - Training Rewards', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 评估奖励
        ax = axes[1]
        for idx, (algo_name, history) in enumerate(env_results.items()):
            if len(history['eval_rewards']) > 0:
                ax.plot(history['eval_rewards'], marker='o', linewidth=2, 
                       markersize=4, label=algo_name, color=colors[idx % len(colors)])
        
        ax.set_xlabel('Evaluation', fontsize=11)
        ax.set_ylabel('Average Reward', fontsize=11)
        ax.set_title(f'{env_name} - Evaluation Rewards', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = f'./final_test/{env_name}_comparison.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"[OK] {env_name} 对比图已保存: {save_path}")
    
    # 性能总结
    print("\n" + "="*80)
    print("最终性能总结")
    print("="*80)
    
    for env_name, env_results in all_results.items():
        if not env_results:
            continue
        
        print(f"\n[{env_name}]")
        print("-" * 70)
        
        for algo_name, history in env_results.items():
            rewards = history['episode_rewards']
            if len(rewards) == 0:
                continue
            
            last_n = min(50, len(rewards) // 4)
            final_mean = np.mean(rewards[-last_n:])
            final_std = np.std(rewards[-last_n:])
            
            print(f"\n  {algo_name}:")
            print(f"    全程平均: {np.mean(rewards):.2f} +- {np.std(rewards):.2f}")
            print(f"    最大奖励: {np.max(rewards):.2f}")
            print(f"    最后{last_n}回合: {final_mean:.2f} +- {final_std:.2f}")
            
            if len(history['eval_rewards']) > 0:
                print(f"    最佳评估: {max(history['eval_rewards']):.2f}")
                print(f"    最终评估: {np.mean(history['eval_rewards'][-3:]):.2f}")
    
    print("\n" + "="*80)
    print("测试完成！")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print(f"\n结果保存在: ./final_test/")


if __name__ == '__main__':
    main()


