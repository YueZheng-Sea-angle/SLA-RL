"""
Dual vs Standard PPO 基准测试：双方法对比
对比算法：
1. Standard PPO (基准)
2. Dual PPO (V1: 奖励塑形法)
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
        # 调用父类 update，但由于 intrinsic_reward 已经是0，
        # 且我们不想计算好奇心loss和KL loss，所以最好重写 update 
        # 或者简单地把相关系数设为0
        
        # 为了简单和完全一致性，我们直接使用父类 update，
        #但在初始化时将 intrinsic_coef 和 kl_coef 设为 0
        return super().update(memory, batch_size, n_epochs)

def run_experiment(env_name, agent_class, config_name, config, seed=42):
    """运行单个实验"""
    print(f"\n{'='*70}")
    print(f"测试: {config_name}")
    print(f"环境: {env_name}")
    print(f"配置: IC={config.get('intrinsic_coef', 0)}, KL={config.get('kl_coef', 0)}")
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
    # 注意：不同类可能接受不同参数，需要过滤
    agent_params = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'continuous': continuous,
        'lr': config['lr'],
        'gamma': config['gamma'],
        'intrinsic_coef': config.get('intrinsic_coef', 0.0),
        'kl_coef': config.get('kl_coef', 0.0),
        'entropy_coef': config['entropy_coef'],
        'hidden_dim': config['hidden_dim'],
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    agent = agent_class(**agent_params)
    
    # 创建保存目录
    safe_name = config_name.replace(' ', '_')
    save_dir = f'./dual_standard_test/{env_name}_{safe_name}'
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建训练器
    trainer = Trainer(
        env_name=env_name,
        agent=agent,
        max_episodes=config['max_episodes'],
        update_frequency=config['update_frequency'],
        eval_frequency=config['eval_frequency'],
        save_frequency=10000,
        log_frequency=config['log_frequency'],
        save_dir=save_dir
    )
    
    # 训练
    history = trainer.train()
    trainer.close()
    
    return history

def plot_comparison(results, save_dir='./dual_standard_test'):
    """绘制对比图"""
    n_envs = len(results)
    fig = plt.figure(figsize=(18, 6 * n_envs))
    
    colors = {
        'Standard PPO': 'gray', 
        'Dual PPO (Reward Shaping)': 'blue'
    }
    
    linestyles = {
        'Standard PPO': '--', 
        'Dual PPO (Reward Shaping)': '-'
    }
    
    for env_idx, (env_name, env_results) in enumerate(results.items()):
        if not env_results:
            continue
        
        # 1. 训练奖励 (平滑)
        ax1 = plt.subplot(n_envs, 3, env_idx * 3 + 1)
        for algo_name, history in env_results.items():
            rewards = history['episode_rewards']
            if len(rewards) > 0:
                window = min(50, len(rewards) // 10) if len(rewards) > 10 else 1
                if window > 1:
                    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    x = range(window-1, len(rewards))
                    ax1.plot(x, smoothed, linewidth=2.0, label=algo_name, 
                            color=colors.get(algo_name, 'black'), 
                            linestyle=linestyles.get(algo_name, '-'), alpha=0.8)
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward (Smoothed)')
        ax1.set_title(f'{env_name} - Training Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 评估奖励
        ax2 = plt.subplot(n_envs, 3, env_idx * 3 + 2)
        for algo_name, history in env_results.items():
            if len(history['eval_rewards']) > 0:
                ax2.plot(history['eval_rewards'], marker='o', linewidth=2.0, 
                        label=algo_name,
                        color=colors.get(algo_name, 'black'), 
                        linestyle=linestyles.get(algo_name, '-'), alpha=0.8)
        
        ax2.set_xlabel('Evaluation Steps')
        ax2.set_ylabel('Avg Reward')
        ax2.set_title(f'{env_name} - Evaluation Performance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 成功率/稳定性 (最后100回合均值变化)
        ax3 = plt.subplot(n_envs, 3, env_idx * 3 + 3)
        for algo_name, history in env_results.items():
            rewards = history['episode_rewards']
            if len(rewards) >= 100:
                rolling_mean = []
                for i in range(len(rewards) - 99):
                    rolling_mean.append(np.mean(rewards[i:i+100]))
                ax3.plot(range(99, len(rewards)), rolling_mean, 
                        linewidth=2.0, label=algo_name,
                        color=colors.get(algo_name, 'black'), 
                        linestyle=linestyles.get(algo_name, '-'), alpha=0.8)
        
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('100-Ep Moving Avg')
        ax3.set_title(f'{env_name} - Long-term Stability')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f'{save_dir}/dual_standard_comparison.png'
    plt.savefig(save_path, dpi=300)
    print(f"\n[OK] 对比图已保存: {save_path}")

def main():
    print("\n" + "="*80)
    print("Dual PPO vs Standard PPO 对比实验")
    print("="*80)
    
    # 测试环境
    # 建议先只跑 MountainCar-v0 验证概念，因为它最快且稀疏
    test_envs = {
        'MountainCar-v0': {
            'max_episodes': 600,      # 足够看收敛
            'update_frequency': 2048,
            'eval_frequency': 50,
            'log_frequency': 50,
        },
    }
    
    # 算法配置（只保留 Standard PPO 和 Dual PPO）
    algorithms = {
        'Standard PPO': {
            'class': StandardPPO,
            'lr': 3e-4, 'gamma': 0.99, 'hidden_dim': 256,
            'intrinsic_coef': 0.0, 'kl_coef': 0.0, 'entropy_coef': 0.01
        },
        'Dual PPO (Reward Shaping)': {
            'class': DualPolicyPPO,  # V1
            'lr': 3e-4, 'gamma': 0.99, 'hidden_dim': 256,
            'intrinsic_coef': 0.02,  # 好奇心奖励系数
            'kl_coef': 0.001,        # 固定KL惩罚
            'entropy_coef': 0.01
        }
    }
    
    os.makedirs('./dual_standard_test', exist_ok=True)
    all_results = {}
    
    for env_name, env_config in test_envs.items():
        all_results[env_name] = {}
        for algo_name, algo_config in algorithms.items():
            try:
                config = {**env_config, **algo_config}
                history = run_experiment(
                    env_name,
                    algo_config['class'],
                    algo_name,
                    config,
                    seed=42
                )
                all_results[env_name][algo_name] = history
                print(f"[OK] {algo_name} 完成")
            except Exception as e:
                print(f"[ERROR] {algo_name} 失败: {e}")
                import traceback
                traceback.print_exc()
    
    if any(all_results.values()):
        plot_comparison(all_results)
        
    print("\n测试结束。结果在 ./dual_standard_test/")

if __name__ == '__main__':
    main()

