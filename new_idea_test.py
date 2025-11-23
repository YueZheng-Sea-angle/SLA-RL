"""
新点子测试：Dual PPO 改进版 vs Standard PPO vs Original Dual PPO
改进点：
1. Opponent 更新逻辑：改为软更新 (tau=0.01)，在每次 update 后执行。禁用原有的基于性能提升的硬更新。
2. KL 惩罚调整：给 (1-KL) 系数加底，公式为 0.1 + 0.9 * (1 - KL_penalty)，保证好奇心总有 10% 的基础权重。
"""

import torch
import numpy as np
import gym
import matplotlib.pyplot as plt
import os
from dual_policy_ppo import DualPolicyPPO
from trainer import Trainer

# 1. 定义 Standard PPO (复制自 dual_standard_benchmark.py)
class StandardPPO(DualPolicyPPO):
    """标准PPO（无好奇心，无双策略）"""
    
    def compute_intrinsic_reward(self, states, actions, next_states):
        return torch.zeros(len(states), device=self.device)
    
    def update(self, memory, batch_size=64, n_epochs=10):
        return super().update(memory, batch_size, n_epochs)

# 2. 定义改进版 Dual PPO
class ImprovedDualPPO(DualPolicyPPO):
    """
    改进版 Dual PPO
    1. 软更新 Opponent
    2. 调整 KL 惩罚系数
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = 0.01  # 软更新系数

    def compute_intrinsic_reward(self, states, actions, next_states):
        """
        计算改进的内在奖励
        r_intrinsic = Curiosity(s, a) * [0.1 + 0.9 * (1 - KL_penalty)]
        """
        # 好奇心奖励
        curiosity_reward, _, _ = self.curiosity(states, actions, next_states)
        
        # KL散度（归一化到[0, 1]）
        kl_div = self.compute_kl_divergence(states)
        kl_penalty = torch.clamp(kl_div / self.kl_threshold, 0, 1)
        
        # 修改点：给系数加底
        # 原始系数: 1 - kl_penalty
        # 新系数: 0.1 + 0.9 * (1 - kl_penalty)
        # 当 kl_penalty=0 时，系数=1.0
        # 当 kl_penalty=1 时，系数=0.1
        coefficient = 0.1 + 0.9 * (1 - kl_penalty)
        
        intrinsic_reward = curiosity_reward * coefficient
        
        return intrinsic_reward.detach()

    def update(self, memory, batch_size=64, n_epochs=10):
        # 执行父类更新 (梯度下降)
        stats = super().update(memory, batch_size, n_epochs)
        
        # 修改点：软更新 Opponent
        # 在每次策略更新后，让 Opponent 缓慢跟随 Actor
        for param, target_param in zip(self.actor_critic.parameters(), self.opponent_actor_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return stats

    def update_opponent_policy(self, current_performance):
        # 禁用父类的硬更新逻辑
        # 只记录性能历史
        self.performance_history.append(current_performance)
        return False

# 3. 运行实验函数
def run_experiment(env_name, agent_class, config_name, config, seed=42):
    print(f"\n{'='*70}")
    print(f"测试: {config_name}")
    print(f"环境: {env_name}")
    print(f"{'='*70}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
        continuous = False
    else:
        action_dim = env.action_space.shape[0]
        continuous = True
    
    env.close()
    
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
    
    # 这是一个专门的测试，保存到 new_idea_test 目录
    safe_name = config_name.replace(' ', '_').replace('(', '').replace(')', '')
    save_dir = f'./new_idea_test/{env_name}_{safe_name}'
    os.makedirs(save_dir, exist_ok=True)
    
    trainer = Trainer(
        env_name=env_name,
        agent=agent,
        max_episodes=config['max_episodes'],
        update_frequency=config['update_frequency'],
        eval_frequency=config['eval_frequency'],
        save_frequency=10000, # 不怎么保存中间结果
        log_frequency=config['log_frequency'],
        save_dir=save_dir
    )
    
    history = trainer.train()
    trainer.close()
    
    return history

# 4. 绘图函数
def plot_comparison(results, save_dir='./new_idea_test'):
    n_envs = len(results)
    fig = plt.figure(figsize=(18, 6 * n_envs))
    
    colors = {
        'Standard PPO': 'gray', 
        'Original Dual PPO': 'blue',
        'Improved Dual PPO (Soft Update)': 'red'
    }
    
    linestyles = {
        'Standard PPO': '--', 
        'Original Dual PPO': '-',
        'Improved Dual PPO (Soft Update)': '-'
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
        
        # 3. 稳定性 (100回合均值)
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
    save_path = f'{save_dir}/new_idea_comparison.png'
    plt.savefig(save_path, dpi=300)
    print(f"\n[OK] 对比图已保存: {save_path}")

def main():
    print("\n" + "="*80)
    print("新点子测试: Soft Update + KL Base")
    print("="*80)
    
    test_envs = {
        'MountainCar-v0': {
            'max_episodes': 600, 
            'update_frequency': 2048,
            'eval_frequency': 50,
            'log_frequency': 50,
        },
    }
    
    algorithms = {
        'Standard PPO': {
            'class': StandardPPO,
            'lr': 3e-4, 'gamma': 0.99, 'hidden_dim': 256,
            'intrinsic_coef': 0.0, 'kl_coef': 0.0, 'entropy_coef': 0.01
        },
        'Original Dual PPO': {
            'class': DualPolicyPPO,
            'lr': 3e-4, 'gamma': 0.99, 'hidden_dim': 256,
            'intrinsic_coef': 0.02, 'kl_coef': 0.001, 'entropy_coef': 0.01
        },
        'Improved Dual PPO (Soft Update)': {
            'class': ImprovedDualPPO,
            'lr': 3e-4, 'gamma': 0.99, 'hidden_dim': 256,
            'intrinsic_coef': 0.02, 'kl_coef': 0.001, 'entropy_coef': 0.01
        }
    }
    
    os.makedirs('./new_idea_test', exist_ok=True)
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
        
    print("\n测试结束。结果在 ./new_idea_test/")

if __name__ == '__main__':
    main()

