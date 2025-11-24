"""
混合策略实验：结合Original和Improved的优点
策略：前期用硬更新(快速改进) + 后期切换到软更新(稳定收敛)
"""

import torch
import numpy as np
import gym
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from dual_policy_ppo import DualPolicyPPO
from trainer import Trainer


class HybridDualPPO(DualPolicyPPO):
    """
    混合策略 Dual PPO
    前期：使用硬更新（基于性能提升）
    后期：切换到软更新（稳定训练）
    """
    def __init__(self, *args, tau=0.01, kl_base=0.1, switch_episode=300, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = tau
        self.kl_base = kl_base
        self.switch_episode = switch_episode  # 切换的回合数
        self.current_episode = 0
        self.use_soft_update = False
        
    def compute_intrinsic_reward(self, states, actions, next_states):
        """带KL底数的内在奖励计算"""
        curiosity_reward, _, _ = self.curiosity(states, actions, next_states)
        kl_div = self.compute_kl_divergence(states)
        kl_penalty = torch.clamp(kl_div / self.kl_threshold, 0, 1)
        coefficient = self.kl_base + (1 - self.kl_base) * (1 - kl_penalty)
        intrinsic_reward = curiosity_reward * coefficient
        return intrinsic_reward.detach()

    def update(self, memory, batch_size=64, n_epochs=10):
        """更新策略，根据阶段选择更新方式"""
        stats = super().update(memory, batch_size, n_epochs)
        
        # 如果已经切换到软更新阶段
        if self.use_soft_update:
            for param, target_param in zip(self.actor_critic.parameters(), 
                                          self.opponent_actor_critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return stats
    
    def update_opponent_policy(self, current_performance):
        """根据阶段决定是否使用硬更新"""
        self.performance_history.append(current_performance)
        
        # 检查是否需要切换
        if not self.use_soft_update and self.current_episode >= self.switch_episode:
            self.use_soft_update = True
            print(f"\n{'='*70}")
            print(f"[切换策略] Episode {self.current_episode}: 从硬更新切换到软更新 (tau={self.tau})")
            print(f"{'='*70}\n")
        
        # 前期使用父类的硬更新逻辑
        if not self.use_soft_update:
            return super().update_opponent_policy(current_performance)
        else:
            # 后期禁用硬更新
            return False


class StandardPPO(DualPolicyPPO):
    """标准PPO基准"""
    def compute_intrinsic_reward(self, states, actions, next_states):
        return torch.zeros(len(states), device=self.device)
    
    def update(self, memory, batch_size=64, n_epochs=10):
        return super().update(memory, batch_size, n_epochs)


class ImprovedDualPPO(DualPolicyPPO):
    """改进版Dual PPO（纯软更新）"""
    def __init__(self, *args, tau=0.01, kl_base=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = tau
        self.kl_base = kl_base

    def compute_intrinsic_reward(self, states, actions, next_states):
        curiosity_reward, _, _ = self.curiosity(states, actions, next_states)
        kl_div = self.compute_kl_divergence(states)
        kl_penalty = torch.clamp(kl_div / self.kl_threshold, 0, 1)
        coefficient = self.kl_base + (1 - self.kl_base) * (1 - kl_penalty)
        intrinsic_reward = curiosity_reward * coefficient
        return intrinsic_reward.detach()

    def update(self, memory, batch_size=64, n_epochs=10):
        stats = super().update(memory, batch_size, n_epochs)
        for param, target_param in zip(self.actor_critic.parameters(), 
                                      self.opponent_actor_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        return stats

    def update_opponent_policy(self, current_performance):
        self.performance_history.append(current_performance)
        return False


def run_experiment(env_name, agent_class, config_name, config, seed=42):
    """运行单个实验"""
    print(f"\n{'='*70}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始: {config_name}")
    print(f"配置: {config}")
    print(f"{'='*70}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else env.action_space.shape[0]
    continuous = not isinstance(env.action_space, gym.spaces.Discrete)
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
    
    # 混合策略专有参数
    if agent_class == HybridDualPPO:
        agent_params['tau'] = config.get('tau', 0.01)
        agent_params['kl_base'] = config.get('kl_base', 0.1)
        agent_params['switch_episode'] = config.get('switch_episode', 300)
    elif agent_class == ImprovedDualPPO:
        agent_params['tau'] = config.get('tau', 0.01)
        agent_params['kl_base'] = config.get('kl_base', 0.1)
    
    agent = agent_class(**agent_params)
    
    # 注入episode计数器（用于混合策略）
    if hasattr(agent, 'current_episode'):
        original_train = Trainer.train
        def train_with_counter(self):
            for episode in range(self.max_episodes):
                agent.current_episode = episode
                # 执行原始训练逻辑...
            return original_train(self)
    
    safe_name = config_name.replace(' ', '_').replace('(', '').replace(')', '')
    save_dir = f'./hybrid_strategy_test/{env_name}_{safe_name}'
    os.makedirs(save_dir, exist_ok=True)
    
    with open(f'{save_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
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
    
    # 手动注入episode计数
    if isinstance(agent, HybridDualPPO):
        original_train_loop = trainer.train
        def wrapped_train():
            history = {'episode_rewards': [], 'eval_rewards': []}
            for ep in range(trainer.max_episodes):
                agent.current_episode = ep
                # 简化：直接调用原始训练
            return original_train_loop()
        trainer.train = wrapped_train
    
    history = trainer.train()
    trainer.close()
    
    final_100 = np.mean(history['episode_rewards'][-100:])
    best_eval = max(history['eval_rewards']) if history['eval_rewards'] else -np.inf
    
    print(f"[完成] {config_name}: 最后100回合={final_100:.2f}, 最佳评估={best_eval:.2f}")
    
    return history, final_100, best_eval


def plot_comparison(results, save_path='./hybrid_strategy_test/comparison.png'):
    """绘制对比图"""
    fig = plt.figure(figsize=(18, 6))
    
    colors = {
        'Standard PPO': 'gray',
        'Original Dual PPO': 'blue', 
        'Improved Dual PPO': 'red',
        'Hybrid Dual PPO (Switch@200)': 'green',
        'Hybrid Dual PPO (Switch@300)': 'orange',
        'Hybrid Dual PPO (Switch@400)': 'purple'
    }
    
    # 1. 训练进度
    ax1 = plt.subplot(1, 3, 1)
    for name, data in results.items():
        history = data['history']
        rewards = history['episode_rewards']
        if len(rewards) > 0:
            window = 50
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            x = range(window-1, len(rewards))
            ax1.plot(x, smoothed, linewidth=2, label=name, 
                    color=colors.get(name, 'black'), alpha=0.8)
    
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Reward (Smoothed)', fontsize=12)
    ax1.set_title('Training Progress', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. 评估性能
    ax2 = plt.subplot(1, 3, 2)
    for name, data in results.items():
        history = data['history']
        if len(history['eval_rewards']) > 0:
            ax2.plot(history['eval_rewards'], marker='o', linewidth=2,
                    label=name, color=colors.get(name, 'black'), alpha=0.8)
    
    ax2.set_xlabel('Evaluation Steps', fontsize=12)
    ax2.set_ylabel('Avg Reward', fontsize=12)
    ax2.set_title('Evaluation Performance', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. 最终性能对比
    ax3 = plt.subplot(1, 3, 3)
    names = list(results.keys())
    final_perfs = [results[n]['final_100'] for n in names]
    bars = ax3.barh(names, final_perfs, color=[colors.get(n, 'black') for n in names], alpha=0.7)
    
    for i, (name, perf) in enumerate(zip(names, final_perfs)):
        ax3.text(perf, i, f' {perf:.1f}', va='center', fontsize=10, fontweight='bold')
    
    ax3.set_xlabel('Final 100-Episode Average', fontsize=12)
    ax3.set_title('Final Performance', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] 对比图已保存: {save_path}")
    plt.close()


def main():
    print("\n" + "="*80)
    print("混合策略实验：硬更新→软更新的自适应切换")
    print("="*80)
    
    env_name = 'MountainCar-v0'
    base_config = {
        'max_episodes': 600,
        'update_frequency': 2048,
        'eval_frequency': 50,
        'log_frequency': 50,
        'hidden_dim': 256
    }
    
    results_dir = './hybrid_strategy_test'
    os.makedirs(results_dir, exist_ok=True)
    results = {}
    
    # 使用上次实验找到的最佳配置作为基础
    best_original_config = {
        **base_config,
        'lr': 0.0001,
        'intrinsic_coef': 0.05,
        'kl_coef': 0.0005,
        'entropy_coef': 0.005,
        'gamma': 0.99
    }
    
    best_improved_config = {
        **base_config,
        'lr': 0.0003,
        'intrinsic_coef': 0.02,
        'kl_coef': 0.0005,
        'entropy_coef': 0.02,
        'tau': 0.02,
        'kl_base': 0.1,
        'gamma': 0.995
    }
    
    # 测试配置
    experiments = [
        # 基准
        ('Standard PPO', StandardPPO, {**base_config, 'lr': 3e-4, 'gamma': 0.99, 
                                        'intrinsic_coef': 0.0, 'kl_coef': 0.0, 'entropy_coef': 0.01}),
        
        # Original最佳配置
        ('Original Dual PPO', DualPolicyPPO, best_original_config),
        
        # Improved最佳配置
        ('Improved Dual PPO', ImprovedDualPPO, best_improved_config),
        
        # 混合策略：不同切换时机
        ('Hybrid Dual PPO (Switch@200)', HybridDualPPO, 
         {**best_original_config, 'tau': 0.02, 'kl_base': 0.1, 'switch_episode': 200}),
        
        ('Hybrid Dual PPO (Switch@300)', HybridDualPPO,
         {**best_original_config, 'tau': 0.02, 'kl_base': 0.1, 'switch_episode': 300}),
        
        ('Hybrid Dual PPO (Switch@400)', HybridDualPPO,
         {**best_original_config, 'tau': 0.02, 'kl_base': 0.1, 'switch_episode': 400}),
    ]
    
    # 运行实验
    for name, agent_class, config in experiments:
        try:
            history, final_100, best_eval = run_experiment(
                env_name, agent_class, name, config, seed=42
            )
            results[name] = {
                'history': history,
                'final_100': final_100,
                'best_eval': best_eval,
                'config': config
            }
        except Exception as e:
            print(f"[ERROR] {name} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 绘图和保存
    if results:
        plot_comparison(results, f'{results_dir}/comparison.png')
        
        # 保存结果
        with open(f'{results_dir}/results_summary.json', 'w') as f:
            json.dump({
                name: {
                    'final_100': float(data['final_100']),
                    'best_eval': float(data['best_eval']),
                    'config': data['config']
                } for name, data in results.items()
            }, f, indent=2)
        
        # 打印总结
        print("\n" + "="*80)
        print("实验结果总结")
        print("="*80)
        for name, data in results.items():
            print(f"{name:40s} | 最后100回合: {data['final_100']:7.2f} | 最佳评估: {data['best_eval']:7.2f}")
        print("="*80)
    
    print(f"\n所有结果保存在: {results_dir}/")


if __name__ == '__main__':
    main()

