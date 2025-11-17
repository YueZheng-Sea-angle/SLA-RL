"""
稀疏奖励环境基准测试 V2：改进稳定性
关键改进：
1. 降低内在奖励系数衰减
2. 更严格的KL散度约束
3. 禁用对手策略自动更新（容易导致不稳定）
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


class StableDualPPO(DualPolicyPPO):
    """稳定版双策略PPO：加入自适应机制"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_intrinsic_coef = self.intrinsic_coef
        self.episode_count = 0
        
    def update(self, memory, batch_size=64, n_epochs=10):
        """带内在奖励衰减的更新"""
        self.episode_count += 1
        
        # 内在奖励系数线性衰减（从初始值衰减到0）
        decay_episodes = 600  # 在600个回合内逐渐衰减
        decay_factor = max(0, 1 - self.episode_count / decay_episodes)
        self.intrinsic_coef = self.initial_intrinsic_coef * decay_factor
        
        # 调用父类更新
        stats = super().update(memory, batch_size, n_epochs)
        stats['intrinsic_coef'] = self.intrinsic_coef
        
        return stats
    
    def update_opponent_policy(self, current_performance: float):
        """禁用对手策略自动更新，保持稳定性"""
        self.performance_history.append(current_performance)
        return False  # 不更新对手策略


def run_experiment(env_name, agent_class, config_name, config, seed=42):
    """运行单个实验"""
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
    
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    
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
    
    # 创建保存目录
    safe_name = config_name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
    save_dir = f'./sparse_test_v2/{env_name}_{safe_name}'
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


def plot_three_way_comparison(results, save_dir='./sparse_test_v2'):
    """绘制三方对比图"""
    n_envs = len(results)
    fig = plt.figure(figsize=(18, 5 * n_envs))
    
    colors = {
        'Standard PPO': 'blue', 
        'Dual PPO': 'red',
        'Stable Dual PPO': 'green'
    }
    
    for env_idx, (env_name, env_results) in enumerate(results.items()):
        if not env_results:
            continue
        
        # 训练奖励
        ax1 = plt.subplot(n_envs, 3, env_idx * 3 + 1)
        for algo_name, history in env_results.items():
            rewards = history['episode_rewards']
            if len(rewards) > 0:
                window = min(50, len(rewards) // 10) if len(rewards) > 10 else 1
                if window > 1:
                    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    x = range(window-1, len(rewards))
                    ax1.plot(x, smoothed, linewidth=2.5, label=algo_name, 
                            color=colors.get(algo_name, 'gray'), alpha=0.9)
        
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Reward', fontsize=12)
        ax1.set_title(f'{env_name} - Training Rewards', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 评估奖励
        ax2 = plt.subplot(n_envs, 3, env_idx * 3 + 2)
        for algo_name, history in env_results.items():
            if len(history['eval_rewards']) > 0:
                ax2.plot(history['eval_rewards'], marker='o', linewidth=2.5, 
                        markersize=5, label=algo_name, 
                        color=colors.get(algo_name, 'gray'), alpha=0.9)
        
        ax2.set_xlabel('Evaluation', fontsize=12)
        ax2.set_ylabel('Average Reward', fontsize=12)
        ax2.set_title(f'{env_name} - Evaluation Performance', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 100回合滚动平均
        ax3 = plt.subplot(n_envs, 3, env_idx * 3 + 3)
        for algo_name, history in env_results.items():
            rewards = history['episode_rewards']
            if len(rewards) >= 100:
                rolling_mean = []
                for i in range(len(rewards) - 99):
                    rolling_mean.append(np.mean(rewards[i:i+100]))
                ax3.plot(range(99, len(rewards)), rolling_mean, 
                        linewidth=2.5, label=algo_name, 
                        color=colors.get(algo_name, 'gray'), alpha=0.9)
        
        ax3.set_xlabel('Episode', fontsize=12)
        ax3.set_ylabel('100-Episode Moving Avg', fontsize=12)
        ax3.set_title(f'{env_name} - Stability Comparison', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f'{save_dir}/three_way_comparison.png'
    plt.savefig(save_path, dpi=250, bbox_inches='tight')
    print(f"\n[OK] 三方对比图已保存: {save_path}")


def print_comparison(results):
    """打印对比统计"""
    print("\n" + "="*80)
    print("三方性能对比")
    print("="*80)
    
    for env_name, env_results in results.items():
        if not env_results or len(env_results) == 0:
            continue
        
        print(f"\n{'='*70}")
        print(f"环境: {env_name}")
        print(f"{'='*70}")
        
        # 收集所有算法的最后100回合平均
        final_perfs = {}
        
        for algo_name, history in env_results.items():
            rewards = history['episode_rewards']
            if len(rewards) == 0:
                continue
            
            last_100 = rewards[-100:] if len(rewards) >= 100 else rewards
            final_mean = np.mean(last_100)
            final_std = np.std(last_100)
            final_perfs[algo_name] = final_mean
            
            best_eval = max(history['eval_rewards']) if len(history['eval_rewards']) > 0 else float('-inf')
            
            print(f"\n  {algo_name}:")
            print(f"    最后100回合: {final_mean:.2f} +- {final_std:.2f}")
            print(f"    最佳评估: {best_eval:.2f}")
            print(f"    完成回合: {len(rewards)}")
        
        # 对比分析
        if len(final_perfs) >= 2:
            print(f"\n  [性能排名]")
            sorted_algos = sorted(final_perfs.items(), key=lambda x: -x[1])
            for rank, (algo_name, perf) in enumerate(sorted_algos, 1):
                print(f"    {rank}. {algo_name}: {perf:.2f}")
            
            # 改进百分比
            if 'Standard PPO' in final_perfs:
                baseline = final_perfs['Standard PPO']
                print(f"\n  [相对标准PPO的改进]")
                for algo_name, perf in final_perfs.items():
                    if algo_name != 'Standard PPO':
                        improvement = ((perf - baseline) / abs(baseline)) * 100
                        symbol = '+' if improvement > 0 else ''
                        print(f"    {algo_name}: {symbol}{improvement:.1f}%")


def main():
    print("\n" + "="*80)
    print("稀疏奖励环境基准测试 V2：稳定性改进")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 测试环境
    test_envs = {
        'MountainCar-v0': {
            'max_episodes': 800,
            'update_frequency': 2048,
            'eval_frequency': 50,
            'log_frequency': 50,
        },
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
            'hidden_dim': 256,
        },
        'Dual PPO': {
            'class': DualPolicyPPO,
            'lr': 3e-4,
            'gamma': 0.99,
            'intrinsic_coef': 0.02,
            'kl_coef': 0.001,
            'entropy_coef': 0.02,
            'hidden_dim': 256,
        },
        'Stable Dual PPO': {
            'class': StableDualPPO,
            'lr': 3e-4,
            'gamma': 0.99,
            'intrinsic_coef': 0.02,  # 会自动衰减
            'kl_coef': 0.0001,  # 降低10倍，减少策略偏移
            'entropy_coef': 0.01,  # 降低熵正则，减少随机性
            'hidden_dim': 256,
        },
    }
    
    # 创建结果目录
    os.makedirs('./sparse_test_v2', exist_ok=True)
    
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
                history = run_experiment(
                    env_name,
                    algo_config['class'],
                    algo_name,
                    config,
                    seed=42
                )
                all_results[env_name][algo_name] = history
                
                # 简要统计
                rewards = history['episode_rewards']
                last_100 = rewards[-100:] if len(rewards) >= 100 else rewards
                print(f"\n[OK] {algo_name} 完成")
                print(f"  最后100回合平均: {np.mean(last_100):.2f}")
                if len(history['eval_rewards']) > 0:
                    print(f"  最佳评估: {max(history['eval_rewards']):.2f}")
                
            except Exception as e:
                print(f"\n[X] {algo_name} 失败: {e}")
                import traceback
                traceback.print_exc()
    
    # 绘制对比图
    if any(all_results.values()):
        print("\n" + "="*80)
        print("生成三方对比图")
        print("="*80)
        plot_three_way_comparison(all_results)
    
    # 打印对比统计
    print_comparison(all_results)
    
    print("\n" + "="*80)
    print("测试完成！")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print(f"\n结果保存在: ./sparse_test_v2/")
    
    print("\n" + "="*80)
    print("改进说明")
    print("="*80)
    print("""
Stable Dual PPO的关键改进：
1. 内在奖励系数自动衰减（从0.02衰减到0，600回合内）
   - 早期：高好奇心，快速探索
   - 后期：低好奇心，稳定利用
   
2. KL散度系数降低10倍（0.0001 vs 0.001）
   - 减少与对手策略的偏离
   - 避免灾难性遗忘
   
3. 禁用对手策略自动更新
   - 保持训练目标稳定
   - 避免策略震荡
   
4. 降低熵正则化（0.01 vs 0.02）
   - 后期减少随机探索
   - 提高策略确定性
    """)


if __name__ == '__main__':
    main()


