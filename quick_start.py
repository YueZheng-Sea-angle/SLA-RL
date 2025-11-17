"""
快速开始脚本：用最少的代码运行双策略PPO
"""

import torch
import numpy as np
import gymnasium as gym
from dual_policy_ppo import DualPolicyPPO
from trainer import Trainer


def quick_demo():
    """快速演示：在CartPole环境中训练双策略PPO"""
    
    print("\n" + "="*80)
    print("双策略PPO 快速演示")
    print("="*80)
    print("环境: CartPole-v1")
    print("这是一个简单的演示，展示如何使用双策略PPO算法")
    print("="*80 + "\n")
    
    # 1. 创建环境
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    
    # 2. 获取环境信息
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"状态空间维度: {state_dim}")
    print(f"动作空间维度: {action_dim}")
    print(f"动作类型: 离散\n")
    
    env.close()
    
    # 3. 创建双策略PPO智能体
    print("创建双策略PPO智能体...")
    agent = DualPolicyPPO(
        state_dim=state_dim,
        action_dim=action_dim,
        continuous=False,
        lr=3e-4,
        intrinsic_coef=0.1,    # 内在奖励系数
        kl_coef=0.01,          # KL散度系数
        kl_threshold=0.3,      # KL散度阈值
        hidden_dim=128,
    )
    print("✓ 智能体创建完成\n")
    
    # 4. 创建训练器
    print("创建训练器...")
    trainer = Trainer(
        env_name=env_name,
        agent=agent,
        max_episodes=100,       # 快速演示只训练100回合
        max_steps_per_episode=500,
        update_frequency=2048,
        eval_frequency=10,
        save_frequency=50,
        save_dir='./results/quick_demo'
    )
    print("✓ 训练器创建完成\n")
    
    # 5. 开始训练
    print("开始训练...\n")
    history = trainer.train()
    
    # 6. 关闭环境
    trainer.close()
    
    # 7. 显示结果摘要
    print("\n" + "="*80)
    print("训练结果摘要")
    print("="*80)
    
    episode_rewards = history['episode_rewards']
    print(f"训练回合数: {len(episode_rewards)}")
    print(f"平均奖励: {np.mean(episode_rewards):.2f}")
    print(f"最大奖励: {np.max(episode_rewards):.2f}")
    print(f"最近10回合平均: {np.mean(episode_rewards[-10:]):.2f}")
    
    if len(history['eval_rewards']) > 0:
        print(f"最终评估奖励: {history['eval_rewards'][-1]:.2f}")
    
    print("\n结果已保存到: ./results/quick_demo/")
    print("- 最佳模型: best_model.pth")
    print("- 训练历史: training_history.json")
    print("- 训练图表: training_results.png")
    print("="*80)


def custom_experiment():
    """自定义实验：让用户选择环境和参数"""
    
    print("\n" + "="*80)
    print("双策略PPO 自定义实验")
    print("="*80 + "\n")
    
    # 环境选择
    print("请选择环境:")
    print("1. CartPole-v1 (简单)")
    print("2. Pendulum-v1 (连续控制)")
    print("3. LunarLander-v2 (稀疏奖励)")
    
    choice = input("\n请输入选项 (1/2/3, 默认1): ").strip() or "1"
    
    env_configs = {
        "1": {
            "name": "CartPole-v1",
            "continuous": False,
            "intrinsic_coef": 0.1,
            "max_episodes": 200,
        },
        "2": {
            "name": "Pendulum-v1",
            "continuous": True,
            "intrinsic_coef": 0.05,
            "max_episodes": 300,
        },
        "3": {
            "name": "LunarLander-v2",
            "continuous": False,
            "intrinsic_coef": 0.2,
            "max_episodes": 400,
        }
    }
    
    if choice not in env_configs:
        print("无效选项，使用默认环境 CartPole-v1")
        choice = "1"
    
    config = env_configs[choice]
    env_name = config["name"]
    
    print(f"\n已选择环境: {env_name}")
    
    # 创建环境获取信息
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    
    if config["continuous"]:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n
    
    env.close()
    
    print(f"状态维度: {state_dim}")
    print(f"动作维度: {action_dim}")
    print(f"动作类型: {'连续' if config['continuous'] else '离散'}")
    
    # 询问训练回合数
    max_episodes = input(f"\n训练回合数 (默认{config['max_episodes']}): ").strip()
    max_episodes = int(max_episodes) if max_episodes else config['max_episodes']
    
    # 创建智能体
    print("\n创建智能体...")
    agent = DualPolicyPPO(
        state_dim=state_dim,
        action_dim=action_dim,
        continuous=config["continuous"],
        intrinsic_coef=config["intrinsic_coef"],
        hidden_dim=256,
    )
    
    # 创建训练器
    trainer = Trainer(
        env_name=env_name,
        agent=agent,
        max_episodes=max_episodes,
        save_dir=f'./results/{env_name.lower().replace("-", "_")}_custom'
    )
    
    # 训练
    print(f"\n开始训练 {max_episodes} 回合...\n")
    history = trainer.train()
    trainer.close()
    
    # 结果
    print("\n" + "="*80)
    print("训练完成！")
    print(f"最终平均奖励: {np.mean(history['episode_rewards'][-10:]):.2f}")
    print("="*80)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--custom':
        custom_experiment()
    else:
        quick_demo()
    
    print("\n提示:")
    print("- 运行快速演示: python quick_start.py")
    print("- 运行自定义实验: python quick_start.py --custom")
    print("- 可视化结果: python visualize_results.py --mode single --history ./results/quick_demo/training_history.json")

