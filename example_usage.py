"""
使用示例：展示双策略PPO的各种使用方式
"""

import torch
import numpy as np
import gymnasium as gym
from dual_policy_ppo import DualPolicyPPO, ActorCriticNetwork, CuriosityModule
from trainer import Trainer, ReplayBuffer


def example_1_basic_usage():
    """示例1: 基础使用 - 训练一个简单的智能体"""
    print("\n" + "="*80)
    print("示例1: 基础使用")
    print("="*80)
    
    # 创建智能体
    agent = DualPolicyPPO(
        state_dim=4,
        action_dim=2,
        continuous=False,
        lr=3e-4,
    )
    
    # 创建训练器
    trainer = Trainer(
        env_name='CartPole-v1',
        agent=agent,
        max_episodes=50,
    )
    
    # 训练
    history = trainer.train()
    trainer.close()
    
    print(f"训练完成！平均奖励: {np.mean(history['episode_rewards']):.2f}")


def example_2_custom_parameters():
    """示例2: 自定义超参数 - 针对特定环境调优"""
    print("\n" + "="*80)
    print("示例2: 自定义超参数（稀疏奖励环境）")
    print("="*80)
    
    # 为稀疏奖励环境配置参数
    agent = DualPolicyPPO(
        state_dim=8,
        action_dim=4,
        continuous=False,
        lr=3e-4,
        gamma=0.99,
        clip_epsilon=0.2,
        intrinsic_coef=0.2,        # 更大的内在奖励，促进探索
        kl_coef=0.01,
        kl_threshold=0.3,
        update_opponent_interval=10,
        hidden_dim=256,
    )
    
    trainer = Trainer(
        env_name='LunarLander-v2',
        agent=agent,
        max_episodes=100,
        update_frequency=2048,
        eval_frequency=10,
    )
    
    history = trainer.train()
    trainer.close()
    
    print(f"最终评估奖励: {history['eval_rewards'][-1]:.2f}")


def example_3_continuous_control():
    """示例3: 连续控制 - Pendulum环境"""
    print("\n" + "="*80)
    print("示例3: 连续控制环境")
    print("="*80)
    
    # 连续动作空间配置
    agent = DualPolicyPPO(
        state_dim=3,
        action_dim=1,
        continuous=True,           # 关键: 启用连续动作
        lr=3e-4,
        intrinsic_coef=0.05,       # 连续控制用较小的内在奖励
        kl_coef=0.02,              # 更强的稳定性约束
        kl_threshold=0.2,
        hidden_dim=256,
    )
    
    trainer = Trainer(
        env_name='Pendulum-v1',
        agent=agent,
        max_episodes=100,
        max_steps_per_episode=200,
    )
    
    history = trainer.train()
    trainer.close()
    
    print(f"训练完成！最近10回合平均: {np.mean(history['episode_rewards'][-10:]):.2f}")


def example_4_save_and_load():
    """示例4: 保存和加载模型"""
    print("\n" + "="*80)
    print("示例4: 模型保存和加载")
    print("="*80)
    
    # 训练并保存
    agent = DualPolicyPPO(state_dim=4, action_dim=2, continuous=False)
    trainer = Trainer('CartPole-v1', agent, max_episodes=50, 
                     save_dir='./results/example_save')
    history = trainer.train()
    trainer.close()
    
    print("✓ 模型已训练并保存")
    
    # 加载模型
    new_agent = DualPolicyPPO(state_dim=4, action_dim=2, continuous=False)
    new_agent.load('./results/example_save/best_model.pth')
    
    print("✓ 模型已加载")
    
    # 测试加载的模型
    env = gym.make('CartPole-v1')
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action, _, _ = new_agent.actor_critic.get_action(state_tensor, deterministic=True)
        state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        total_reward += reward
    
    env.close()
    print(f"测试奖励: {total_reward:.2f}")


def example_5_manual_training_loop():
    """示例5: 手动训练循环 - 更灵活的控制"""
    print("\n" + "="*80)
    print("示例5: 手动训练循环")
    print("="*80)
    
    # 创建环境和智能体
    env = gym.make('CartPole-v1')
    agent = DualPolicyPPO(state_dim=4, action_dim=2, continuous=False)
    buffer = ReplayBuffer()
    
    # 手动训练循环
    n_episodes = 20
    update_freq = 1000
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # 选择动作
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob, value = agent.actor_critic.get_action(state_tensor)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            # 存储经验
            buffer.add(state, action.item(), log_prob.item(), reward, 
                      float(done), value.item(), next_state)
            
            state = next_state
            episode_reward += reward
            
            # 更新策略
            if len(buffer) >= update_freq:
                memory = buffer.get()
                memory = {k: v.to(agent.device) for k, v in memory.items()}
                stats = agent.update(memory)
                buffer.clear()
                
                print(f"  更新 - 策略损失: {stats['policy_loss']:.4f}, "
                      f"KL: {stats['kl_divergence']:.4f}")
        
        print(f"回合 {episode+1}: 奖励 = {episode_reward:.2f}")
    
    env.close()
    print("手动训练完成！")


def example_6_analyze_components():
    """示例6: 分析各个组件 - 理解内部机制"""
    print("\n" + "="*80)
    print("示例6: 组件分析")
    print("="*80)
    
    # 创建组件
    state_dim, action_dim = 4, 2
    
    # 1. Actor-Critic网络
    print("\n1. Actor-Critic网络")
    actor_critic = ActorCriticNetwork(state_dim, action_dim, hidden_dim=128)
    
    # 测试前向传播
    state = torch.randn(1, state_dim)
    dist, value = actor_critic(state)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    
    print(f"   动作分布类型: {type(dist).__name__}")
    print(f"   采样动作: {action.item()}")
    print(f"   对数概率: {log_prob.item():.4f}")
    print(f"   状态价值: {value.item():.4f}")
    
    # 2. 好奇心模块
    print("\n2. 好奇心模块")
    curiosity = CuriosityModule(state_dim, action_dim, hidden_dim=128)
    
    state_t = torch.randn(1, state_dim)
    action_t = torch.randint(0, action_dim, (1,))
    state_t1 = torch.randn(1, state_dim)
    
    curiosity_reward, forward_loss, inverse_loss = curiosity(state_t, action_t, state_t1)
    
    print(f"   好奇心奖励: {curiosity_reward.item():.4f}")
    print(f"   前向损失: {forward_loss.item():.4f}")
    print(f"   逆向损失: {inverse_loss.item():.4f}")
    
    # 3. 双策略系统
    print("\n3. 双策略系统")
    agent = DualPolicyPPO(state_dim, action_dim, continuous=False)
    
    # 计算KL散度
    states = torch.randn(10, state_dim)
    kl_div = agent.compute_kl_divergence(states)
    
    print(f"   平均KL散度: {kl_div.mean().item():.4f}")
    print(f"   KL散度范围: [{kl_div.min().item():.4f}, {kl_div.max().item():.4f}]")
    
    # 4. 内在奖励
    print("\n4. 内在奖励计算")
    states = torch.randn(10, state_dim)
    actions = torch.randint(0, action_dim, (10,))
    next_states = torch.randn(10, state_dim)
    
    intrinsic_rewards = agent.compute_intrinsic_reward(states, actions, next_states)
    
    print(f"   平均内在奖励: {intrinsic_rewards.mean().item():.4f}")
    print(f"   内在奖励std: {intrinsic_rewards.std().item():.4f}")


def example_7_compare_with_without_curiosity():
    """示例7: 对比有无好奇心的效果"""
    print("\n" + "="*80)
    print("示例7: 有无好奇心对比")
    print("="*80)
    
    n_episodes = 50
    
    # 无好奇心（标准PPO）
    print("\n训练标准PPO（无好奇心）...")
    agent_no_curiosity = DualPolicyPPO(
        state_dim=4, action_dim=2, continuous=False,
        intrinsic_coef=0.0  # 关闭内在奖励
    )
    trainer1 = Trainer('CartPole-v1', agent_no_curiosity, max_episodes=n_episodes,
                      save_dir='./results/no_curiosity', log_frequency=10)
    history1 = trainer1.train()
    trainer1.close()
    
    # 有好奇心（双策略PPO）
    print("\n训练双策略PPO（有好奇心）...")
    agent_with_curiosity = DualPolicyPPO(
        state_dim=4, action_dim=2, continuous=False,
        intrinsic_coef=0.1  # 启用内在奖励
    )
    trainer2 = Trainer('CartPole-v1', agent_with_curiosity, max_episodes=n_episodes,
                      save_dir='./results/with_curiosity', log_frequency=10)
    history2 = trainer2.train()
    trainer2.close()
    
    # 对比结果
    print("\n" + "="*80)
    print("对比结果:")
    print(f"标准PPO平均奖励: {np.mean(history1['episode_rewards']):.2f}")
    print(f"双策略PPO平均奖励: {np.mean(history2['episode_rewards']):.2f}")
    print(f"改进: {(np.mean(history2['episode_rewards']) - np.mean(history1['episode_rewards'])):.2f}")
    print("="*80)


def main():
    """运行所有示例"""
    print("\n" + "="*80)
    print("双策略PPO使用示例集合")
    print("="*80)
    
    examples = [
        ("基础使用", example_1_basic_usage),
        ("自定义超参数", example_2_custom_parameters),
        ("连续控制", example_3_continuous_control),
        ("保存和加载", example_4_save_and_load),
        ("手动训练循环", example_5_manual_training_loop),
        ("组件分析", example_6_analyze_components),
        ("有无好奇心对比", example_7_compare_with_without_curiosity),
    ]
    
    print("\n可用示例:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"{i}. {name}")
    
    print("\n选择示例运行 (1-7, 或 'all' 运行全部, 'q' 退出): ", end='')
    choice = input().strip().lower()
    
    if choice == 'q':
        return
    elif choice == 'all':
        for name, func in examples:
            try:
                func()
            except Exception as e:
                print(f"\n示例 '{name}' 运行失败: {e}")
    elif choice.isdigit() and 1 <= int(choice) <= len(examples):
        idx = int(choice) - 1
        name, func = examples[idx]
        try:
            func()
        except Exception as e:
            print(f"\n示例 '{name}' 运行失败: {e}")
    else:
        print("无效选择！")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # 命令行直接指定示例
        example_map = {
            '1': example_1_basic_usage,
            'basic': example_1_basic_usage,
            '2': example_2_custom_parameters,
            'custom': example_2_custom_parameters,
            '3': example_3_continuous_control,
            'continuous': example_3_continuous_control,
            '4': example_4_save_and_load,
            'save': example_4_save_and_load,
            '5': example_5_manual_training_loop,
            'manual': example_5_manual_training_loop,
            '6': example_6_analyze_components,
            'analyze': example_6_analyze_components,
            '7': example_7_compare_with_without_curiosity,
            'compare': example_7_compare_with_without_curiosity,
        }
        
        arg = sys.argv[1].lower()
        if arg in example_map:
            example_map[arg]()
        else:
            print(f"未知示例: {arg}")
            print("可用选项: 1-7, basic, custom, continuous, save, manual, analyze, compare")
    else:
        # 交互式选择
        main()

