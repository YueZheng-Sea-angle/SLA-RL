"""
快速测试脚本
===========
用于验证元认知框架的基本功能
"""

import torch
import numpy as np
import gym
from base_algorithms import DQN
from meta_wrapper import SimpleMetaWrapper


def quick_test():
    """快速功能测试"""
    print("="*70)
    print("元认知框架 - 快速功能测试")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n设备: {device}")
    
    # 创建环境
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"环境: {env_name}")
    print(f"状态维度: {state_dim}")
    print(f"动作维度: {action_dim}")
    
    # 测试1: 基础DQN
    print("\n" + "-"*70)
    print("测试 1: 基础DQN")
    print("-"*70)
    
    base_dqn = DQN(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
        lr=1e-3,
        device=device
    )
    
    # 收集一些经验
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    
    for _ in range(100):
        action = base_dqn.select_action(state)
        step_result = env.step(action)
        if len(step_result) == 4:
            next_state, reward, done, _ = step_result
        else:
            next_state, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        
        base_dqn.store_transition(state, action, reward, next_state, done)
        state = next_state if not done else env.reset()
        if isinstance(state, tuple):
            state = state[0]
    
    # 更新几次
    print("收集了100个转移，进行更新...")
    for i in range(5):
        loss, _ = base_dqn.update(batch_size=32)
        print(f"  更新 {i+1}: Loss = {loss:.4f}")
    
    print("✓ 基础DQN测试通过")
    
    # 测试2: 元认知DQN
    print("\n" + "-"*70)
    print("测试 2: 元认知DQN")
    print("-"*70)
    
    meta_dqn_base = DQN(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
        lr=1e-3,
        device=device
    )
    
    meta_dqn = SimpleMetaWrapper(
        base_algorithm=meta_dqn_base,
        state_dim=state_dim,
        action_dim=action_dim,
        device=device
    )
    
    # 收集经验
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    
    episode_reward = 0
    for _ in range(100):
        action = meta_dqn.select_action(state)
        step_result = env.step(action)
        if len(step_result) == 4:
            next_state, reward, done, _ = step_result
        else:
            next_state, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        
        meta_dqn.store_transition(state, action, reward, next_state, done)
        episode_reward += reward
        
        if done:
            meta_dqn.add_episode_reward(episode_reward)
            episode_reward = 0
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
        else:
            state = next_state
    
    # 更新几次
    print("收集了100个转移，进行元认知更新...")
    for i in range(5):
        result = meta_dqn.update(batch_size=32)
        print(f"  更新 {i+1}: Base Loss = {result['base_loss']:.4f}, "
              f"Meta Loss = {result['meta_loss']:.4f}")
    
    print("✓ 元认知DQN测试通过")
    
    # 测试3: 对比性能
    print("\n" + "-"*70)
    print("测试 3: 简短对比测试 (10 episodes)")
    print("-"*70)
    
    def run_episodes(agent, n_episodes=10):
        rewards = []
        for ep in range(n_episodes):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            
            ep_reward = 0
            done = False
            steps = 0
            
            while not done and steps < 500:
                action = agent.select_action(state, eval_mode=True)
                step_result = env.step(action)
                if len(step_result) == 4:
                    next_state, reward, done, _ = step_result
                else:
                    next_state, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                
                ep_reward += reward
                state = next_state
                steps += 1
            
            rewards.append(ep_reward)
        
        return rewards
    
    base_rewards = run_episodes(base_dqn, n_episodes=10)
    meta_rewards = run_episodes(meta_dqn, n_episodes=10)
    
    print(f"\n基础DQN平均奖励: {np.mean(base_rewards):.2f} ± {np.std(base_rewards):.2f}")
    print(f"元认知DQN平均奖励: {np.mean(meta_rewards):.2f} ± {np.std(meta_rewards):.2f}")
    
    env.close()
    
    print("\n" + "="*70)
    print("✓ 所有测试通过！元认知框架运行正常。")
    print("="*70)
    print("\n提示: 运行 'python test_framework.py' 进行完整实验")


if __name__ == '__main__':
    quick_test()

