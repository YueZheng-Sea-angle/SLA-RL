"""
可视化工具：分析和可视化训练结果
"""

import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def setup_chinese_font():
    """设置中文字体"""
    try:
        # 尝试设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("警告: 无法设置中文字体，可能无法正确显示中文")


def load_training_history(history_path: str) -> dict:
    """加载训练历史"""
    with open(history_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_single_experiment(history_path: str, save_dir: str = None):
    """绘制单个实验的完整分析图"""
    setup_chinese_font()
    
    history = load_training_history(history_path)
    
    # 创建大图
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. 训练奖励（带统计信息）
    ax1 = fig.add_subplot(gs[0, :2])
    episode_rewards = np.array(history['episode_rewards'])
    ax1.plot(episode_rewards, alpha=0.3, color='blue', label='原始奖励')
    
    # 多窗口平滑
    for window in [10, 50, 100]:
        if len(episode_rewards) >= window:
            smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(episode_rewards)), smoothed, 
                    linewidth=2, label=f'平滑(窗口={window})')
    
    ax1.set_xlabel('回合')
    ax1.set_ylabel('奖励')
    ax1.set_title('训练奖励曲线（多尺度平滑）')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加统计信息
    textstr = f'最大奖励: {np.max(episode_rewards):.2f}\n'
    textstr += f'最小奖励: {np.min(episode_rewards):.2f}\n'
    textstr += f'平均奖励: {np.mean(episode_rewards):.2f}\n'
    textstr += f'标准差: {np.std(episode_rewards):.2f}'
    ax1.text(0.98, 0.02, textstr, transform=ax1.transAxes,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. 奖励分布直方图
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(episode_rewards, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(np.mean(episode_rewards), color='red', linestyle='--', 
               linewidth=2, label=f'均值={np.mean(episode_rewards):.1f}')
    ax2.set_xlabel('奖励')
    ax2.set_ylabel('频数')
    ax2.set_title('奖励分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 评估奖励
    ax3 = fig.add_subplot(gs[1, 0])
    if len(history['eval_rewards']) > 0:
        eval_rewards = history['eval_rewards']
        ax3.plot(eval_rewards, marker='o', linewidth=2, markersize=6, color='green')
        ax3.fill_between(range(len(eval_rewards)), eval_rewards, alpha=0.3, color='green')
        ax3.set_xlabel('评估次数')
        ax3.set_ylabel('平均奖励')
        ax3.set_title('评估性能')
        ax3.grid(True, alpha=0.3)
    
    # 4. 损失曲线
    ax4 = fig.add_subplot(gs[1, 1])
    if len(history['training_stats']) > 0:
        policy_losses = [stat['policy_loss'] for stat in history['training_stats']]
        value_losses = [stat['value_loss'] for stat in history['training_stats']]
        ax4.plot(policy_losses, linewidth=2, label='策略损失', color='red')
        ax4.plot(value_losses, linewidth=2, label='价值损失', color='orange')
        ax4.set_xlabel('更新次数')
        ax4.set_ylabel('损失')
        ax4.set_title('训练损失')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. KL散度
    ax5 = fig.add_subplot(gs[1, 2])
    if len(history['training_stats']) > 0:
        kl_divs = [stat['kl_divergence'] for stat in history['training_stats']]
        ax5.plot(kl_divs, linewidth=2, color='purple')
        ax5.axhline(0.3, color='red', linestyle='--', alpha=0.5, label='阈值=0.3')
        ax5.set_xlabel('更新次数')
        ax5.set_ylabel('KL散度')
        ax5.set_title('主策略 vs 对手策略 KL散度')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # 6. 内在奖励
    ax6 = fig.add_subplot(gs[2, 0])
    if len(history['training_stats']) > 0:
        intrinsic_means = [stat['intrinsic_reward_mean'] for stat in history['training_stats']]
        intrinsic_stds = [stat['intrinsic_reward_std'] for stat in history['training_stats']]
        x = range(len(intrinsic_means))
        ax6.plot(x, intrinsic_means, linewidth=2, color='teal', label='均值')
        ax6.fill_between(x,
                        np.array(intrinsic_means) - np.array(intrinsic_stds),
                        np.array(intrinsic_means) + np.array(intrinsic_stds),
                        alpha=0.3, color='teal', label='±1标准差')
        ax6.set_xlabel('更新次数')
        ax6.set_ylabel('内在奖励')
        ax6.set_title('内在奖励统计')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    # 7. 好奇心损失
    ax7 = fig.add_subplot(gs[2, 1])
    if len(history['training_stats']) > 0:
        curiosity_losses = [stat['curiosity_loss'] for stat in history['training_stats']]
        ax7.plot(curiosity_losses, linewidth=2, color='brown')
        ax7.set_xlabel('更新次数')
        ax7.set_ylabel('损失')
        ax7.set_title('好奇心模块损失')
        ax7.grid(True, alpha=0.3)
    
    # 8. 学习进度分析
    ax8 = fig.add_subplot(gs[2, 2])
    if len(episode_rewards) > 100:
        # 将训练分为多个阶段，计算每个阶段的平均表现
        n_stages = 5
        stage_size = len(episode_rewards) // n_stages
        stage_means = []
        stage_labels = []
        
        for i in range(n_stages):
            start = i * stage_size
            end = (i + 1) * stage_size if i < n_stages - 1 else len(episode_rewards)
            stage_mean = np.mean(episode_rewards[start:end])
            stage_means.append(stage_mean)
            stage_labels.append(f'阶段{i+1}')
        
        bars = ax8.bar(range(n_stages), stage_means, color='steelblue', alpha=0.7)
        ax8.set_xticks(range(n_stages))
        ax8.set_xticklabels(stage_labels)
        ax8.set_ylabel('平均奖励')
        ax8.set_title('学习进度分析（按阶段）')
        ax8.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, stage_means)):
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}',
                    ha='center', va='bottom')
    
    plt.suptitle('双策略PPO训练分析报告', fontsize=18, fontweight='bold', y=0.995)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'detailed_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 详细分析图已保存到: {save_path}")
    
    plt.show()


def compare_multiple_experiments(history_paths: dict, save_dir: str = None):
    """对比多个实验"""
    setup_chinese_font()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('多实验对比分析', fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(history_paths)))
    
    for (exp_name, history_path), color in zip(history_paths.items(), colors):
        history = load_training_history(history_path)
        
        # 1. 训练奖励对比
        ax = axes[0, 0]
        episode_rewards = np.array(history['episode_rewards'])
        window = 50
        if len(episode_rewards) >= window:
            smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(episode_rewards)), smoothed, 
                   linewidth=2, label=exp_name, color=color)
        
        # 2. 评估奖励对比
        ax = axes[0, 1]
        if len(history['eval_rewards']) > 0:
            ax.plot(history['eval_rewards'], marker='o', linewidth=2, 
                   markersize=4, label=exp_name, color=color)
        
        # 3. KL散度对比
        ax = axes[1, 0]
        if len(history['training_stats']) > 0:
            kl_divs = [stat['kl_divergence'] for stat in history['training_stats']]
            ax.plot(kl_divs, linewidth=2, label=exp_name, color=color)
        
        # 4. 内在奖励对比
        ax = axes[1, 1]
        if len(history['training_stats']) > 0:
            intrinsic_means = [stat['intrinsic_reward_mean'] for stat in history['training_stats']]
            ax.plot(intrinsic_means, linewidth=2, label=exp_name, color=color)
    
    # 设置标签和标题
    axes[0, 0].set_xlabel('回合')
    axes[0, 0].set_ylabel('奖励（平滑）')
    axes[0, 0].set_title('训练奖励对比')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('评估次数')
    axes[0, 1].set_ylabel('平均奖励')
    axes[0, 1].set_title('评估性能对比')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('更新次数')
    axes[1, 0].set_ylabel('KL散度')
    axes[1, 0].set_title('KL散度对比')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('更新次数')
    axes[1, 1].set_ylabel('内在奖励')
    axes[1, 1].set_title('内在奖励对比')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'multi_experiment_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 多实验对比图已保存到: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='可视化训练结果')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'compare'],
                       help='可视化模式')
    parser.add_argument('--history', type=str,
                       help='训练历史文件路径（单实验模式）')
    parser.add_argument('--experiments', type=str, nargs='+',
                       help='实验名称列表（对比模式）')
    parser.add_argument('--results-dir', type=str, default='./results',
                       help='结果目录')
    parser.add_argument('--save-dir', type=str,
                       help='图表保存目录')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        if not args.history:
            print("错误: 单实验模式需要提供 --history 参数")
            return
        
        plot_single_experiment(args.history, args.save_dir)
    
    elif args.mode == 'compare':
        if not args.experiments:
            print("错误: 对比模式需要提供 --experiments 参数")
            return
        
        # 构建历史文件路径字典
        history_paths = {}
        for exp_name in args.experiments:
            history_path = os.path.join(args.results_dir, exp_name, 'training_history.json')
            if os.path.exists(history_path):
                history_paths[exp_name] = history_path
            else:
                print(f"警告: 找不到实验 {exp_name} 的历史文件: {history_path}")
        
        if history_paths:
            compare_multiple_experiments(history_paths, args.save_dir)
        else:
            print("错误: 没有找到有效的实验历史文件")


if __name__ == '__main__':
    main()

