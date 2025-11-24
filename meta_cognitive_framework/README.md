# 元认知框架 (Meta-Cognitive Framework)

## 🎯 核心创想

将**好奇心**从一个"算法组件"升级为一个"优化框架"，作为一种**元认知能力**来引导任何基础RL算法的学习过程。

### 核心思想

> 好奇心不仅是探索的驱动力，更是算法对"自身学习过程"的思考能力。

这个框架将好奇心抽象为一种元认知能力——**算法对自身学习过程的评估和优化**。它可以作为一个外部优化器，引导任何基础RL算法（如DQN、SAC等）更好地学习。

## 🏗️ 框架架构

### 双层结构设计

```
┌─────────────────────────────────────────────────┐
│           好奇心评价器 (元层)                    │
│   - 评估数据批次的学习价值                      │
│   - 输出权重 w ∈ [0, 2]                         │
│   - 自监督训练（拟合优势函数提升）              │
└──────────────┬──────────────────────────────────┘
               │ 权重 w
               ↓
┌─────────────────────────────────────────────────┐
│         基础RL算法 (底层)                       │
│   - DQN / SAC / 任何RL算法                     │
│   - 传统损失函数 L_base                        │
│   - 加权更新: L_total = w * L_base             │
└─────────────────────────────────────────────────┘
```

### 工作流程

1. **数据采样**: 从回放池采样一个批次 `(s, a, r, s')`
2. **计算梯度**: 底层算法计算损失和梯度 `g`
3. **元评估**: 好奇心评价器评估学习价值 `w = Evaluator(s, a, g)`
4. **加权更新**: 使用权重调整损失 `L_total = w * L_base`
5. **元学习**: 训练评价器拟合优势提升 `Target_w = clip(A_new - A_old, 0, 2)`

## 📁 文件结构

```
meta_cognitive_framework/
├── README.md                    # 本文档
├── curiosity_evaluator.py       # 好奇心评价器实现
├── base_algorithms.py           # 基础RL算法（DQN, SAC）
├── meta_wrapper.py              # 元认知框架包装器
├── test_framework.py            # 测试脚本
└── results/                     # 实验结果（自动生成）
```

## 🚀 快速开始

### 1. 运行实验

```bash
cd meta_cognitive_framework
python test_framework.py
```

### 2. 使用示例

```python
from base_algorithms import DQN
from meta_wrapper import SimpleMetaWrapper

# 创建基础DQN
base_dqn = DQN(state_dim=4, action_dim=2, device='cpu')

# 用元认知框架包装
meta_agent = SimpleMetaWrapper(
    base_algorithm=base_dqn,
    state_dim=4,
    action_dim=2,
    device='cpu'
)

# 使用与普通算法相同的接口
action = meta_agent.select_action(state)
meta_agent.store_transition(state, action, reward, next_state, done)
meta_agent.update(batch_size=64)
```

## 🔬 实验设计

### 测试算法

1. **基础DQN** vs **元认知DQN**
   - 环境: CartPole-v1, Acrobot-v1
   - 对比指标: 训练速度、稳定性、最终性能

2. **基础SAC** vs **元认知SAC** (可扩展)
   - 环境: 连续控制任务
   - 验证框架的通用性

### 评估指标

- **训练曲线**: 回合奖励随训练进程的变化
- **评估性能**: 定期评估的平均奖励
- **学习效率**: 达到目标性能所需的样本数
- **稳定性**: 性能的方差和波动
- **元损失**: 好奇心评价器的训练损失

## 💡 设计亮点

### 1. 通用性
- 可以包装**任何**基础RL算法
- 不修改底层算法的核心逻辑
- 即插即用的元优化器

### 2. 自监督学习
- 好奇心评价器通过优势函数提升自我训练
- 无需额外的人工标注
- 自适应学习"什么样的数据值得学习"

### 3. 理论优雅性
- 将好奇心提升到"元认知"层面
- 从算法组件升级为优化框架
- 体现了"learning to learn"的思想

### 4. 实现灵活性
- 提供完整版和简化版两种实现
- 可根据需求调整评价器复杂度
- 支持离散和连续动作空间

## 📊 预期效果

### 好奇心评价器的作用

1. **识别高价值数据**: 给能带来大幅性能提升的数据更高权重
2. **过滤低质量样本**: 降低无效或有害样本的影响
3. **加速收敛**: 让算法专注于最有学习价值的经验
4. **提升稳定性**: 通过智能采样减少训练波动

### 理想的学习曲线

```
奖励
  ↑
  │     ╱─────── Meta-Cognitive (更快、更稳定)
  │   ╱
  │ ╱
  │╱
  │─────── Base (较慢、波动大)
  │
  └────────────────────────────────────→ 训练步数
```

## 🔧 技术细节

### 好奇心评价器网络结构

```python
# 完整版（使用梯度信息）
Input: (state, action, gradient_stats)
  ↓
State Encoder (MLP) → state_features
Action Encoder (MLP) → action_features  
Gradient Encoder (MLP) → gradient_features
  ↓
Concatenate → combined_features
  ↓
Value Network (MLP) → weight ∈ [0, 2]
```

```python
# 简化版（仅状态-动作）
Input: (state, action)
  ↓
MLP → weight ∈ [0, 2]
```

### 优势函数提升的计算

```python
# 目标权重
Target_w = clip(A_new(s) - A_old(s), 0, 2)

# 对于DQN，使用Q值作为优势的代理
A(s,a) ≈ Q(s,a)

# 对于SAC，使用V值
A(s,a) ≈ V(s)
```

### 训练策略

- **元评价器更新频率**: 每次底层算法更新后
- **目标权重计算**: 基于值函数的历史变化
- **权重范围**: [0, 2]，其中1.0为中性
  - w > 1: 增强学习（高价值数据）
  - w < 1: 减弱学习（低价值数据）
  - w ≈ 1: 正常学习（中性数据）

## 🎓 理论基础

### 元学习视角

这个框架体现了"元学习"（Meta-Learning）的思想：
- **元知识**: 好奇心评价器学习"如何评估数据价值"
- **基础学习**: 底层算法学习"如何完成任务"
- **协同优化**: 两层相互促进，共同进步

### 课程学习视角

好奇心评价器本质上在进行"课程学习"（Curriculum Learning）：
- 动态调整样本权重
- 优先学习高价值经验
- 类似于教师安排课程难度

### 注意力机制视角

权重 `w` 类似于注意力机制中的注意力分数：
- 告诉模型"关注什么"
- 不是在空间维度，而是在**数据批次**维度
- 实现了"meta-attention"

## 🔮 未来扩展方向

### 1. 多算法联合测试
- 扩展到A3C、TD3等更多算法
- 验证框架的普适性

### 2. 复杂环境测试
- Atari游戏
- MuJoCo连续控制
- 多智能体环境

### 3. 评价器架构优化
- 引入Transformer
- 使用RNN处理时序信息
- 自适应网络容量

### 4. 理论分析
- 收敛性证明
- 样本复杂度分析
- 与其他元学习方法的对比

### 5. 实际应用
- 机器人控制
- 游戏AI
- 推荐系统

## 📝 引用

如果这个框架对您的研究有帮助，欢迎引用：

```
元认知框架：好奇心作为强化学习的通用元优化器
Meta-Cognitive Framework: Curiosity as a Universal Meta-Optimizer for RL
```

## 📄 许可证

继承主仓库的许可证。

## 👨‍💻 作者

基于RLBoss项目的好奇心机制扩展而来。

---

**核心理念**: *让算法学会思考"什么值得学习"，而不仅仅是"如何学习"。*

