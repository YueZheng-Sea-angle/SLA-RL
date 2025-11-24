"""
元认知框架 (Meta-Cognitive Framework)
====================================

将好奇心作为元认知能力，引导基础RL算法的学习过程。

主要组件:
- CuriosityEvaluator: 好奇心评价器
- DQN, SAC: 基础RL算法
- MetaCognitiveWrapper: 元认知框架包装器
"""

from .curiosity_evaluator import CuriosityEvaluator, SimplifiedCuriosityEvaluator
from .base_algorithms import DQN, SAC, ReplayBuffer
from .meta_wrapper import MetaCognitiveWrapper, SimpleMetaWrapper

__version__ = '0.1.0'
__all__ = [
    'CuriosityEvaluator',
    'SimplifiedCuriosityEvaluator',
    'DQN',
    'SAC',
    'ReplayBuffer',
    'MetaCognitiveWrapper',
    'SimpleMetaWrapper',
]

