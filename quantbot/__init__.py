"""
QuantBot - 量化回测框架

支持强大的因子依赖关系功能，允许创建依赖于其他因子或基础数据的复杂因子。
系统会自动处理依赖关系的拓扑排序和分层计算，确保因子按正确的顺序计算。
底层采用numpy.array作为数据缓存，实现因子指标的高效计算。
"""

from .core.data import BarData, DataBuffer
from .core.factor import Factor, FactorEngine
from .core.dependency import DependencyManager
from .factors.technical import SMA, EMA, RSI, MACD
from .engine import QuantBotEngine

__version__ = "1.0.0"

__all__ = [
    "BarData",
    "DataBuffer",
    "Factor",
    "FactorEngine",
    "DependencyManager",
    "SMA",
    "EMA", 
    "RSI",
    "MACD",
    "QuantBotEngine"
]