"""
基础数据结构模块

包含Bar数据结构和统一的DataBuffer实现
"""

import numpy as np
from typing import Dict, Any, Optional, List, Union
from datetime import datetime


class BarData:
    """
    Bar数据结构，包含OHLCV数据
    """
    
    def __init__(
        self,
        timestamp: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        amount: float = 0.0
    ):
        self.timestamp = timestamp
        self.open = open_price
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.amount = amount
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'amount': self.amount
        }
    
    def __repr__(self) -> str:
        return f"BarData({self.timestamp}, O:{self.open}, H:{self.high}, L:{self.low}, C:{self.close}, V:{self.volume})"

class BarBuffer:
    """
    Bar数据缓存器
    
    使用循环队列高效存储Bar数据
    """
    
    def __init__(self, max_size: int = 1000):
        """
        初始化Bar缓存器
        
        Args:
            max_size: 最大缓存大小
        """
        self.max_size = max_size
        self.size = 0
        self.head = 0
        self.total_bars = 0
        
        # 使用numpy数组存储各字段数据
        self.timestamps = np.full(max_size, np.datetime64('NaT'), dtype='datetime64[us]')
        self.opens = np.full(max_size, np.nan, dtype=np.float64)
        self.highs = np.full(max_size, np.nan, dtype=np.float64)
        self.lows = np.full(max_size, np.nan, dtype=np.float64)
        self.closes = np.full(max_size, np.nan, dtype=np.float64)
        self.volumes = np.full(max_size, np.nan, dtype=np.float64)
        self.amounts = np.full(max_size, np.nan, dtype=np.float64)
    
    def add_bar(self, bar: BarData) -> None:
        """
        添加新的Bar数据
        
        Args:
            bar: Bar数据对象
        """
        index = (self.head + self.size) % self.max_size
        
        self.timestamps[index] = np.datetime64(bar.timestamp, 'us')
        self.opens[index] = bar.open
        self.highs[index] = bar.high
        self.lows[index] = bar.low
        self.closes[index] = bar.close
        self.volumes[index] = bar.volume
        self.amounts[index] = bar.amount
        
        if self.size < self.max_size:
            self.size += 1
        else:
            self.head = (self.head + 1) % self.max_size
        
        self.total_bars += 1
    
    def get_field(self, field_name: str, lookback: int = 1) -> np.ndarray:
        """
        获取Bar字段数据
        
        Args:
            field_name: 字段名 ('open', 'high', 'low', 'close', 'volume', 'amount', 'timestamp')
            lookback: 回望期数，默认为1（最新数据）
            
        Returns:
            numpy数组，包含指定字段的数据
        """
        if lookback == 0 or self.size == 0:
            return np.array([])
        
        lookback = min(lookback, self.size)
            
        if field_name == 'timestamp':
            data = self.timestamps
        elif field_name == 'open':
            data = self.opens
        elif field_name == 'high':
            data = self.highs
        elif field_name == 'low':
            data = self.lows
        elif field_name == 'close':
            data = self.closes
        elif field_name == 'volume':
            data = self.volumes
        elif field_name == 'amount':
            data = self.amounts
        else:
            raise ValueError(f"Unknown bar field: {field_name}")
        
        # 获取最新的lookback个数据
        if self.size < self.max_size:
            start_idx = max(0, self.size - lookback)
            return data[start_idx:self.size].copy()
        else:
            # 处理循环队列
            start_idx = (self.head + self.size - lookback) % self.max_size
            if start_idx + lookback <= self.max_size:
                return data[start_idx:start_idx + lookback].copy()
            else:
                # 跨越边界的情况
                part1 = data[start_idx:self.max_size]
                part2 = data[0:(start_idx + lookback) % self.max_size]
                return np.concatenate([part1, part2])
    
    def get_latest_bar(self) -> Optional[BarData]:
        """获取最新的Bar数据"""
        if self.size == 0:
            return None
            
        index = (self.head + self.size - 1) % self.max_size
        
        return BarData(
            timestamp=self.timestamps[index].astype('datetime64[us]').astype(datetime),
            open_price=float(self.opens[index]),
            high=float(self.highs[index]),
            low=float(self.lows[index]),
            close=float(self.closes[index]),
            volume=float(self.volumes[index]),
            amount=float(self.amounts[index])
        )
    
    def get_size(self) -> int:
        """获取当前缓存大小"""
        return self.size
    
    def get_total_count(self) -> int:
        """获取历史总Bar数"""
        return self.total_bars
    
    def clear(self) -> None:
        """清空缓存"""
        self.size = 0
        self.head = 0
        self.total_bars = 0
    
    def __repr__(self) -> str:
        return f"BarBuffer(size={self.size}, total={self.total_bars}, max_size={self.max_size})"


class FactorBuffer:
    """
    因子缓存
    """
    def __init__(self, max_size: int, output_names: List[str]):
        self.max_size = max_size
        self.size = 0
        self.head = 0
        self.total_factor = 0
        self.results = {}
        self.output_names = output_names

        for output_name in output_names:
            self.results[output_name] = np.full(max_size, np.nan, dtype=np.float64)

    def add_result(self, result: Dict[str, float]) -> None:
        """
        添加因子计算结果
        """
        if not isinstance(result, dict):
            raise ValueError(f"Factor result must be Dict[str, float]")

        if set(result.keys()) != set(self.output_names):
            raise ValueError(f"Factor result keys {result.keys()} don't match expected {self.output_names}")

        for output_name, value in result.items():
            if self.size < self.max_size:
                self.results[output_name][self.size] = value
            else:
                self.results[output_name][self.head] = value
        
        if self.size < self.max_size:
            self.size += 1
        else:
            self.head = (self.head + 1) % self.max_size

        self.total_factor += 1

    def get_result(self, output_name: str) -> float:
        """
        获取最新的因子计算结果
        """
        if output_name not in self.output_names:
            raise ValueError(f"Output name {output_name} not found")

        if self.size == 0:
            return np.nan
        
        if self.size < self.max_size:
            return self.results[output_name][self.size - 1]
        else:
            latest_idx = (self.head - 1 + self.max_size) % self.max_size
            return self.results[output_name][latest_idx]

    def get_historical_data(self, output_name: str, lookback: int = 1) -> np.ndarray:
        """
        获取历史因子数据
        
        Args:
            output_name: 输出名称
            lookback: 回望期数，默认为1（最新数据）
            
        Returns:
            numpy数组，包含指定的历史数据，按时间顺序排列（最旧到最新）
        """
        if output_name not in self.output_names:
            raise ValueError(f"Output name {output_name} not found")
        
        if lookback == 0 or self.size == 0:
            return np.array([])
        
        lookback = min(lookback, self.size)
        data = self.results[output_name]
        
        if self.size < self.max_size:
            # 缓冲区未满：数据从0到size-1，取最新的lookback个
            start_idx = max(0, self.size - lookback)
            return data[start_idx:self.size].copy()
        else:
            # 缓冲区已满：head指向最旧数据的下一个位置
            # 最新数据的位置是 (head - 1 + max_size) % max_size
            latest_idx = (self.head - 1 + self.max_size) % self.max_size
            start_idx = (latest_idx - lookback + 1 + self.max_size) % self.max_size
            
            if start_idx <= latest_idx:
                # 不跨边界
                return data[start_idx:latest_idx + 1].copy()
            else:
                # 跨边界：从start_idx到末尾，再从0到latest_idx
                part1 = data[start_idx:self.max_size]
                part2 = data[0:latest_idx + 1]
                return np.concatenate([part1, part2])
        

class DataBuffer:
    """
    统一数据缓存管理器
    
    缓存Bar数据和因子数据，因子数据格式为Dict[str, FactorBuffer]
    使用循环队列实现高效的数据存储和访问
    """
    
    def __init__(self, max_size: int = 1000):
        """
        初始化数据缓存管理器
        
        Args:
            max_size: 最大缓存大小
        """
        self.max_size = max_size
        self.bar_buffer = BarBuffer(max_size)
        self.factor_buffer: Dict[str, FactorBuffer] = {}
        
    
    def add_bar(self, bar: BarData) -> None:
        """
        添加新的Bar数据
        
        Args:
            bar: Bar数据对象
        """
        self.bar_buffer.add_bar(bar)
    
    def get_bar_field(self, field_name: str, lookback: int = 1) -> np.ndarray:
        """
        获取Bar字段数据
        
        Args:
            field_name: 字段名 ('open', 'high', 'low', 'close', 'volume', 'amount', 'timestamp')
            lookback: 回望期数，默认为1（最新数据）
            
        Returns:
            numpy数组，包含指定字段的数据
        """
        return self.bar_buffer.get_field(field_name, lookback)
    
    def register_factor(self, factor_name: str, output_names: List[str]) -> None:
        """
        注册因子到缓存管理器
        
        Args:
            factor_name: 因子名称
            output_names: 输出名称列表
        """
        if factor_name in self.factor_buffer:
            return  # 已经注册过
        
        # 创建FactorBuffer实例
        self.factor_buffer[factor_name] = FactorBuffer(self.max_size, output_names)
    
    def add_factor_result(self, factor_name: str, result: Dict[str, float]) -> None:
        """
        添加因子计算结果
        
        Args:
            factor_name: 因子名称
            result: 计算结果，格式为Dict[str, float]
        """
        if factor_name not in self.factor_buffer:
            raise ValueError(f"Factor '{factor_name}' not registered")
        
        self.factor_buffer[factor_name].add_result(result)
    
    def get_factor_data(self, factor_name: str, lookback: int = 1, 
                       output_key: str = None) -> np.ndarray:
        """
        获取因子数据
        
        Args:
            factor_name: 因子名称
            lookback: 回望期数，默认为1（最新数据）
            output_key: 指定输出键，如果因子只有一个输出可省略
            
        Returns:
            numpy数组，包含指定的数据，按时间顺序排列（最旧到最新）
        """
        if factor_name not in self.factor_buffer:
            return np.array([])
        
        factor_buf = self.factor_buffer[factor_name]
        
        # 确定输出键
        if output_key is None:
            if len(factor_buf.output_names) == 1:
                output_key = factor_buf.output_names[0]
            else:
                raise ValueError(
                    f"Factor '{factor_name}' has multiple outputs {factor_buf.output_names}, "
                    f"must specify output_key"
                )
        
        return factor_buf.get_historical_data(output_key, lookback)
    
    def get_latest_bar(self) -> Optional[BarData]:
        """获取最新的Bar数据"""
        return self.bar_buffer.get_latest_bar()
    
    def get_latest_factor_result(self, factor_name: str) -> Dict[str, float]:
        """
        获取因子最新的所有输出结果
        
        Args:
            factor_name: 因子名称
            
        Returns:
            最新结果字典
        """
        if factor_name not in self.factor_buffer:
            return {}
        
        factor_buf = self.factor_buffer[factor_name]
        results = {}
        
        for output_name in factor_buf.output_names:
            try:
                result = factor_buf.get_result(output_name)
                results[output_name] = result if not np.isnan(result) else None
            except Exception:
                results[output_name] = None
        
        return results
    
    def get_bar_size(self) -> int:
        """获取Bar数据大小"""
        return self.bar_buffer.get_size()
    
    def get_total_bars(self) -> int:
        """获取历史总Bar数"""
        return self.bar_buffer.get_total_count()
    
    def get_factor_size(self, factor_name: str) -> int:
        """
        获取因子数据大小
        
        Args:
            factor_name: 因子名称
            
        Returns:
            数据大小
        """
        if factor_name not in self.factor_buffer:
            return 0
        return self.factor_buffer[factor_name].size
    
    def get_factor_total_count(self, factor_name: str) -> int:
        """
        获取因子历史总计算次数
        
        Args:
            factor_name: 因子名称
            
        Returns:
            历史总计算次数
        """
        if factor_name not in self.factor_buffer:
            return 0
        return self.factor_buffer[factor_name].total_factor
    
    def has_factor(self, factor_name: str) -> bool:
        """检查因子是否已注册"""
        return factor_name in self.factor_buffer
    
    def get_factor_output_names(self, factor_name: str) -> List[str]:
        """获取因子的输出名称列表"""
        if factor_name not in self.factor_buffer:
            return []
        return self.factor_buffer[factor_name].output_names.copy()
    
    def get_registered_factors(self) -> List[str]:
        """获取所有已注册的因子名称"""
        return list(self.factor_buffer.keys())
    
    def get_factor_buffer(self, factor_name: str) -> Optional[FactorBuffer]:
        """
        获取指定因子的FactorBuffer
        
        Args:
            factor_name: 因子名称
            
        Returns:
            FactorBuffer实例或None
        """
        return self.factor_buffer.get(factor_name)
    
    def get_factor_result_from_buffer(self, factor_name: str, output_name: str) -> float:
        """
        从FactorBuffer获取因子结果
        
        Args:
            factor_name: 因子名称
            output_name: 输出名称
            
        Returns:
            最新的因子结果值
        """
        if factor_name not in self.factor_buffer:
            return np.nan
        
        return self.factor_buffer[factor_name].get_result(output_name)
    
    def clear_factor(self, factor_name: str) -> None:
        """
        清空指定因子的缓存
        
        Args:
            factor_name: 因子名称
        """
        if factor_name in self.factor_buffer:
            # 重新创建FactorBuffer实现清空
            output_names = self.factor_buffer[factor_name].output_names
            self.factor_buffer[factor_name] = FactorBuffer(self.max_size, output_names)
    
    def clear_all_factors(self) -> None:
        """清空所有因子的缓存"""
        for factor_name in list(self.factor_buffer.keys()):
            self.clear_factor(factor_name)
    
    def clear_bars(self) -> None:
        """清空Bar数据缓存"""
        self.bar_buffer.clear()
    
    def clear_all(self) -> None:
        """清空所有数据缓存"""
        self.clear_bars()
        self.clear_all_factors()
    
    def __repr__(self) -> str:
        return f"DataBuffer(bars={self.bar_buffer.get_size()}, factors={len(self.factor_buffer)}, max_size={self.max_size})"