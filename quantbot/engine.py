"""
QuantBot主引擎

整合所有组件，提供统一的接口
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import time
import warnings

from .core.data import BarData, DataBuffer
from .core.factor import Factor, FactorEngine
from .core.dependency import DependencyManager


class QuantBotEngine:
    """
    QuantBot主引擎
    
    整合数据缓存、因子计算引擎和依赖管理
    支持高效的因子计算和并行处理
    """
    
    def __init__(self, buffer_size: int = 1000):
        """
        初始化引擎
        
        Args:
            buffer_size: 数据缓存大小
        """
        self.buffer_size = buffer_size
        
        # 初始化核心组件
        self.data_buffer = DataBuffer(buffer_size)
        self.factor_engine = FactorEngine(self.data_buffer)
        self.dependency_manager = DependencyManager()
        
        # 计算统计信息
        self.calculation_count = 0
        self.last_calculation_time = None
        self.total_calculation_time = 0.0
        self.average_calculation_time = 0.0
        
        # 错误追踪
        self.error_count = 0
        self.last_error = None
        self.error_history = []  # 最近的错误历史

    def register_factor(self, factor: Factor) -> None:
        """
        注册因子
        
        Args:
            factor: 要注册的因子实例
            
        Raises:
            ValueError: 当因子已存在或依赖存在问题时
        """
        try:
            # 检查因子名称冲突
            if hasattr(factor, 'output_names'):
                for output_name in factor.output_names:
                    if len(factor.output_names) == 1:
                        full_name = factor.name
                    else:
                        full_name = f"{factor.name}_{output_name}"
                    
                    if self.factor_engine.has_factor(full_name):
                        raise ValueError(f"Factor output '{full_name}' already exists")
            
            # 注册因子
            self.factor_engine.register_factor(factor)
            
            # 检查循环依赖
            if self.has_circular_dependency():
                # 如果存在循环依赖，回滚注册
                self.factor_engine.unregister_factor(factor.name)
                raise ValueError(f"Registering factor '{factor.name}' would create circular dependency")
            
            # 同步更新依赖管理器
            self._update_dependency_manager()
            
        except Exception as e:
            self._log_error(f"Failed to register factor '{factor.name}': {e}")
            raise
    
    def unregister_factor(self, factor_name: str) -> None:
        """
        注销因子
        
        Args:
            factor_name: 要注销的因子名称
        """
        self.factor_engine.unregister_factor(factor_name)
        
        # 重新构建依赖图
        self._rebuild_dependency_manager()

    def add_bar(self, bar: BarData) -> None:
        """
        添加新的Bar数据
        
        Args:
            bar: Bar数据
            
        Raises:
            ValueError: 当Bar数据无效时
        """
        try:
            # 验证Bar数据
            if not self._validate_bar_data(bar):
                raise ValueError(f"Invalid bar data: {bar}")
            
            self.data_buffer.add_bar(bar)
            
        except Exception as e:
            self._log_error(f"Failed to add bar data: {e}")
            raise
    
    def calculate_factors(self) -> Dict[str, float]:
        """
        计算所有因子
            
        Returns:
            所有因子的计算结果
            
        Raises:
            RuntimeError: 当计算过程出现严重错误时
        """
        start_time = time.perf_counter()
        
        try:
            # 检查是否有数据
            if self.data_buffer.get_bar_size() == 0:
                warnings.warn("No data available for factor calculation")
                return {}
            
            # 检查是否有注册的因子
            if len(self.factor_engine.factors) == 0:
                warnings.warn("No factors registered for calculation")
                return {}
            
            # 使用顺序计算
            results = self.factor_engine.calculate_all_factors()

            # 更新统计信息
            calc_time = time.perf_counter() - start_time
            self.calculation_count += 1
            self.last_calculation_time = calc_time
            self.total_calculation_time += calc_time
            self.average_calculation_time = self.total_calculation_time / self.calculation_count
            
            return results
            
        except Exception as e:
            self._log_error(f"Factor calculation failed: {e}")
            raise RuntimeError(f"Factor calculation failed: {e}") from e

    def _update_dependency_manager(self) -> None:
        """更新依赖管理器"""
        # 清空现有依赖
        self.dependency_manager.clear()
        
        # 添加基础数据字段
        basic_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'amount']
        for field in basic_fields:
            self.dependency_manager.add_node(field)
        
        # 添加因子依赖
        for factor_name, factor in self.factor_engine.factors.items():
            self.dependency_manager.add_node(factor_name)
            for dependency in factor.get_dependencies():
                self.dependency_manager.add_dependency(factor_name, dependency)
    
    def _rebuild_dependency_manager(self) -> None:
        """重新构建依赖管理器"""
        self._update_dependency_manager()
    
    def get_factor_value(self, factor_name: str, lookback: int = 1, output_key: str = None) -> np.ndarray:
        """
        获取因子值（增强版，支持多输出因子）
        
        Args:
            factor_name: 因子名称
            lookback: 回望期数
            output_key: 对于多输出因子，指定获取哪个输出
            
        Returns:
            因子数据
            
        Raises:
            ValueError: 当因子不存在时
        """
        try:
            # 检查因子是否存在
            if not self.data_buffer.has_factor(factor_name):
                raise ValueError(f"Factor '{factor_name}' not found")
            
            # 从DataBuffer获取因子数据
            return self.data_buffer.get_factor_data(factor_name, lookback, output_key)
                
        except Exception as e:
            self._log_error(f"Failed to get factor value '{factor_name}': {e}")
            raise
    
    def get_basic_data(self, field_name: str, lookback: int = 1) -> np.ndarray:
        """
        获取基础数据
        
        Args:
            field_name: 字段名称
            lookback: 回望期数
            
        Returns:
            基础数据
        """
        return self.data_buffer.get_bar_field(field_name, lookback)
    
    def get_latest_bar(self) -> Optional[BarData]:
        """获取最新的Bar数据"""
        return self.data_buffer.get_latest_bar()
    
    def get_calculation_order(self) -> List[str]:
        """获取因子计算顺序"""
        return self.factor_engine.get_calculation_order()
    
    def get_dependency_info(self) -> Dict[str, List[str]]:
        """获取依赖关系信息"""
        return self.factor_engine.get_dependency_info()
    
    def get_registered_factors(self) -> List[str]:
        """获取已注册的因子列表"""
        return self.factor_engine.get_factor_list()
    
    def has_circular_dependency(self) -> bool:
        """检查是否存在循环依赖"""
        self._update_dependency_manager()
        return self.dependency_manager.has_cycle()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取引擎统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'buffer_size': self.buffer_size,
            'current_data_size': self.data_buffer.get_bar_size(),
            'registered_factors': len(self.factor_engine.factors),
            'calculation_count': self.calculation_count,
            'last_calculation_time': self.last_calculation_time,
            'total_calculation_time': self.total_calculation_time,
            'average_calculation_time': self.average_calculation_time,
            'has_circular_dependency': self.has_circular_dependency(),
            'error_count': self.error_count,
            'last_error': self.last_error
        }
    
    def reset(self) -> None:
        """重置引擎状态"""
        self.data_buffer.clear_all()
        self.factor_engine.reset_all_factors()
        self.calculation_count = 0
        self.last_calculation_time = None
    
    def shutdown(self) -> None:
        """关闭引擎，释放资源"""
        pass  # 无需释放资源
    
    def __del__(self):
        """析构函数，确保线程池被正确关闭"""
        try:
            self.shutdown()
        except:
            pass
    
    def _validate_bar_data(self, bar: BarData) -> bool:
        """
        验证Bar数据的有效性
        
        Args:
            bar: Bar数据
            
        Returns:
            是否有效
        """
        try:
            # 检查必要字段是否存在
            if not hasattr(bar, 'timestamp') or bar.timestamp is None:
                return False
            if not hasattr(bar, 'close') or bar.close is None or np.isnan(bar.close):
                return False
            
            # 检查OHLC逻辑
            if hasattr(bar, 'high') and hasattr(bar, 'low') and hasattr(bar, 'open'):
                if bar.high < bar.low or bar.close > bar.high or bar.close < bar.low:
                    return False
                if bar.open > bar.high or bar.open < bar.low:
                    return False
            
            # 检查成交量和成交额
            if hasattr(bar, 'volume') and bar.volume < 0:
                return False
            if hasattr(bar, 'amount') and bar.amount < 0:
                return False
                
            return True
            
        except Exception:
            return False
    
    def _log_error(self, error_msg: str) -> None:
        """
        记录错误信息
        
        Args:
            error_msg: 错误消息
        """
        self.error_count += 1
        self.last_error = error_msg
        self.error_history.append({
            'timestamp': datetime.now(),
            'message': error_msg
        })
        
        # 保持最近50个错误记录
        if len(self.error_history) > 50:
            self.error_history = self.error_history[-50:]
    
    def get_factor_instances(self) -> Dict[str, Factor]:
        """
        获取所有因子实例的字典
        
        Returns:
            因子名到因子实例的映射
        """
        unique_factors = {}
        for factor_name, factor in self.factor_engine.factors.items():
            # 对于多输出因子，只保留一个实例
            base_name = factor.name
            if base_name not in unique_factors:
                unique_factors[base_name] = factor
        return unique_factors
    
    def get_factor_result(self, factor_name: str, output_key: str = None) -> Union[float, Dict[str, float], None]:
        """
        获取指定因子的最新结果
        
        Args:
            factor_name: 因子名称（基础名称，不包含输出后缀）
            output_key: 对于多输出因子，指定获取哪个输出；None表示获取所有输出
            
        Returns:
            因子结果值或字典
        """
        try:
            # 检查因子是否存在
            if not self.data_buffer.has_factor(factor_name):
                return None
            
            # 从DataBuffer获取最新结果
            latest_result = self.data_buffer.get_latest_factor_result(factor_name)
            
            if output_key is not None:
                return latest_result.get(output_key)
            else:
                return latest_result
                
        except Exception as e:
            self._log_error(f"Failed to get factor result '{factor_name}': {e}")
            return None
    
    def clear_error_history(self) -> None:
        """
        清空错误历史记录
        """
        self.error_history.clear()
        self.error_count = 0
        self.last_error = None
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        获取引擎健康状态
        
        Returns:
            健康状态信息
        """
        total_calculations = self.calculation_count
        error_rate = self.error_count / total_calculations if total_calculations > 0 else 0.0
        
        health_score = 1.0 - min(error_rate, 1.0)  # 基于错误率计算健康分数
        
        status = "healthy"
        if error_rate > 0.1:  # 错误率超过10%
            status = "unhealthy"
        elif error_rate > 0.05:  # 错误率超过5%
            status = "warning"
        
        return {
            'status': status,
            'health_score': health_score,
            'error_rate': error_rate,
            'total_calculations': total_calculations,
            'recent_errors': self.error_history[-5:] if self.error_history else [],
            'has_circular_dependency': self.has_circular_dependency(),
            'memory_usage': self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> Dict[str, int]:
        """
        估算内存使用情况
        
        Returns:
            内存使用估算
        """
        try:
            data_buffer_size = self.data_buffer.get_bar_size() * 8 * 7  # 假设每个Bar有7个float64字段
            
            factor_cache_size = 0
            registered_factors = self.data_buffer.get_registered_factors()
            for factor_name in registered_factors:
                factor_size = self.data_buffer.get_factor_size(factor_name)
                output_names = self.data_buffer.get_factor_output_names(factor_name)
                factor_cache_size += factor_size * len(output_names) * 8  # float64
            
            return {
                'data_buffer_bytes': data_buffer_size,
                'factor_cache_bytes': factor_cache_size,
                'total_bytes': data_buffer_size + factor_cache_size
            }
        except Exception:
            return {'error': 'Unable to estimate memory usage'}
    
    def __repr__(self) -> str:
        return f"QuantBotEngine(factors={len(self.factor_engine.factors)}, buffer_size={self.buffer_size}, health={self.get_health_status()['status']})"