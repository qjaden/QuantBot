"""
因子基类和计算引擎

定义因子的基础接口和计算逻辑
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union

# 导入依赖管理器和数据缓存
from .dependency import DependencyManager
from .data import DataBuffer

class Factor(ABC):
    """
    因子基类
    
    所有因子都应该继承此类并实现calculate方法
    因子只负责计算逻辑，不负责数据存储
    
    重要特性：
    - 增量计算：Factor应该根据新增的Bar数据进行增量计算，而不是每次都重新计算整个历史数据
    - 无状态：Factor不应该保存计算状态，所有历史数据都通过DataBuffer获取
    - 依赖管理：通过dependencies声明依赖的数据字段或其他因子
    """

    def __init__(self, name: str, dependencies: Optional[List[str]] = None, output_names: List[str] = None):
        """
        初始化因子
        
        Args:
            name: 因子名称
            dependencies: 依赖的字段或因子列表
            output_names: 输出名称列表（必须！子类必须在自己的__init__中提供）
        """
        if (output_names is None or 
            not isinstance(output_names, list) or 
            len(output_names) == 0):
            raise ValueError(f"Factor '{name}' output_names must be a non-empty list, got: {output_names}")
        
        # 检查output_names中不能有重复的名称
        if len(output_names) != len(set(output_names)):
            raise ValueError(f"Factor '{name}' output_names cannot contain duplicates: {output_names}")
        
        self.name = name
        self.dependencies = dependencies or []
        self.output_names = output_names

    @abstractmethod
    def calculate(self, data_buffer: DataBuffer) -> Union[float, Dict[str, float]]:
        """
        计算因子值（增量计算）
        
        这个方法应该基于最新的Bar数据进行增量计算，只计算最新的因子值。
        历史数据可以通过data_buffer.get_bar_field()和data_buffer.get_factor_data()获取。
        
        Args:
            data_buffer: 统一数据缓存，包含Bar数据和因子数据
            
        Returns:
            单输出因子返回float，多输出因子返回Dict[str, float]
            注意：内部统一处理为Dict结构
            
        实现指南：
        - 只计算并返回最新的因子值，不要计算整个历史序列
        - 如果需要历史数据，使用data_buffer的方法获取所需的回望窗口数据
        - 确保计算逻辑支持增量更新，提高计算效率
        """
        pass
    
    def get_dependencies(self) -> List[str]:
        """获取依赖列表"""
        return self.dependencies.copy()
    
    def __repr__(self) -> str:
        return f"Factor(name='{self.name}', dependencies={self.dependencies})"


class FactorEngine:
    """
    因子计算引擎
    
    管理因子的注册、计算和缓存
    """

    def __init__(self, data_buffer: DataBuffer):
        """
        初始化因子计算引擎
        
        Args:
            data_buffer: 统一数据缓存
        """
        self.data_buffer = data_buffer
        self.factors: Dict[str, Factor] = {}
        self.calculation_order: List[str] = []
        self.dependency_graph: Dict[str, List[str]] = {}
        self._is_sorted = False

    def register_factor(self, factor: Factor) -> None:
        """
        注册因子
        
        Args:
            factor: 要注册的因子实例
        """
        # 注册因子到数据缓存
        self.data_buffer.register_factor(factor.name, factor.output_names)
        
        # 更新本地的factors引用和依赖图
        for output_name in factor.output_names:
            if len(factor.output_names) == 1:
                # 单输出因子：直接使用因子名称
                full_name = factor.name
            else:
                # 多输出因子：使用因子名_输出名
                full_name = f"{factor.name}_{output_name}"
            
            self.factors[full_name] = factor
            self.dependency_graph[full_name] = factor.get_dependencies()
        
        self._is_sorted = False

    def unregister_factor(self, factor_name: str) -> None:
        """
        注销因子
        
        Args:
            factor_name: 要注销的因子名称
        """
        if factor_name in self.factors:
            # 从本地字典中删除
            del self.factors[factor_name]
            del self.dependency_graph[factor_name]
            self._is_sorted = False

    def get_factor(self, factor_name: str) -> Optional[Factor]:
        """获取因子实例"""
        return self.factors.get(factor_name)
    
    def has_factor(self, factor_name: str) -> bool:
        """检查是否存在指定因子"""
        return factor_name in self.factors
    
    def calculate_factor(self, factor_name: str) -> Dict[str, float]:
        """
        计算指定因子
        
        Args:
            factor_name: 因子名称
            
        Returns:
            最新的全部因子结果，格式为Dict[str, float]
        """
        if factor_name not in self.factors:
            raise ValueError(f"Factor '{factor_name}' not found")

        factor = self.factors[factor_name]

        # 使用DataBuffer的历史总数来判断是否需要重新计算
        bar_total_count = self.data_buffer.get_total_bars()
        factor_total_count = self.data_buffer.get_factor_total_count(factor.name)

        try:
            # 检查是否已经计算过这次bar
            if factor_total_count == bar_total_count:
                # 已经计算过，直接返回所有结果
                return self.data_buffer.get_latest_factor_result(factor.name)

            # 计算因子的所有输出
            results = factor.calculate(self.data_buffer)

            # 规范化results为Dict格式（如果calculate返回的是单个float值）
            if isinstance(results, (int, float)):
                if len(factor.output_names) == 1:
                    results = {factor.output_names[0]: float(results)}
                else:
                    raise ValueError(f"Factor '{factor.name}' has multiple outputs but calculate() returned a single value")

            # 将结果存储到DataBuffer
            self.data_buffer.add_factor_result(factor.name, results)

            # 返回最新的全部因子结果
            return self.data_buffer.get_latest_factor_result(factor.name)

        except Exception as e:
            print(f"Error calculating factor '{factor_name}': {e}")
            # 返回空字典或None值字典
            return {output_name: None for output_name in factor.output_names}

    def calculate_all_factors(self) -> Dict[str, float]:
        """
        计算所有因子
        
        Returns:
            所有因子的计算结果
        """
        if not self._is_sorted:
            self._topological_sort()

        results = {}

        for factor_name in self.calculation_order:
            result = self.calculate_factor(factor_name)
            results[factor_name] = result

        return results

    def _topological_sort(self) -> None:
        """
        拓扑排序确定计算顺序
        """
        
        # 构建依赖图
        dependency_manager = DependencyManager()
        
        # 添加所有基础数据字段
        basic_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'amount']
        for field in basic_fields:
            dependency_manager.add_node(field)
        
        # 添加所有因子
        for factor_name in self.factors:
            dependency_manager.add_node(factor_name)
        
        # 添加依赖关系
        for factor_name, dependencies in self.dependency_graph.items():
            for dep in dependencies:
                dependency_manager.add_dependency(factor_name, dep)
        
        # 执行拓扑排序
        try:
            sorted_nodes = dependency_manager.topological_sort()
            # 只保留注册的因子
            self.calculation_order = [node for node in sorted_nodes if node in self.factors]
            self._is_sorted = True
        except Exception as e:
            raise ValueError(f"Failed to sort factors: {e}")
    
    def get_calculation_order(self) -> List[str]:
        """获取计算顺序"""
        if not self._is_sorted:
            self._topological_sort()
        return self.calculation_order.copy()
    
    def reset_all_factors(self) -> None:
        """重置所有因子的计算状态"""
        # 清空所有因子的内部缓存
        self.data_buffer.clear_all_factors()
        self._is_sorted = False
    
    def get_factor_list(self) -> List[str]:
        """获取所有注册的因子名称"""
        return list(self.factors.keys())
    
    def get_dependency_info(self) -> Dict[str, List[str]]:
        """获取依赖关系信息"""
        return self.dependency_graph.copy()