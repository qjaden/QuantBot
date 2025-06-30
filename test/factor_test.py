import unittest
import numpy as np
from typing import List, Optional
from quantbot.core.factor import Factor, FactorEngine
from quantbot.core.data import BarData, DataBuffer

from datetime import datetime, timedelta

class TestFactor1(Factor):
    """
    测试因子
    """
    def __init__(self, name: str, dependencies: Optional[List[str]] = None):
        super().__init__(name, dependencies)

    def calculate(self, data_buffer: DataBuffer) -> float:
        close = data_buffer.get_field('close', 1)[0]
        return close

class TestFactor2(Factor):
    """
    测试因子, 依赖Test1
    """
    def __init__(self, name: str, dependencies: Optional[List[str]] = None):
        super().__init__(name, dependencies)

    def calculate(self, data_buffer: DataBuffer) -> float:
        close = data_buffer.get_field('close', 1)[0]
        test1 = data_buffer.get_factor('test_factor1', 1)[0]
        return close + test1
    
class TestFactor(unittest.TestCase):
    """测试技术指标因子"""
    
    def setUp(self):
        self.buffer_max_size = 50
        self.sample_bars_count = 100
        self.data_buffer = DataBuffer(max_size=self.buffer_max_size)
        self.factor_engine = FactorEngine(self.data_buffer)
        
        # 创建具有明确价格趋势的测试数据
        self.sample_bars = self._create_trending_bars(self.sample_bars_count)
    
    def _create_trending_bars(self, count: int = 100):
        """创建有趋势的示例Bar数据，便于测试技术指标"""
        bars = []
        base_time = datetime.now()
        base_price = 100.0
        
        # 创建30个数据点，呈现上升趋势
        for i in range(count):
            base_price += 0.5  # 每次上涨0.5
            
            bar = BarData(
                timestamp=base_time + timedelta(minutes=i),
                open_price=base_price - 0.2,
                high=base_price + 0.3,
                low=base_price - 0.4,
                close=base_price,
                volume=1000.0 + i * 10,
                amount=(base_price) * (1000.0 + i * 10)
            )
            bars.append(bar)
        
        return bars
    
    def test_sma_factor(self):
        """测试SMA因子"""
        # 清空数据缓存
        self.data_buffer.clear()
        
        # 注册SMA因子
        test_factor1 = TestFactor1('test_factor1')
        test_factor2 = TestFactor2('test_factor2', dependencies=[test_factor1.name])
        self.factor_engine.register_factor(test_factor1)
        self.factor_engine.register_factor(test_factor2)

        
        # 添加数据并计算因子
        for i, bar in enumerate(self.sample_bars):
            self.data_buffer.add_bar(bar)
            # 测试历史Bar数
            self.assertEqual(self.data_buffer.get_bar_buffer().get_history_total_bars(), i + 1)
            result = self.factor_engine.calculate_all_factors()
            self.assertEqual(len(result), 2)
            self.assertEqual(result['test_factor1'], bar.close)
            self.assertEqual(result['test_factor2'], bar.close * 2)

            self.assertEqual(self.data_buffer.get_history_total_factors('test_factor1'), i + 1)
            self.assertEqual(self.data_buffer.get_history_total_factors('test_factor2'), i + 1)
    
        close = self.data_buffer.get_field('close', 50)
        test_factor1 = self.data_buffer.get_factor('test_factor1', 50)
        test_factor2 = self.data_buffer.get_factor('test_factor2', 50)
        np.testing.assert_array_equal(close, test_factor1)
        np.testing.assert_array_equal(close * 2, test_factor2)

        test_factor1_history_total_count = self.data_buffer.get_history_total_factors('test_factor1')
        test_factor2_history_total_count = self.data_buffer.get_history_total_factors('test_factor2')
        bar_total_count = self.data_buffer.get_history_total_bars()
        self.assertEqual(test_factor1_history_total_count, self.sample_bars_count)
        self.assertEqual(test_factor2_history_total_count, self.sample_bars_count)
        self.assertEqual(bar_total_count, self.sample_bars_count)
