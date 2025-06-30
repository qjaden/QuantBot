import unittest
import numpy as np
from quantbot.core.factor import FactorEngine
from quantbot.core.data import BarData, DataBuffer
from quantbot.factors.technical import BollingerBands

from datetime import datetime, timedelta


class TestComplexFactor(unittest.TestCase):
    """测试复合因子功能"""
    
    def setUp(self):
        self.buffer_max_size = 150
        self.sample_bars_count = 100
        self.data_buffer = DataBuffer(max_size=self.buffer_max_size)
        self.factor_engine = FactorEngine(self.data_buffer)
        
        # 创建测试数据
        self.sample_bars = self._create_test_bars(self.sample_bars_count)
    
    def _create_test_bars(self, count: int = 100):
        """创建测试Bar数据"""
        bars = []
        base_time = datetime.now()
        base_price = 100.0
        
        for i in range(count):
            # 创建简单的价格走势
            base_price += 0.5 * np.sin(i * 0.1) + 0.1 * np.random.randn()
            base_price = max(base_price, 50.0)  # 确保价格为正
            
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
    
    def test_bollinger_bands_complex_basic(self):
        """测试布林带复合因子基本功能"""
        # 清空数据缓存
        self.data_buffer.clear()
        
        # 注册布林带因子
        bb = BollingerBands(20, 2.0, 'bb_test')
        self.factor_engine.register_factor(bb)
        
        # 验证因子注册正确
        self.assertTrue(self.factor_engine.has_factor('bb_test_upper'))
        self.assertTrue(self.factor_engine.has_factor('bb_test_middle'))
        self.assertTrue(self.factor_engine.has_factor('bb_test_lower'))
        
        # 添加数据并计算因子
        for i, bar in enumerate(self.sample_bars[:30]):
            self.data_buffer.add_bar(bar)
            result = self.factor_engine.calculate_all_factors()
            
            upper_val = result.get("bb_test_upper")
            middle_val = result.get("bb_test_middle")
            lower_val = result.get("bb_test_lower")
            
            if i < 19:  # 布林带期数为20，所以前19个数据点（索引0-18）应该是NaN
                # 期数小于20时，检查是否为NaN或None
                if upper_val is not None:
                    # 如果不是None，那么前19期应该是NaN（布林带期数为20）
                    self.assertTrue(np.isnan(upper_val), f"Expected NaN at period {i}, got {upper_val}")
                if middle_val is not None:
                    self.assertTrue(np.isnan(middle_val), f"Expected NaN at period {i}, got {middle_val}")
                if lower_val is not None:
                    self.assertTrue(np.isnan(lower_val), f"Expected NaN at period {i}, got {lower_val}")
            else:
                # 数据足够时应该有有效值（从第20个数据点开始，索引19）
                if upper_val is not None and middle_val is not None and lower_val is not None:
                    self.assertFalse(np.isnan(upper_val))
                    self.assertFalse(np.isnan(middle_val))
                    self.assertFalse(np.isnan(lower_val))
                    
                    # 上轨应该大于中轨，中轨应该大于下轨
                    self.assertGreater(upper_val, middle_val)
                    self.assertGreater(middle_val, lower_val)
    
    def test_complex_factor_caching(self):
        """测试多输出因子的缓存机制"""
        # 清空数据缓存
        self.data_buffer.clear()
        
        # 注册布林带因子
        bb = BollingerBands(10, 2.0, 'bb_cache_test')
        self.factor_engine.register_factor(bb)

        # 添加足够的数据
        for i in range(15):
            self.data_buffer.add_bar(self.sample_bars[i])
        
        # 第一次计算
        result1 = self.factor_engine.calculate_all_factors()
        
        # 验证可以通过因子实例直接获取结果
        bb_upper = bb.get_result('upper')
        bb_middle = bb.get_result('middle')
        bb_lower = bb.get_result('lower')
        
        # 验证缓存的值与计算结果一致（如果计算结果不为None）
        if result1.get('bb_cache_test_upper') is not None:
            self.assertEqual(bb_upper, result1['bb_cache_test_upper'])
        if result1.get('bb_cache_test_middle') is not None:
            self.assertEqual(bb_middle, result1['bb_cache_test_middle'])
        if result1.get('bb_cache_test_lower') is not None:
            self.assertEqual(bb_lower, result1['bb_cache_test_lower'])
        
        # 验证因子内部缓存大小（现在统一管理，不需要指定output_key）
        self.assertGreater(bb.get_factor_size(), 0)
    
    def test_complex_factor_calculation_consistency(self):
        """测试复合因子计算的一致性"""
        # 清空数据缓存
        self.data_buffer.clear()
        
        # 注册布林带因子
        bb = BollingerBands(5, 1.5, 'bb_consistency')
        self.factor_engine.register_factor(bb)

        # 添加一些数据
        for i in range(10):
            self.data_buffer.add_bar(self.sample_bars[i])
        
        # 计算多次，结果应该一致
        result1 = self.factor_engine.calculate_all_factors()
        result2 = self.factor_engine.calculate_all_factors()
        
        # 只在结果不为None时比较
        if result1.get('bb_consistency_upper') is not None and result2.get('bb_consistency_upper') is not None:
            self.assertEqual(result1['bb_consistency_upper'], result2['bb_consistency_upper'])
        if result1.get('bb_consistency_middle') is not None and result2.get('bb_consistency_middle') is not None:
            self.assertEqual(result1['bb_consistency_middle'], result2['bb_consistency_middle'])
        if result1.get('bb_consistency_lower') is not None and result2.get('bb_consistency_lower') is not None:
            self.assertEqual(result1['bb_consistency_lower'], result2['bb_consistency_lower'])
    
    def test_complex_factor_data_flow(self):
        """测试复合因子的数据流"""
        # 清空数据缓存
        self.data_buffer.clear()
        
        # 注册布林带因子
        bb = BollingerBands(3, 1.0, 'bb_flow')
        self.factor_engine.register_factor(bb)

        # 逐步添加数据，验证数据流
        results = []
        for i in range(6):
            self.data_buffer.add_bar(self.sample_bars[i])
            result = self.factor_engine.calculate_all_factors()
            results.append(result.copy())
        
        # 验证历史数据计数正确
        bar_size = self.data_buffer.get_history_total_bars()
        self.assertEqual(bar_size, 6)
        # 验证因子内部缓存有数据被存储（现在统一管理，不需要指定output_key）
        self.assertGreater(bb.get_history_total_factors(), 0)


if __name__ == '__main__':
    unittest.main()