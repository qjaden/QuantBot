import unittest
import numpy as np
from quantbot.core.factor import FactorEngine
from quantbot.core.data import BarData, DataBuffer
from quantbot.factors.technical import SMA, BollingerBands

from datetime import datetime, timedelta


class TestNewFactorInterface(unittest.TestCase):
    """测试新的Factor接口"""
    
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
    
    def test_single_output_factor(self):
        """测试单输出因子（SMA）"""
        # 创建SMA因子
        sma_5 = SMA(5, 'sma_5_close', max_cache_size=50)
        
        # 验证因子属性
        self.assertEqual(sma_5.name, 'sma_5_close')
        self.assertEqual(sma_5.max_cache_size, 50)
        self.assertEqual(len(sma_5.output_names), 1)  # 单输出因子
        self.assertEqual(sma_5.output_names, ['default'])  # 现在统一使用列表格式
        
        # 添加数据并手动计算
        for i, bar in enumerate(self.sample_bars[:10]):
            self.data_buffer.add_bar(bar)
            
            if i >= 4:  # SMA需要5个数据点
                result = sma_5.calculate(self.data_buffer)
                self.assertFalse(np.isnan(result))
                
                # 手动将结果添加到因子缓存
                sma_5.add_result(result)
                
                # 验证可以获取结果
                latest_result = sma_5.get_result()
                self.assertEqual(latest_result, result)
                
                # 验证缓存大小
                self.assertEqual(sma_5.get_factor_size(), i - 3)  # 从第5个数据点开始有结果
    
    def test_multi_output_factor(self):
        """测试多输出因子（BollingerBands）"""
        # 创建布林带因子
        bb = BollingerBands(20, 2.0, 'bb_20_2', max_cache_size=50)
        
        # 验证因子属性
        self.assertEqual(bb.name, 'bb_20_2')
        self.assertEqual(bb.max_cache_size, 50)
        self.assertEqual(bb.output_names, ['upper', 'middle', 'lower'])
        
        # 添加数据并手动计算
        for i, bar in enumerate(self.sample_bars[:25]):
            self.data_buffer.add_bar(bar)
            
            if i >= 19:  # 布林带需要20个数据点
                result = bb.calculate(self.data_buffer)
                self.assertIsInstance(result, dict)
                self.assertIn('upper', result)
                self.assertIn('middle', result)
                self.assertIn('lower', result)
                
                # 手动将结果添加到因子缓存
                bb.add_result(result)
                
                # 验证可以获取各个输出的结果
                upper_result = bb.get_result('upper')
                middle_result = bb.get_result('middle')
                lower_result = bb.get_result('lower')
                
                self.assertEqual(upper_result, result['upper'])
                self.assertEqual(middle_result, result['middle'])
                self.assertEqual(lower_result, result['lower'])
                
                # 验证布林带的关系：上轨 > 中轨 > 下轨
                self.assertGreater(upper_result, middle_result)
                self.assertGreater(middle_result, lower_result)
                
                # 验证获取所有结果
                all_results = bb.get_all_results()
                self.assertIsInstance(all_results, dict)
                self.assertEqual(all_results['upper'], upper_result)
                self.assertEqual(all_results['middle'], middle_result)
                self.assertEqual(all_results['lower'], lower_result)
    
    def test_factor_engine_integration(self):
        """测试Factor与FactorEngine的集成"""
        # 创建因子
        sma_5 = SMA(5, 'sma_5_close', max_cache_size=50)
        bb = BollingerBands(10, 2.0, 'bb_10_2', max_cache_size=50)
        
        # 注册因子
        self.factor_engine.register_factor(sma_5)
        self.factor_engine.register_factor(bb)
        
        # 验证注册成功
        self.assertTrue(self.factor_engine.has_factor('sma_5_close'))
        self.assertTrue(self.factor_engine.has_factor('bb_10_2_upper'))
        self.assertTrue(self.factor_engine.has_factor('bb_10_2_middle'))
        self.assertTrue(self.factor_engine.has_factor('bb_10_2_lower'))
        
        # 添加数据并计算
        for i, bar in enumerate(self.sample_bars[:15]):
            self.data_buffer.add_bar(bar)
            result = self.factor_engine.calculate_all_factors()
            
            if i >= 9:  # 布林带需要10个数据点
                # 验证所有因子都有结果
                self.assertIsNotNone(result.get('sma_5_close'))
                self.assertIsNotNone(result.get('bb_10_2_upper'))
                self.assertIsNotNone(result.get('bb_10_2_middle'))
                self.assertIsNotNone(result.get('bb_10_2_lower'))
                
                # 验证可以通过因子实例直接获取结果
                sma_direct = sma_5.get_result()
                bb_upper_direct = bb.get_result('upper')
                
                self.assertEqual(sma_direct, result['sma_5_close'])
                self.assertEqual(bb_upper_direct, result['bb_10_2_upper'])
    
    def test_factor_cache_functionality(self):
        """测试因子的缓存功能"""
        # 创建SMA因子
        sma_3 = SMA(3, 'sma_3_close', max_cache_size=5)  # 小缓存测试循环覆盖
        
        # 添加数据
        for i, bar in enumerate(self.sample_bars[:8]):
            self.data_buffer.add_bar(bar)
            
            if i >= 2:  # SMA需要3个数据点
                result = sma_3.calculate(self.data_buffer)
                sma_3.add_result(result)
        
        # 验证缓存大小（应该是5，因为max_cache_size=5）
        self.assertEqual(sma_3.get_factor_size(), 5)
        
        # 验证历史总数（应该是6，因为计算了6次）
        self.assertEqual(sma_3.get_history_total_factors(), 6)
        
        # 验证可以获取历史数据
        history_data = sma_3.get_factor(3)  # 获取最近3个数据
        self.assertEqual(len(history_data), 3)
        
        # 验证清空缓存
        sma_3.clear_cache()
        self.assertEqual(sma_3.get_factor_size(), 0)
        self.assertEqual(sma_3.get_history_total_factors(), 0)


if __name__ == '__main__':
    unittest.main()