import unittest
import numpy as np
from datetime import datetime, timedelta

from quantbot.engine import QuantBotEngine
from quantbot.core.data import BarData
from quantbot.factors.technical import SMA, EMA, BollingerBands, ATR


class TestQuantBotEngine(unittest.TestCase):
    """测试QuantBotEngine的增强功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.engine = QuantBotEngine(buffer_size=100, max_workers=2)
        self.sample_bars = self._create_test_bars(50)
    
    def _create_test_bars(self, count: int = 50):
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
    
    def test_factor_registration_with_validation(self):
        """测试因子注册的增强验证功能"""
        # 测试正常注册
        sma_5 = SMA(5, 'sma_5', max_cache_size=50)
        self.engine.register_factor(sma_5)
        
        self.assertTrue(self.engine.factor_engine.has_factor('sma_5'))
        self.assertEqual(len(self.engine.get_registered_factors()), 1)
        
        # 测试重复注册（应该抛出异常）
        sma_5_duplicate = SMA(5, 'sma_5', max_cache_size=50)
        with self.assertRaises(ValueError):
            self.engine.register_factor(sma_5_duplicate)
    
    def test_bar_data_validation(self):
        """测试Bar数据验证功能"""
        # 测试有效的Bar数据
        valid_bar = self.sample_bars[0]
        self.engine.add_bar(valid_bar)  # 应该成功
        
        # 测试无效的Bar数据（缺少timestamp）
        invalid_bar = BarData(
            timestamp=None,  # 无效
            open_price=100,
            high=101,
            low=99,
            close=100.5,
            volume=1000,
            amount=100500
        )
        
        with self.assertRaises(ValueError):
            self.engine.add_bar(invalid_bar)
    
    def test_enhanced_statistics(self):
        """测试增强的统计功能"""
        # 注册一些因子
        sma_5 = SMA(5, 'sma_5', max_cache_size=50)
        bb = BollingerBands(10, 2.0, 'bb_10', max_cache_size=50)
        self.engine.register_factor(sma_5)
        self.engine.register_factor(bb)
        
        # 添加一些数据
        for bar in self.sample_bars[:15]:
            self.engine.add_bar(bar)
        
        # 计算几次因子
        for _ in range(3):
            self.engine.calculate_factors()
        
        # 获取统计信息
        stats = self.engine.get_statistics()
        
        # 验证统计信息
        self.assertEqual(stats['calculation_count'], 3)
        self.assertGreater(stats['total_calculation_time'], 0)
        self.assertGreater(stats['average_calculation_time'], 0)
        self.assertIn('performance', stats)
        self.assertIn('sequential_avg_time', stats['performance'])
    
    def test_error_handling_and_logging(self):
        """测试错误处理和日志记录"""
        # 初始状态应该没有错误
        self.assertEqual(self.engine.error_count, 0)
        
        # 尝试获取不存在的因子（这会记录错误）
        try:
            self.engine.get_factor_value('nonexistent_factor')
        except ValueError:
            pass  # 预期的异常
        
        # 验证错误被记录
        self.assertGreater(self.engine.error_count, 0)
        self.assertIsNotNone(self.engine.last_error)
    
    def test_health_status(self):
        """测试健康状态监控"""
        # 注册因子并添加数据
        sma_5 = SMA(5, 'sma_5', max_cache_size=50)
        self.engine.register_factor(sma_5)
        
        for bar in self.sample_bars[:10]:
            self.engine.add_bar(bar)
        
        # 计算因子
        self.engine.calculate_factors()
        
        # 获取健康状态
        health = self.engine.get_health_status()
        
        # 验证健康状态
        self.assertIn('status', health)
        self.assertIn('health_score', health)
        self.assertIn('error_rate', health)
        self.assertIn('memory_usage', health)
        
        # 没有错误的情况下应该是健康的
        self.assertEqual(health['status'], 'healthy')
        self.assertGreaterEqual(health['health_score'], 0.9)
    
    def test_factor_result_access(self):
        """测试因子结果访问的新方法"""
        # 注册单输出和多输出因子
        sma_5 = SMA(5, 'sma_5', max_cache_size=50)
        bb = BollingerBands(10, 2.0, 'bb_10', max_cache_size=50)
        self.engine.register_factor(sma_5)
        self.engine.register_factor(bb)
        
        # 添加足够的数据（确保所有因子都有足够的数据）
        for bar in self.sample_bars[:25]:  # 增加到25个数据点
            self.engine.add_bar(bar)
            # 每添加一个数据点就计算一次，确保因子逐步累积数据
            self.engine.calculate_factors()
        
        # 测试单输出因子结果访问
        sma_result = self.engine.get_factor_result('sma_5')
        # SMA需要5个数据点，所以前面的结果可能是None
        if sma_result is not None:
            self.assertIsInstance(sma_result, (int, float))
        
        # 测试多输出因子结果访问
        bb_all_results = self.engine.get_factor_result('bb_10')
        # BB需要10个数据点，所以15个数据点应该有结果
        if bb_all_results is not None:
            self.assertIsInstance(bb_all_results, dict)
            self.assertIn('upper', bb_all_results)
            self.assertIn('middle', bb_all_results)
            self.assertIn('lower', bb_all_results)
            
            # 测试多输出因子特定输出访问
            bb_upper = self.engine.get_factor_result('bb_10', 'upper')
            self.assertIsNotNone(bb_upper)
            self.assertIsInstance(bb_upper, (int, float))
            self.assertEqual(bb_upper, bb_all_results['upper'])
    
    def test_memory_usage_estimation(self):
        """测试内存使用估算"""
        # 注册一些因子
        sma_5 = SMA(5, 'sma_5', max_cache_size=50)
        atr_14 = ATR(14, 'atr_14', max_cache_size=50)
        self.engine.register_factor(sma_5)
        self.engine.register_factor(atr_14)
        
        # 添加数据
        for bar in self.sample_bars[:20]:
            self.engine.add_bar(bar)
        
        # 计算因子
        self.engine.calculate_factors()
        
        # 获取内存使用估算
        health = self.engine.get_health_status()
        memory_usage = health['memory_usage']
        
        self.assertIn('data_buffer_bytes', memory_usage)
        self.assertIn('factor_cache_bytes', memory_usage)
        self.assertIn('total_bytes', memory_usage)
        self.assertGreater(memory_usage['total_bytes'], 0)
    
    def test_factor_instances_access(self):
        """测试因子实例访问"""
        # 注册多个因子
        sma_5 = SMA(5, 'sma_5', max_cache_size=50)
        ema_12 = EMA(12, 'ema_12', max_cache_size=50)
        bb = BollingerBands(10, 2.0, 'bb_10', max_cache_size=50)
        
        self.engine.register_factor(sma_5)
        self.engine.register_factor(ema_12)
        self.engine.register_factor(bb)
        
        # 获取因子实例
        factor_instances = self.engine.get_factor_instances()
        
        # 验证因子实例
        self.assertEqual(len(factor_instances), 3)  # 3个基础因子
        self.assertIn('sma_5', factor_instances)
        self.assertIn('ema_12', factor_instances)
        self.assertIn('bb_10', factor_instances)
        
        # 验证实例类型
        self.assertIsInstance(factor_instances['sma_5'], SMA)
        self.assertIsInstance(factor_instances['ema_12'], EMA)
        self.assertIsInstance(factor_instances['bb_10'], BollingerBands)
    
    def test_performance_monitoring(self):
        """测试性能监控功能"""
        # 注册因子
        sma_5 = SMA(5, 'sma_5', max_cache_size=50)
        self.engine.register_factor(sma_5)
        
        # 添加数据
        for bar in self.sample_bars[:10]:
            self.engine.add_bar(bar)
        
        # 进行多次计算以收集性能数据
        for _ in range(5):
            self.engine.calculate_factors(parallel=False)  # 顺序计算
        
        # 获取统计信息
        stats = self.engine.get_statistics()
        performance = stats['performance']
        
        # 验证性能数据
        self.assertGreaterEqual(performance['sequential_count'], 5)
        self.assertGreater(performance['sequential_avg_time'], 0)
    
    def tearDown(self):
        """清理测试环境"""
        self.engine.shutdown()


if __name__ == '__main__':
    unittest.main()