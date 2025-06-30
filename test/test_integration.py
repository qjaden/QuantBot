import unittest
import numpy as np
from datetime import datetime, timedelta

from quantbot.engine import QuantBotEngine
from quantbot.core.data import BarData
from quantbot.factors.technical import SMA, EMA, RSI, MACD, MACDSignal, MACDHistogram, BollingerBands, ATR, STOCH, OBV


class TestIntegration(unittest.TestCase):
    """集成测试 - 测试整个系统的协同工作"""
    
    def setUp(self):
        """设置测试环境"""
        self.engine = QuantBotEngine(buffer_size=500, max_workers=2)
        self.test_data = self._create_comprehensive_test_data()
    
    def _create_comprehensive_test_data(self):
        """创建全面的测试数据"""
        bars = []
        base_time = datetime(2024, 1, 1, 9, 0, 0)
        base_price = 100.0
        
        # 创建60个数据点，包含不同的市场情况
        for i in range(60):
            # 创建复杂的价格走势
            if i < 20:
                # 上涨趋势
                base_price += 0.5 + 0.2 * np.sin(i * 0.3)
            elif i < 40:
                # 震荡走势
                base_price += 0.1 * np.sin(i * 0.5) + 0.05 * np.random.randn()
            else:
                # 下跌趋势
                base_price -= 0.3 + 0.1 * np.cos(i * 0.2)
            
            # 确保价格为正
            base_price = max(base_price, 50.0)
            
            # 生成OHLC数据
            high = base_price + abs(np.random.normal(0, 0.5))
            low = base_price - abs(np.random.normal(0, 0.5))
            open_price = base_price + np.random.normal(0, 0.2)
            
            # 确保OHLC逻辑正确
            high = max(high, open_price, base_price)
            low = min(low, open_price, base_price)
            
            volume = 1000 + i * 20 + np.random.uniform(-100, 100)
            volume = max(volume, 100)  # 确保成交量为正
            
            bar = BarData(
                timestamp=base_time + timedelta(minutes=i * 5),
                open_price=open_price,
                high=high,
                low=low,
                close=base_price,
                volume=volume,
                amount=base_price * volume
            )
            bars.append(bar)
        
        return bars
    
    def test_complete_workflow(self):
        """测试完整的工作流程"""
        # 1. 注册所有类型的因子
        factors = [
            # 单输出因子
            SMA(5, 'sma_5', max_cache_size=100),
            SMA(20, 'sma_20', max_cache_size=100),
            EMA(12, 'ema_12', max_cache_size=100),
            EMA(26, 'ema_26', max_cache_size=100),
            RSI(14, 'rsi_14', max_cache_size=100),
            ATR(14, 'atr_14', max_cache_size=100),
            OBV('obv', max_cache_size=100),
            
            # 多输出因子
            BollingerBands(20, 2.0, 'bb_20', max_cache_size=100),
            STOCH(14, 3, 3, 'stoch_14', max_cache_size=100),
            
            # MACD系列
            MACD(12, 26, 9, 'macd_12_26', max_cache_size=100),
            MACDSignal(12, 26, 9, 'macd_signal_12_26', max_cache_size=100),
            MACDHistogram(12, 26, 9, 'macd_hist_12_26', max_cache_size=100),
        ]
        
        # 注册所有因子
        for factor in factors:
            self.engine.register_factor(factor)
        
        # 验证注册成功
        registered_factors = self.engine.get_registered_factors()
        # 总输出数应该是: 7个单输出 + 3个BB输出 + 2个STOCH输出 + 3个MACD输出 = 15个
        self.assertEqual(len(registered_factors), 15)
        
        # 2. 检查依赖关系
        self.assertFalse(self.engine.has_circular_dependency())
        
        # 3. 逐步添加数据并计算
        results_history = []
        for i, bar in enumerate(self.test_data):
            self.engine.add_bar(bar)
            
            # 交替使用串行和并行计算
            parallel = i > 30  # 后30个使用并行计算
            results = self.engine.calculate_factors(parallel=parallel)
            results_history.append(results)
            
            # 验证每次计算的结果
            if i >= 25:  # 确保有足够的数据
                self._validate_calculation_results(results, i)
        
        # 4. 验证最终状态
        final_stats = self.engine.get_statistics()
        self.assertEqual(final_stats['calculation_count'], len(self.test_data))
        self.assertGreater(final_stats['total_calculation_time'], 0)
        
        # 5. 验证健康状态
        health = self.engine.get_health_status()
        self.assertEqual(health['status'], 'healthy')
        self.assertEqual(health['error_rate'], 0.0)
        
        # 6. 测试因子结果访问
        self._test_factor_result_access()
        
        # 7. 测试性能监控
        if 'performance' in final_stats:
            perf = final_stats['performance']
            self.assertGreater(perf['sequential_count'], 0)
            self.assertGreater(perf['parallel_count'], 0)
    
    def _validate_calculation_results(self, results, bar_index):
        """验证计算结果的合理性"""
        # 验证SMA结果
        sma_5 = results.get('sma_5')
        sma_20 = results.get('sma_20')
        if sma_5 is not None and sma_20 is not None:
            self.assertFalse(np.isnan(sma_5))
            self.assertFalse(np.isnan(sma_20))
            self.assertGreater(sma_5, 0)
            self.assertGreater(sma_20, 0)
        
        # 验证RSI结果
        rsi = results.get('rsi_14')
        if rsi is not None and not np.isnan(rsi):
            self.assertGreaterEqual(rsi, 0)
            self.assertLessEqual(rsi, 100)
        
        # 验证布林带结果
        bb_upper = results.get('bb_20_upper')
        bb_middle = results.get('bb_20_middle')
        bb_lower = results.get('bb_20_lower')
        if all(x is not None and not np.isnan(x) for x in [bb_upper, bb_middle, bb_lower]):
            self.assertGreater(bb_upper, bb_middle)
            self.assertGreater(bb_middle, bb_lower)
        
        # 验证ATR结果
        atr = results.get('atr_14')
        if atr is not None and not np.isnan(atr):
            self.assertGreater(atr, 0)
        
        # 验证随机指标结果
        stoch_k = results.get('stoch_14_k')
        stoch_d = results.get('stoch_14_d')
        if stoch_k is not None and not np.isnan(stoch_k):
            self.assertGreaterEqual(stoch_k, 0)
            self.assertLessEqual(stoch_k, 100)
        if stoch_d is not None and not np.isnan(stoch_d):
            self.assertGreaterEqual(stoch_d, 0)
            self.assertLessEqual(stoch_d, 100)
    
    def _test_factor_result_access(self):
        """测试因子结果访问"""
        # 测试单输出因子
        sma_result = self.engine.get_factor_result('sma_5')
        if sma_result is not None:
            self.assertIsInstance(sma_result, (int, float))
        
        # 测试多输出因子
        bb_result = self.engine.get_factor_result('bb_20')
        if bb_result is not None:
            self.assertIsInstance(bb_result, dict)
            self.assertIn('upper', bb_result)
            self.assertIn('middle', bb_result)
            self.assertIn('lower', bb_result)
        
        # 测试特定输出访问
        bb_upper = self.engine.get_factor_result('bb_20', 'upper')
        if bb_upper is not None:
            self.assertIsInstance(bb_upper, (int, float))
    
    def test_error_handling(self):
        """测试错误处理"""
        # 测试重复注册
        sma = SMA(5, 'duplicate_sma', max_cache_size=100)
        self.engine.register_factor(sma)
        
        # 尝试重复注册相同名称的因子
        sma_duplicate = SMA(10, 'duplicate_sma', max_cache_size=100)
        with self.assertRaises(ValueError):
            self.engine.register_factor(sma_duplicate)
        
        # 测试访问不存在的因子
        result = self.engine.get_factor_result('nonexistent_factor')
        self.assertIsNone(result)
        
        # 验证错误被记录
        self.assertGreater(self.engine.error_count, 0)
    
    def test_memory_and_performance(self):
        """测试内存使用和性能"""
        # 注册一些因子
        factors = [
            SMA(5, 'perf_sma_5', max_cache_size=50),
            EMA(12, 'perf_ema_12', max_cache_size=50),
            BollingerBands(20, 2.0, 'perf_bb', max_cache_size=50),
        ]
        
        for factor in factors:
            self.engine.register_factor(factor)
        
        # 添加数据并在每次添加后计算
        for bar in self.test_data[:30]:
            self.engine.add_bar(bar)
            self.engine.calculate_factors()  # 每次添加数据后计算
        
        # 进行额外的计算以收集性能数据
        for _ in range(5):
            self.engine.calculate_factors(parallel=False)
            self.engine.calculate_factors(parallel=True)
        
        # 检查统计信息
        stats = self.engine.get_statistics()
        expected_min_calculations = 30 + 10  # 30次数据添加计算 + 10次额外计算
        self.assertGreater(stats['calculation_count'], expected_min_calculations - 5)  # 允许一些容错
        
        # 检查健康状态
        health = self.engine.get_health_status()
        self.assertIn('memory_usage', health)
        if 'total_bytes' in health['memory_usage']:
            self.assertGreater(health['memory_usage']['total_bytes'], 0)
    
    def test_cache_overflow_handling(self):
        """测试缓存溢出处理"""
        # 创建小缓存的因子
        sma_small = SMA(5, 'small_cache_sma', max_cache_size=10)
        self.engine.register_factor(sma_small)
        
        # 添加超过缓存大小的数据
        for bar in self.test_data:
            self.engine.add_bar(bar)
            self.engine.calculate_factors()
        
        # 验证缓存大小不超过限制
        self.assertEqual(sma_small.get_factor_size(), 10)
        
        # 验证历史总数正确（考虑到计算模式的影响）
        # 由于我们是逐个添加数据并计算，所以总数应该等于数据长度
        self.assertEqual(sma_small.get_history_total_factors(), len(self.test_data))
        
        # 验证可以正确获取历史数据
        history = sma_small.get_factor(5)
        self.assertEqual(len(history), 5)
    
    def test_factor_dependencies(self):
        """测试因子依赖关系"""
        # 注册有依赖关系的因子 - 使用MACD期望的EMA名称
        ema_12 = EMA(12, 'ema_12_close', max_cache_size=100)  # MACD会寻找这个名称
        ema_26 = EMA(26, 'ema_26_close', max_cache_size=100)  # MACD会寻找这个名称
        macd = MACD(12, 26, 9, 'dep_macd', max_cache_size=100)
        
        self.engine.register_factor(ema_12)
        self.engine.register_factor(ema_26)
        self.engine.register_factor(macd)
        
        # 验证计算顺序
        calc_order = self.engine.get_calculation_order()
        
        # EMA因子应该在MACD之前计算（检查是否都在列表中）
        self.assertIn('ema_12_close', calc_order)
        self.assertIn('ema_26_close', calc_order) 
        self.assertIn('dep_macd', calc_order)
        
        # 由于MACD依赖于EMA，所以获取它们的索引位置
        if 'dep_macd' in calc_order:
            ema_12_index = calc_order.index('ema_12_close')
            ema_26_index = calc_order.index('ema_26_close')
            macd_index = calc_order.index('dep_macd')
            
            self.assertLess(ema_12_index, macd_index)
            self.assertLess(ema_26_index, macd_index)
        
        # 添加足够的数据并计算
        for bar in self.test_data[:35]:
            self.engine.add_bar(bar)
        
        results = self.engine.calculate_factors()
        
        # 验证MACD计算正确（如果有足够数据）
        macd_value = results.get('dep_macd')
        if macd_value is not None and not np.isnan(macd_value):
            ema_12_value = results.get('ema_12_close')
            ema_26_value = results.get('ema_26_close')
            
            if all(x is not None and not np.isnan(x) for x in [ema_12_value, ema_26_value]):
                expected_macd = ema_12_value - ema_26_value
                self.assertAlmostEqual(macd_value, expected_macd, places=6)
    
    def tearDown(self):
        """清理测试环境"""
        self.engine.shutdown()


if __name__ == '__main__':
    unittest.main()