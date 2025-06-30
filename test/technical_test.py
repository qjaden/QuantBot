import unittest
import numpy as np
import talib
from quantbot.core.factor import FactorEngine
from quantbot.core.data import BarData, DataBuffer
from quantbot.factors.technical import (
    SMA, EMA, MACD, MACDSignal, MACDHistogram, RSI, 
    BollingerBands
)

from datetime import datetime, timedelta


class TestTechnicalIndicators(unittest.TestCase):
    """测试技术指标因子"""
    
    def setUp(self):
        self.buffer_max_size = 150
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
        
        # 创建更复杂的价格走势，包含上升、下降和震荡
        for i in range(count):
            # 创建复杂的价格模式
            if i < 30:
                # 前30个数据点上升
                base_price += 0.5 + 0.3 * np.sin(i * 0.2)
            elif i < 60:
                # 中间30个数据点下降
                base_price -= 0.3 + 0.2 * np.cos(i * 0.1)  
            else:
                # 后面震荡
                base_price += 0.2 * np.sin(i * 0.5) + 0.1 * np.random.randn()
            
            # 确保价格为正
            base_price = max(base_price, 50.0)
            
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
    
    def _assert_nan_equal(self, actual, expected, msg_prefix="", places=6):
        """
        比较两个可能包含NaN的值
        
        Args:
            actual: 实际值
            expected: 期望值
            msg_prefix: 错误消息前缀
            places: 浮点数比较精度
        """
        if np.isnan(actual) and np.isnan(expected):
            # 两个都是 NaN，认为相等
            return
        elif np.isnan(actual) or np.isnan(expected):
            # 只有一个是 NaN，不相等
            self.fail(f"{msg_prefix}: 一个是 NaN 另一个不是。actual: {actual}, expected: {expected}")
        else:
            # 两个都不是 NaN，使用近似比较
            self.assertAlmostEqual(actual, expected, places=places, msg=f"{msg_prefix}: 数值不匹配")

    def test_sma_factor(self):
        """测试SMA因子"""
        # 清空数据缓存
        self.data_buffer.clear()
        
        # 注册SMA因子
        sma_5 = SMA(5, 'sma_5_close')
        sma_20 = SMA(20, 'sma_20_close')
        self.factor_engine.register_factor(sma_5)
        self.factor_engine.register_factor(sma_20)

        close = np.array([bar.close for bar in self.sample_bars])
        
        sma_5_talib = talib.SMA(close, timeperiod=5)
        sma_20_talib = talib.SMA(close, timeperiod=20)

        # 添加数据并计算因子
        for i, bar in enumerate(self.sample_bars):
            self.data_buffer.add_bar(bar)
            # 测试历史Bar数
            self.assertEqual(self.data_buffer.get_bar_buffer().get_history_total_bars(), i + 1)
            result = self.factor_engine.calculate_all_factors()

            self._assert_nan_equal(result["sma_5_close"], sma_5_talib[i], msg_prefix=f"SMA 5 at {i}")
            self._assert_nan_equal(result["sma_20_close"], sma_20_talib[i], msg_prefix=f"SMA 20 at {i}")
        
        bar_size = self.data_buffer.get_history_total_bars()
        self.assertEqual(bar_size, self.data_buffer.get_history_total_factors('sma_5_close'))
        self.assertEqual(bar_size, self.data_buffer.get_history_total_factors('sma_20_close'))
    
    def test_ema_factor(self):
        """测试EMA因子"""
        # 清空数据缓存
        self.data_buffer.clear()
        
        # 注册EMA因子
        ema_12 = EMA(12, 'ema_12_close')
        ema_26 = EMA(26, 'ema_26_close')
        self.factor_engine.register_factor(ema_12)
        self.factor_engine.register_factor(ema_26)

        close = np.array([bar.close for bar in self.sample_bars])
        ema_12_talib = talib.EMA(close, timeperiod=12)
        ema_26_talib = talib.EMA(close, timeperiod=26)

        for i, bar in enumerate(self.sample_bars):
            self.data_buffer.add_bar(bar)
            result = self.factor_engine.calculate_all_factors()
            self._assert_nan_equal(result["ema_12_close"], ema_12_talib[i], msg_prefix=f"EMA 12 at {i}")
            self._assert_nan_equal(result["ema_26_close"], ema_26_talib[i], msg_prefix=f"EMA 26 at {i}")

        bar_size = self.data_buffer.get_history_total_bars()
        self.assertEqual(bar_size, self.data_buffer.get_history_total_factors('ema_12_close'))
        self.assertEqual(bar_size, self.data_buffer.get_history_total_factors('ema_26_close'))
    
    def test_macd_factor(self):
        """测试MACD因子"""
        # 清空数据缓存
        self.data_buffer.clear()
        
        # 首先注册依赖的EMA因子
        ema_12 = EMA(12, 'ema_12_close')
        ema_26 = EMA(26, 'ema_26_close')
        self.factor_engine.register_factor(ema_12)
        self.factor_engine.register_factor(ema_26)
        
        # 注册MACD因子
        macd = MACD(12, 26, 9, 'macd_12_26_9_close')
        self.factor_engine.register_factor(macd)

        close = np.array([bar.close for bar in self.sample_bars])
        macd_talib, _, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

        for i, bar in enumerate(self.sample_bars):
            self.data_buffer.add_bar(bar)
            result = self.factor_engine.calculate_all_factors()
            self._assert_nan_equal(result["macd_12_26_9_close"], macd_talib[i], msg_prefix=f"MACD at {i}")

        bar_size = self.data_buffer.get_history_total_bars()
        self.assertEqual(bar_size, self.data_buffer.get_history_total_factors('macd_12_26_9_close'))
    
    def test_macd_complete(self):
        """测试完整的MACD系统（MACD线、信号线、柱状图）"""
        # 清空数据缓存
        self.data_buffer.clear()
        
        # 注册所有必要的因子
        ema_12 = EMA(12, 'ema_12_close')
        ema_26 = EMA(26, 'ema_26_close')
        macd = MACD(12, 26, 9, 'macd_12_26_9_close')
        macd_signal = MACDSignal(12, 26, 9, 'macd_signal_12_26_9_close')
        macd_hist = MACDHistogram(12, 26, 9, 'macd_hist_12_26_9_close')
        
        self.factor_engine.register_factor(ema_12)
        self.factor_engine.register_factor(ema_26)
        self.factor_engine.register_factor(macd)
        self.factor_engine.register_factor(macd_signal)
        self.factor_engine.register_factor(macd_hist)

        close = np.array([bar.close for bar in self.sample_bars])
        macd_talib, signal_talib, hist_talib = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

        for i, bar in enumerate(self.sample_bars):
            self.data_buffer.add_bar(bar)
            result = self.factor_engine.calculate_all_factors()
            
            self._assert_nan_equal(result["macd_12_26_9_close"], macd_talib[i], msg_prefix=f"MACD Line at {i}")
            self._assert_nan_equal(result["macd_signal_12_26_9_close"], signal_talib[i], msg_prefix=f"MACD Signal at {i}")
            self._assert_nan_equal(result["macd_hist_12_26_9_close"], hist_talib[i], msg_prefix=f"MACD Histogram at {i}")

    def test_rsi_factor(self):
        """测试RSI因子"""
        # 清空数据缓存
        self.data_buffer.clear()
        
        # 注册RSI因子
        rsi_14 = RSI(14, 'rsi_14_close')
        self.factor_engine.register_factor(rsi_14)

        close = np.array([bar.close for bar in self.sample_bars])
        rsi_14_talib = talib.RSI(close, timeperiod=14)

        for i, bar in enumerate(self.sample_bars):
            self.data_buffer.add_bar(bar)
            result = self.factor_engine.calculate_all_factors()
            self._assert_nan_equal(result["rsi_14_close"], rsi_14_talib[i], msg_prefix=f"RSI 14 at {i}")

        bar_size = self.data_buffer.get_history_total_bars()
        self.assertEqual(bar_size, self.data_buffer.get_history_total_factors('rsi_14_close'))

    def test_bollinger_bands_factor(self):
        """测试布林带因子"""
        # 清空数据缓存
        self.data_buffer.clear()
        
        # 注册布林带因子（新的多输出接口）
        bb = BollingerBands(20, 2.0, 'bb_20_2')
        self.factor_engine.register_factor(bb)

        close = np.array([bar.close for bar in self.sample_bars])
        upper_talib, middle_talib, lower_talib = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

        for i, bar in enumerate(self.sample_bars):
            self.data_buffer.add_bar(bar)
            result = self.factor_engine.calculate_all_factors()
            
            self._assert_nan_equal(result["bb_20_2_middle"], middle_talib[i], msg_prefix=f"BB Middle at {i}")
            self._assert_nan_equal(result["bb_20_2_upper"], upper_talib[i], msg_prefix=f"BB Upper at {i}")
            self._assert_nan_equal(result["bb_20_2_lower"], lower_talib[i], msg_prefix=f"BB Lower at {i}")

        bar_size = self.data_buffer.get_history_total_bars()
        self.assertEqual(bar_size, bb.get_history_total_factors('middle'))
        self.assertEqual(bar_size, bb.get_history_total_factors('upper'))
        self.assertEqual(bar_size, bb.get_history_total_factors('lower'))

    def test_multiple_indicators(self):
        """测试多个指标同时运行"""
        # 清空数据缓存
        self.data_buffer.clear()
        
        # 注册多个指标
        sma_10 = SMA(10, 'sma_10_close')
        ema_10 = EMA(10, 'ema_10_close')
        rsi_14 = RSI(14, 'rsi_14_close')
        bb = BollingerBands(20, 2.0, 'bb_20_2')
        
        self.factor_engine.register_factor(sma_10)
        self.factor_engine.register_factor(ema_10)
        self.factor_engine.register_factor(rsi_14)
        self.factor_engine.register_factor(bb)

        close = np.array([bar.close for bar in self.sample_bars])
        sma_10_talib = talib.SMA(close, timeperiod=10)
        ema_10_talib = talib.EMA(close, timeperiod=10)
        rsi_14_talib = talib.RSI(close, timeperiod=14)
        _, bb_middle_talib, _ = talib.BBANDS(close, timeperiod=20)

        for i, bar in enumerate(self.sample_bars):
            self.data_buffer.add_bar(bar)
            result = self.factor_engine.calculate_all_factors()
            
            self._assert_nan_equal(result["sma_10_close"], sma_10_talib[i], msg_prefix=f"SMA 10 at {i}")
            self._assert_nan_equal(result["ema_10_close"], ema_10_talib[i], msg_prefix=f"EMA 10 at {i}")
            self._assert_nan_equal(result["rsi_14_close"], rsi_14_talib[i], msg_prefix=f"RSI 14 at {i}")
            self._assert_nan_equal(result["bb_20_2_middle"], bb_middle_talib[i], msg_prefix=f"BB Middle at {i}")

    def test_edge_cases(self):
        """测试边界情况"""
        # 清空数据缓存
        self.data_buffer.clear()
        
        # 测试短周期的指标
        sma_2 = SMA(2, 'sma_2_close')
        ema_3 = EMA(3, 'ema_3_close')
        
        self.factor_engine.register_factor(sma_2)
        self.factor_engine.register_factor(ema_3)

        # 只添加少量数据
        for i in range(5):
            self.data_buffer.add_bar(self.sample_bars[i])
            result = self.factor_engine.calculate_all_factors()
            
            # 前2个数据点SMA应该是NaN
            if i < 2:
                self.assertTrue(np.isnan(result["sma_2_close"]), f"SMA 2 should be NaN at index {i}")
            else:
                self.assertFalse(np.isnan(result["sma_2_close"]), f"SMA 2 should not be NaN at index {i}")
            
            # 前3个数据点EMA应该是NaN
            if i < 3:
                self.assertTrue(np.isnan(result["ema_3_close"]), f"EMA 3 should be NaN at index {i}")
            else:
                self.assertFalse(np.isnan(result["ema_3_close"]), f"EMA 3 should not be NaN at index {i}")

    def test_bollinger_bands_complex(self):
        """测试布林带复合因子"""
        # 清空数据缓存
        self.data_buffer.clear()
        
        # 注册布林带复合因子
        bb_complex = BollingerBandsComplex(20, 2.0, 'bb_complex_20_2')
        self.factor_engine.register_factor(bb_complex)

        close = np.array([bar.close for bar in self.sample_bars])
        upper_talib, middle_talib, lower_talib = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

        for i, bar in enumerate(self.sample_bars):
            self.data_buffer.add_bar(bar)
            result = self.factor_engine.calculate_all_factors()
            
            # 验证复合因子的三个输出
            self._assert_nan_equal(result["bb_complex_20_2_upper"], upper_talib[i], msg_prefix=f"BB Complex Upper at {i}")
            self._assert_nan_equal(result["bb_complex_20_2_middle"], middle_talib[i], msg_prefix=f"BB Complex Middle at {i}")
            self._assert_nan_equal(result["bb_complex_20_2_lower"], lower_talib[i], msg_prefix=f"BB Complex Lower at {i}")

        bar_size = self.data_buffer.get_history_total_bars()
        self.assertEqual(bar_size, self.data_buffer.get_history_total_factors('bb_complex_20_2_upper'))
        self.assertEqual(bar_size, self.data_buffer.get_history_total_factors('bb_complex_20_2_middle'))
        self.assertEqual(bar_size, self.data_buffer.get_history_total_factors('bb_complex_20_2_lower'))

    def test_complex_factor_caching(self):
        """测试复合因子的缓存机制"""
        # 清空数据缓存
        self.data_buffer.clear()
        
        # 注册布林带复合因子
        bb_complex = BollingerBandsComplex(20, 2.0, 'bb_cache_test_20_2')
        self.factor_engine.register_factor(bb_complex)

        # 添加足够的数据
        for i in range(25):
            self.data_buffer.add_bar(self.sample_bars[i])
        
        # 第一次计算
        result1 = self.factor_engine.calculate_all_factors()
        
        # 验证缓存是否生效（通过检查内部缓存字典）
        self.assertIn('bb_cache_test_20_2', bb_complex._result_cache)
        cached_results = bb_complex._result_cache['bb_cache_test_20_2']
        
        # 验证缓存的结果
        self.assertIn('bb_cache_test_20_2_upper', cached_results)
        self.assertIn('bb_cache_test_20_2_middle', cached_results)
        self.assertIn('bb_cache_test_20_2_lower', cached_results)
        
        # 验证缓存的值与计算结果一致
        self.assertEqual(cached_results['bb_cache_test_20_2_upper'], result1['bb_cache_test_20_2_upper'])
        self.assertEqual(cached_results['bb_cache_test_20_2_middle'], result1['bb_cache_test_20_2_middle'])
        self.assertEqual(cached_results['bb_cache_test_20_2_lower'], result1['bb_cache_test_20_2_lower'])


if __name__ == '__main__':
    unittest.main()
    