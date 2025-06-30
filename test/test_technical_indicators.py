import unittest
import numpy as np
from datetime import datetime, timedelta

from quantbot.core.data import BarData, DataBuffer
from quantbot.core.factor import FactorEngine
from quantbot.factors.technical import SMA, EMA, RSI, MACD, MACDSignal, MACDHistogram, BollingerBands, ATR, STOCH, OBV


class TestTechnicalIndicators(unittest.TestCase):
    """测试所有技术指标的计算正确性"""
    
    def setUp(self):
        """设置测试环境"""
        self.data_buffer = DataBuffer(max_size=200)
        self.factor_engine = FactorEngine(self.data_buffer)
        
        # 创建测试数据（使用固定的价格序列便于验证）
        self.test_bars = self._create_test_data()
    
    def _create_test_data(self):
        """创建测试数据"""
        bars = []
        base_time = datetime(2024, 1, 1, 9, 0, 0)
        
        # 使用固定的价格序列进行测试
        prices = [
            100.0, 101.0, 102.0, 103.0, 104.0,  # 上涨趋势
            103.5, 103.0, 102.5, 102.0, 101.5,  # 下跌趋势
            101.0, 100.5, 100.0, 99.5, 99.0,    # 继续下跌
            99.5, 100.0, 100.5, 101.0, 101.5,   # 反弹
            102.0, 102.5, 103.0, 103.5, 104.0,  # 上涨
            103.8, 103.6, 103.4, 103.2, 103.0,  # 小幅下跌
            103.2, 103.4, 103.6, 103.8, 104.0,  # 小幅上涨
            104.2, 104.4, 104.6, 104.8, 105.0   # 继续上涨
        ]
        
        for i, price in enumerate(prices):
            high = price + np.random.uniform(0.1, 0.5)
            low = price - np.random.uniform(0.1, 0.5)
            open_price = price + np.random.uniform(-0.2, 0.2)
            volume = 1000 + i * 10
            
            bar = BarData(
                timestamp=base_time + timedelta(minutes=i),
                open_price=open_price,
                high=high,
                low=low,
                close=price,
                volume=volume,
                amount=price * volume
            )
            bars.append(bar)
        
        return bars
    
    def test_sma_calculation(self):
        """测试SMA计算"""
        sma_5 = SMA(5, 'sma_5', max_cache_size=100)
        self.factor_engine.register_factor(sma_5)
        
        # 逐个添加数据并计算因子
        sma_value = None
        for i, bar in enumerate(self.test_bars):
            self.data_buffer.add_bar(bar)
            results = self.factor_engine.calculate_all_factors()
            sma_value = results.get('sma_5')
            
            # SMA需要至少5个数据点
            if i >= 4:
                self.assertIsNotNone(sma_value)
                self.assertFalse(np.isnan(sma_value))
        
        # 验证最终结果
        self.assertIsNotNone(sma_value)
        
        # 手动验证最后5个数据的平均值
        last_5_closes = self.data_buffer.get_field('close', 5)
        expected_sma = np.mean(last_5_closes)
        self.assertAlmostEqual(sma_value, expected_sma, places=4)
    
    def test_ema_calculation(self):
        """测试EMA计算"""
        ema_12 = EMA(12, 'ema_12', max_cache_size=100)
        self.factor_engine.register_factor(ema_12)
        
        # 逐步添加数据并计算EMA
        for i in range(12, len(self.test_bars)):
            # 重新创建因子引擎以模拟逐步计算
            temp_buffer = DataBuffer(max_size=200)
            temp_engine = FactorEngine(temp_buffer)
            temp_ema = EMA(12, 'ema_12_temp', max_cache_size=100)
            temp_engine.register_factor(temp_ema)
            
            # 添加前i+1个数据
            for j in range(i + 1):
                temp_buffer.add_bar(self.test_bars[j])
            
            # 计算EMA
            temp_results = temp_engine.calculate_all_factors()
            ema_value = temp_results.get('ema_12_temp')
            
            if ema_value is not None:
                self.assertFalse(np.isnan(ema_value))
                self.assertGreater(ema_value, 0)
    
    def test_rsi_calculation(self):
        """测试RSI计算"""
        rsi_14 = RSI(14, 'rsi_14', max_cache_size=100)
        self.factor_engine.register_factor(rsi_14)
        
        results = self.factor_engine.calculate_all_factors()
        rsi_value = results.get('rsi_14')
        
        self.assertIsNotNone(rsi_value)
        self.assertFalse(np.isnan(rsi_value))
        # RSI应该在0-100之间
        self.assertGreaterEqual(rsi_value, 0)
        self.assertLessEqual(rsi_value, 100)
    
    def test_bollinger_bands_calculation(self):
        """测试布林带计算"""
        bb = BollingerBands(20, 2.0, 'bb_20', max_cache_size=100)
        self.factor_engine.register_factor(bb)
        
        results = self.factor_engine.calculate_all_factors()
        
        # 验证布林带的三个输出
        bb_upper = results.get('bb_20_upper')
        bb_middle = results.get('bb_20_middle')
        bb_lower = results.get('bb_20_lower')
        
        self.assertIsNotNone(bb_upper)
        self.assertIsNotNone(bb_middle)
        self.assertIsNotNone(bb_lower)
        
        # 验证布林带关系：上轨 > 中轨 > 下轨
        self.assertGreater(bb_upper, bb_middle)
        self.assertGreater(bb_middle, bb_lower)
        
        # 验证中轨等于SMA
        sma_data = self.data_buffer.get_field('close', 20)
        expected_middle = np.mean(sma_data)
        self.assertAlmostEqual(bb_middle, expected_middle, places=4)
    
    def test_atr_calculation(self):
        """测试ATR计算"""
        atr_14 = ATR(14, 'atr_14', max_cache_size=100)
        self.factor_engine.register_factor(atr_14)
        
        results = self.factor_engine.calculate_all_factors()
        atr_value = results.get('atr_14')
        
        self.assertIsNotNone(atr_value)
        self.assertFalse(np.isnan(atr_value))
        # ATR应该大于0
        self.assertGreater(atr_value, 0)
    
    def test_stoch_calculation(self):
        """测试随机指标计算"""
        stoch = STOCH(14, 3, 3, 'stoch_14', max_cache_size=100)
        self.factor_engine.register_factor(stoch)
        
        results = self.factor_engine.calculate_all_factors()
        
        stoch_k = results.get('stoch_14_k')
        stoch_d = results.get('stoch_14_d')
        
        if stoch_k is not None:
            self.assertFalse(np.isnan(stoch_k))
            # %K应该在0-100之间
            self.assertGreaterEqual(stoch_k, 0)
            self.assertLessEqual(stoch_k, 100)
        
        if stoch_d is not None:
            self.assertFalse(np.isnan(stoch_d))
            # %D应该在0-100之间
            self.assertGreaterEqual(stoch_d, 0)
            self.assertLessEqual(stoch_d, 100)
    
    def test_obv_calculation(self):
        """测试OBV计算"""
        obv = OBV('obv', max_cache_size=100)
        self.factor_engine.register_factor(obv)
        
        results = self.factor_engine.calculate_all_factors()
        obv_value = results.get('obv')
        
        self.assertIsNotNone(obv_value)
        self.assertFalse(np.isnan(obv_value))
    
    def test_macd_series(self):
        """测试MACD系列指标"""
        # 需要先注册EMA因子
        ema_12 = EMA(12, 'ema_12_close', max_cache_size=100)
        ema_26 = EMA(26, 'ema_26_close', max_cache_size=100)
        macd = MACD(12, 26, 9, 'macd_12_26_9', max_cache_size=100)
        macd_signal = MACDSignal(12, 26, 9, 'macd_signal_12_26_9', max_cache_size=100)
        macd_histogram = MACDHistogram(12, 26, 9, 'macd_histogram_12_26_9', max_cache_size=100)
        
        self.factor_engine.register_factor(ema_12)
        self.factor_engine.register_factor(ema_26)
        self.factor_engine.register_factor(macd)
        self.factor_engine.register_factor(macd_signal)
        self.factor_engine.register_factor(macd_histogram)
        
        results = self.factor_engine.calculate_all_factors()
        
        macd_value = results.get('macd_12_26_9')
        signal_value = results.get('macd_signal_12_26_9')
        hist_value = results.get('macd_histogram_12_26_9')
        
        if all(v is not None and not np.isnan(v) for v in [macd_value, signal_value, hist_value]):
            # 验证MACD关系：柱状图 = MACD线 - 信号线
            expected_hist = macd_value - signal_value
            self.assertAlmostEqual(hist_value, expected_hist, places=6)
    
    def test_input_validation(self):
        """测试输入验证"""
        # 测试无效的period
        with self.assertRaises(ValueError):
            SMA(0, 'invalid_sma', max_cache_size=100)
        
        with self.assertRaises(ValueError):
            SMA(-1, 'invalid_sma', max_cache_size=100)
        
        # 测试无效的标准差倍数
        with self.assertRaises(ValueError):
            BollingerBands(20, -1.0, 'invalid_bb', max_cache_size=100)
        
        # 测试无效的STOCH参数
        with self.assertRaises(ValueError):
            STOCH(0, 3, 3, 'invalid_stoch', max_cache_size=100)
    
    def test_factor_caching(self):
        """测试因子缓存功能"""
        sma_5 = SMA(5, 'sma_5_cache', max_cache_size=10)  # 小缓存测试
        
        # 手动添加结果到缓存
        for i in range(15):  # 超过缓存大小
            sma_5.add_result(100.0 + i)
        
        # 验证缓存大小
        self.assertEqual(sma_5.get_factor_size(), 10)  # 应该等于max_cache_size
        
        # 验证历史总数
        self.assertEqual(sma_5.get_history_total_factors(), 15)
        
        # 验证最新结果
        latest = sma_5.get_result()
        self.assertEqual(latest, 114.0)  # 100 + 14
        
        # 验证历史数据获取
        history = sma_5.get_factor(5)
        self.assertEqual(len(history), 5)
        expected = [110.0, 111.0, 112.0, 113.0, 114.0]  # 最近5个值
        np.testing.assert_array_equal(history, expected)
    
    def test_multi_output_factor_validation(self):
        """测试多输出因子的输入验证"""
        bb = BollingerBands(20, 2.0, 'bb_validation', max_cache_size=100)
        
        # 测试正确的字典输入
        valid_result = {'upper': 105.0, 'middle': 100.0, 'lower': 95.0}
        bb.add_result(valid_result)
        
        # 测试缺少键的字典输入
        with self.assertRaises(ValueError):
            bb.add_result({'upper': 105.0, 'middle': 100.0})  # 缺少'lower'
        
        # 测试多余键的字典输入
        with self.assertRaises(ValueError):
            bb.add_result({'upper': 105.0, 'middle': 100.0, 'lower': 95.0, 'extra': 50.0})
        
        # 测试单值输入（应该失败，因为是多输出因子）
        with self.assertRaises(ValueError):
            bb.add_result(100.0)
    
    def test_single_output_factor_validation(self):
        """测试单输出因子的输入验证"""
        sma_5 = SMA(5, 'sma_validation', max_cache_size=100)
        
        # 测试正确的单值输入
        sma_5.add_result(100.0)
        
        # 测试正确的字典输入
        sma_5.add_result({'default': 101.0})
        
        # 测试错误的字典输入（多个键）
        with self.assertRaises(ValueError):
            sma_5.add_result({'default': 100.0, 'extra': 50.0})
        
        # 测试错误的字典输入（错误的键名）
        with self.assertRaises(ValueError):
            sma_5.add_result({'wrong_key': 100.0})


if __name__ == '__main__':
    unittest.main()