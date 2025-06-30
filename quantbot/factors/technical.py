"""
技术指标因子实现

包含常用的技术分析指标：
- SMA: 简单移动平均线
- EMA: 指数移动平均线
- RSI: 相对强弱指标
- MACD: 移动平均收敛发散指标(包含macd/signal/histogram三个输出)
- BollingerBands: 布林带
- STOCH: 随机指标(KD)
- ATR: 平均真实波幅
- OBV: 能量潮指标

所有指标都按照TA-Lib的计算方式实现，确保精度和一致性。
"""

import numpy as np
from talib import stream as ta_stream
from typing import List, Optional, Dict
from ..core.factor import Factor
from ..core.data import DataBuffer


class SMA(Factor):
    """
    简单移动平均线 (Simple Moving Average)
    按照talib相同的计算方式实现
    """

    def __init__(self, period: int, name: str, source_field: str='close', dependencies: Optional[List[str]] = None):
        """
        初始化SMA因子
        
        Args:
            period: 移动平均周期，必须大于0
            name: 指标名称（必填）
            source_field: 数据源字段，默认为'close'
            dependencies: 依赖列表（可选，如不指定则使用source_field）
            
        Raises:
            ValueError: 当period小于等于0时
        """
        if not isinstance(period, int) or period <= 0:
            raise ValueError(f"SMA period must be a positive integer, got {period} (type: {type(period)})")
            
        self.period = period
        self.source_field = source_field

        if dependencies is None:
            dependencies = [source_field]

        # SMA是单输出因子，使用['default']
        super().__init__(name, dependencies, output_names=['default'])

    def calculate(self, data_buffer: DataBuffer) -> float:
        """
        计算SMA值，采用TA-Lib相同的方式：
        1. 当数据不足period时返回NaN
        2. 计算最近period个数据的算术平均值
        
        Args:
            data_buffer: 统一数据缓存对象
            
        Returns:
            float: SMA值，数据不足时返回NaN
        """
        bar_size = data_buffer.get_bar_size()
        
        if bar_size < self.period:
            return np.nan
        
        # 获取数据
        try:
            data = data_buffer.get_bar_field(self.source_field, self.period)
            if len(data) < self.period:
                return np.nan
            
            # 检查数据有效性
            if np.any(np.isnan(data)):
                return np.nan
            
            return float(np.mean(data))
        except Exception:
            return np.nan
    
    def __repr__(self) -> str:
        return f"SMA(period={self.period}, name='{self.name}', source_field='{self.source_field}')"


class EMA(Factor):
    """
    指数移动平均线 (Exponential Moving Average)
    按照talib相同的计算方式实现
    """
    
    def __init__(self, period: int, name: str, source_field: str = 'close', dependencies: Optional[List[str]] = None):
        """
        初始化EMA因子
        
        Args:
            period: 移动平均周期
            name: 指标名称（必填）
            source_field: 数据源字段
            dependencies: 依赖列表（可选，如不指定则使用source_field）
        """
        # 输入验证
        if not isinstance(period, int) or period <= 0:
            raise ValueError(f"EMA period must be a positive integer, got {period} (type: {type(period)})")
            
        self.period = period
        self.source_field = source_field
        self.alpha = 2.0 / (period + 1)

        if dependencies is None:
            dependencies = [source_field]
            
        # EMA是单输出因子，使用['default']
        super().__init__(name, dependencies, output_names=['default'])

    def calculate(self, data_buffer: DataBuffer) -> float:
        """
        计算EMA值，采用TA-Lib相同的方式：
        1. 当数据不足period时返回NaN
        2. 当数据刚好等于period时，使用SMA作为初始EMA值
        3. 之后使用标准EMA公式：EMA = alpha * current + (1 - alpha) * previous_EMA
        
        Args:
            data_buffer: 统一数据缓存对象
            
        Returns:
            float: EMA值，数据不足时返回NaN
        """
        bar_size = data_buffer.get_bar_size()
        
        if bar_size < self.period:
            return np.nan
        
        try:
            # 获取当前价格
            current_data = data_buffer.get_bar_field(self.source_field, 1)
            if len(current_data) == 0 or np.isnan(current_data[0]):
                return np.nan
                
            current_price = current_data[0]
            
            if bar_size == self.period:
                # 第一次计算EMA，使用SMA作为初始值
                data = data_buffer.get_bar_field(self.source_field, self.period)
                if len(data) < self.period or np.any(np.isnan(data)):
                    return np.nan
                return float(np.mean(data))
        
            # 获取前一个EMA值
            previous_ema_data = data_buffer.get_factor_data(self.name, 1, 'default')
            
            if len(previous_ema_data) == 0:
                # 如果没有历史EMA数据，使用SMA作为初始值
                data = data_buffer.get_bar_field(self.source_field, self.period)
                return float(np.mean(data[-self.period:]))
            
            previous_ema = previous_ema_data[-1]
            
            if np.isnan(previous_ema):
                # 如果前一个EMA值为NaN，使用SMA作为初始值
                data = data_buffer.get_bar_field(self.source_field, self.period)
                return float(np.mean(data[-self.period:]))
            
            # 计算EMA：alpha * current + (1 - alpha) * previous_EMA
            result = self.alpha * current_price + (1 - self.alpha) * previous_ema
            return float(result)
            
        except Exception:
            return np.nan
    
    def __repr__(self) -> str:
        return f"EMA(period={self.period}, name='{self.name}', source_field='{self.source_field}', alpha={self.alpha:.4f})"


class RSI(Factor):
    """
    相对强弱指标 (Relative Strength Index)
    按照talib相同的计算方式实现
    """
    
    def __init__(self, period: int, name: str, source_field: str = 'close', dependencies: Optional[List[str]] = None):
        """
        初始化RSI因子
        
        Args:
            period: 计算周期
            name: 指标名称（必填）
            source_field: 数据源字段
            dependencies: 依赖列表（可选，如不指定则使用source_field）
        """
        # 输入验证
        if not isinstance(period, int) or period <= 0:
            raise ValueError(f"RSI period must be a positive integer, got {period} (type: {type(period)})")
            
        self.period = period
        self.source_field = source_field
        self.alpha = 1.0 / period  # EMA的alpha值
        
        if dependencies is None:
            dependencies = [source_field]
            
        # RSI是单输出因子，使用['default']
        super().__init__(name, dependencies, output_names=['default'])
    
    def calculate(self, data_buffer: DataBuffer) -> float:
        """
        计算RSI值，采用talib相同的方式：
        1. 当数据不足period+1时返回NaN  
        2. 第一次计算使用SMA作为初始值
        3. 之后使用EMA方式计算平均收益和损失
        """
        bar_size = data_buffer.get_bar_size()
        
        if bar_size <= self.period:
            return np.nan
        
        # 获取数据（需要period+1个数据来计算period个价格变化）
        data = data_buffer.get_bar_field(self.source_field, self.period + 1)
        
        if len(data) <= self.period:
            return np.nan
        
        # 计算价格变化
        price_changes = np.diff(data)
        
        # 分离上涨和下跌
        gains = np.where(price_changes > 0, price_changes, 0.0)
        losses = np.where(price_changes < 0, -price_changes, 0.0)
        
        if bar_size == self.period + 1:
            # 第一次计算，使用SMA作为初始值
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
        else:
            # 获取前一个RSI计算时的平均收益和损失
            # 由于我们需要保存状态，这里简化处理：重新计算最近period个值的EMA
            if len(gains) >= self.period:
                # 计算EMA，第一个值使用SMA
                avg_gain = gains[0]
                avg_loss = losses[0]
                
                for i in range(1, len(gains)):
                    avg_gain = self.alpha * gains[i] + (1 - self.alpha) * avg_gain
                    avg_loss = self.alpha * losses[i] + (1 - self.alpha) * avg_loss
            else:
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
        
        # 避免除零
        if avg_loss == 0:
            return 100.0
        
        try:
            # 计算RSI
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
            
            return float(rsi)
        except Exception:
            return np.nan
    
    def __repr__(self) -> str:
        return f"RSI(period={self.period}, name='{self.name}', source_field='{self.source_field}')"


class MACD(Factor):
    """
    移动平均收敛发散指标 (Moving Average Convergence Divergence)
    一次计算返回MACD线、信号线、柱状图三个结果
    按照talib相同的计算方式实现
    """
    
    def __init__(self, fast_period: int, slow_period: int, signal_period: int, name: str, source_field: str = 'close', dependencies: Optional[List[str]] = None):
        """
        初始化MACD因子
        
        Args:
            fast_period: 快速EMA周期
            slow_period: 慢速EMA周期  
            signal_period: 信号线EMA周期
            name: 指标名称（必填）
            source_field: 数据源字段
            dependencies: 依赖列表（可选，如不指定则使用EMA依赖）
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.source_field = source_field
        self.signal_alpha = 2.0 / (signal_period + 1)

        if dependencies is None:
            # MACD依赖于快速和慢速EMA
            fast_ema_name = f"ema_{fast_period}_{source_field}"
            slow_ema_name = f"ema_{slow_period}_{source_field}"
            dependencies = [fast_ema_name, slow_ema_name]
        
        # 输出名称：MACD线、信号线、柱状图
        output_names = ['macd', 'signal', 'histogram']
        super().__init__(name, dependencies, output_names)

    def calculate(self, data_buffer: DataBuffer) -> Dict[str, float]:
        """
        计算MACD的所有值，采用talib相同的方式：
        MACD线 = 快速EMA - 慢速EMA
        信号线 = MACD线的EMA
        柱状图 = MACD线 - 信号线
        
        Returns:
            包含'macd', 'signal', 'histogram'的字典
        """
        bar_size = data_buffer.get_bar_size()
        
        # 检查是否有足够的数据
        if bar_size < self.slow_period:
            return {
                'macd': np.nan,
                'signal': np.nan,
                'histogram': np.nan
            }
        
        fast_ema_name = f"ema_{self.fast_period}_{self.source_field}"
        slow_ema_name = f"ema_{self.slow_period}_{self.source_field}"
        
        # 检查依赖的EMA是否存在且长度匹配
        fast_ema_size = data_buffer.get_factor_size(fast_ema_name)
        slow_ema_size = data_buffer.get_factor_size(slow_ema_name)
        
        if fast_ema_size == 0 or slow_ema_size == 0:
            return {
                'macd': np.nan,
                'signal': np.nan,
                'histogram': np.nan
            }
        
        fast_ema_data = data_buffer.get_factor_data(fast_ema_name, 1, 'default')
        slow_ema_data = data_buffer.get_factor_data(slow_ema_name, 1, 'default')

        if len(fast_ema_data) == 0 or len(slow_ema_data) == 0:
            return {
                'macd': np.nan,
                'signal': np.nan,
                'histogram': np.nan
            }

        fast_ema = fast_ema_data[-1]
        slow_ema = slow_ema_data[-1]
        
        if np.isnan(fast_ema) or np.isnan(slow_ema):
            return {
                'macd': np.nan,
                'signal': np.nan,
                'histogram': np.nan
            }
        
        # 计算MACD线 = 快速EMA - 慢速EMA
        macd_line = fast_ema - slow_ema
        
        # 计算信号线（MACD线的EMA）
        signal_size = data_buffer.get_factor_size(self.name)
        
        if signal_size == 0:
            # 第一次计算，信号线使用MACD值作为初始值
            signal_line = macd_line
        else:
            # 获取前一个信号线值
            previous_signal_data = data_buffer.get_factor_data(self.name, 1, 'signal')
            
            if len(previous_signal_data) == 0:
                # 第一次计算，使用MACD值作为初始值
                signal_line = macd_line
            else:
                previous_signal = previous_signal_data[-1]
                if np.isnan(previous_signal):
                    # 如果上一个信号线值为NaN，使用MACD值作为初始值
                    signal_line = macd_line
                else:
                    # 计算信号线（MACD的EMA）
                    signal_line = self.signal_alpha * macd_line + (1 - self.signal_alpha) * previous_signal
        
        # 计算柱状图 = MACD线 - 信号线
        histogram = macd_line - signal_line
        
        return {
            'macd': float(macd_line),
            'signal': float(signal_line),
            'histogram': float(histogram)
        }
    
    def __repr__(self) -> str:
        return f"MACD(fast_period={self.fast_period}, slow_period={self.slow_period}, signal_period={self.signal_period}, name='{self.name}', source_field='{self.source_field}')"


class BollingerBands(Factor):
    """
    布林带因子 (Bollinger Bands)
    一次计算返回上轨、中轨、下轨三个结果
    按照talib相同的计算方式实现
    """
    
    def __init__(self, period: int, std_dev: float, name: str, source_field: str = 'close', dependencies: Optional[List[str]] = None):
        """
        初始化布林带因子
        
        Args:
            period: 计算周期
            std_dev: 标准差倍数
            name: 因子名称前缀
            source_field: 数据源字段
            dependencies: 依赖列表（可选，如不指定则使用source_field）
            
        Raises:
            ValueError: 当period小于等于0或std_dev小于0时
        """
        if not isinstance(period, int) or period <= 0:
            raise ValueError(f"BollingerBands period must be a positive integer, got {period} (type: {type(period)})")
        if std_dev < 0:
            raise ValueError(f"BollingerBands std_dev must be non-negative, got {std_dev}")
            
        self.period = period
        self.std_dev = std_dev
        self.source_field = source_field
        
        if dependencies is None:
            dependencies = [source_field]
        
        # 输出名称：上轨、中轨、下轨
        output_names = ['upper', 'middle', 'lower']
        super().__init__(name, dependencies, output_names)
    
    def calculate(self, data_buffer: DataBuffer) -> Dict[str, float]:
        """
        计算布林带的所有值，采用talib相同的方式：
        中轨 = 简单移动平均线(SMA)
        上轨 = SMA + (标准差 * 倍数)
        下轨 = SMA - (标准差 * 倍数)
        注意：talib使用的是样本标准差（ddof=0）
        
        Returns:
            包含'upper', 'middle', 'lower'的字典
        """
        bar_size = data_buffer.get_bar_size()
        
        if bar_size < self.period:
            return {
                'upper': np.nan,
                'middle': np.nan,
                'lower': np.nan
            }
        
        # 获取数据
        data = data_buffer.get_bar_field(self.source_field, self.period)
        
        if len(data) < self.period:
            return {
                'upper': np.nan,
                'middle': np.nan,
                'lower': np.nan
            }
        
        # 中轨 = 简单移动平均
        middle_band = np.mean(data)
        
        # talib使用样本标准差（ddof=0）
        std_deviation = np.std(data, ddof=0)
        
        # 上轨 = 中轨 + (标准差 * 倍数)
        upper_band = middle_band + (std_deviation * self.std_dev)
        
        # 下轨 = 中轨 - (标准差 * 倍数)
        lower_band = middle_band - (std_deviation * self.std_dev)
        
        return {
            'upper': float(upper_band),
            'middle': float(middle_band),
            'lower': float(lower_band)
        }
    
    def __repr__(self) -> str:
        return f"BollingerBands(period={self.period}, std_dev={self.std_dev}, name='{self.name}', source_field='{self.source_field}')"


class ATR(Factor):
    """
    平均真实波幅 (Average True Range)
    按照talib相同的计算方式实现
    """
    
    def __init__(self, period: int, name: str, dependencies: Optional[List[str]] = None):
        """
        初始化ATR因子
        
        Args:
            period: 计算周期
            name: 指标名称（必填）
            dependencies: 依赖列表（可选，如不指定则使用['high', 'low', 'close']）
            
        Raises:
            ValueError: 当period小于等于0时
        """
        if not isinstance(period, int) or period <= 0:
            raise ValueError(f"ATR period must be a positive integer, got {period} (type: {type(period)})")
            
        self.period = period
        self.alpha = 1.0 / period  # EMA的alpha值
        
        if dependencies is None:
            dependencies = ['high', 'low', 'close']
            
        # ATR是单输出因子，使用['default']
        super().__init__(name, dependencies, output_names=['default'])
    
    def calculate(self, data_buffer: DataBuffer) -> float:
        """
        计算ATR值，采用TA-Lib相同的方式：
        1. 计算真实波幅(TR) = max(high-low, abs(high-prev_close), abs(low-prev_close))
        2. ATR = TR的指数移动平均
        
        Args:
            data_buffer: 统一数据缓存对象
            
        Returns:
            float: ATR值，数据不足时返回NaN
        """
        bar_size = data_buffer.get_bar_size()
        
        if bar_size < 2:  # 需要至少2个数据点来计算TR
            return np.nan
        
        try:
            # 获取当前数据
            current_high = data_buffer.get_bar_field('high', 1)[0]
            current_low = data_buffer.get_bar_field('low', 1)[0]
            current_close = data_buffer.get_bar_field('close', 1)[0]
            
            # 获取前一个收盘价
            prev_close = data_buffer.get_bar_field('close', 2)[0]  # 前一个收盘价
            
            # 检查数据有效性
            if any(np.isnan([current_high, current_low, current_close, prev_close])):
                return np.nan
            
            # 计算真实波幅(TR)
            tr1 = current_high - current_low
            tr2 = abs(current_high - prev_close)
            tr3 = abs(current_low - prev_close)
            
            true_range = max(tr1, tr2, tr3)
            
            if bar_size == 2:
                # 第一次计算ATR，直接使用TR值
                return float(true_range)
            
            # 获取前一个ATR值
            previous_atr_data = data_buffer.get_factor_data(self.name, 1, 'default')
            
            if len(previous_atr_data) == 0:
                # 如果没有历史ATR数据，使用当前TR值
                return float(true_range)
            
            previous_atr = previous_atr_data[-1]
            
            if np.isnan(previous_atr):
                # 如果前一个ATR值为NaN，使用当前TR值
                return float(true_range)
            
            # 计算ATR：使用EMA方式
            atr = self.alpha * true_range + (1 - self.alpha) * previous_atr
            
            return float(atr)
            
        except Exception:
            return np.nan
    
    def __repr__(self) -> str:
        return f"ATR(period={self.period}, name='{self.name}')"


class STOCH(Factor):
    """
    随机指标 (Stochastic Oscillator)
    返回%K和%D两个值
    按照talib相同的计算方式实现
    """
    
    def __init__(self, k_period: int, k_slowing: int, d_period: int, name: str, dependencies: Optional[List[str]] = None):
        """
        初始化STOCH因子
        
        Args:
            k_period: %K计算周期
            k_slowing: %K平滑周期
            d_period: %D计算周期
            name: 指标名称（必填）
            dependencies: 依赖列表（可选，如不指定则使用['high', 'low', 'close']）
            
        Raises:
            ValueError: 当周期参数小于等于0时
        """
        if (not isinstance(k_period, int) or k_period <= 0 or
            not isinstance(k_slowing, int) or k_slowing <= 0 or
            not isinstance(d_period, int) or d_period <= 0):
            raise ValueError(
                f"STOCH periods must be positive integers, got k_period={k_period} "
                f"(type: {type(k_period)}), k_slowing={k_slowing} (type: {type(k_slowing)}), "
                f"d_period={d_period} (type: {type(d_period)})"
            )
            
        self.k_period = k_period
        self.k_slowing = k_slowing
        self.d_period = d_period
        
        if dependencies is None:
            dependencies = ['high', 'low', 'close']
            
        # STOCH是多输出因子，返回%K和%D
        output_names = ['k', 'd']
        super().__init__(name, dependencies, output_names)
    
    def calculate(self, data_buffer: DataBuffer) -> Dict[str, float]:
        """
        计算STOCH值，采用TA-Lib相同的方式：
        1. 计算原始%K = (当前收盘价 - 最低价) / (最高价 - 最低价) * 100
        2. %K = 原始%K的SMA(k_slowing)
        3. %D = %K的SMA(d_period)
        
        Returns:
            包含'k', 'd'的字典
        """
        bar_size = data_buffer.get_bar_size()
        
        min_required_bars = self.k_period + self.k_slowing + self.d_period - 2
        
        if bar_size < min_required_bars:
            return {
                'k': np.nan,
                'd': np.nan
            }
        
        try:
            # 获取足够的数据来计算
            lookback = min(bar_size, self.k_period + self.k_slowing + self.d_period)
            
            high_data = data_buffer.get_bar_field('high', lookback)
            low_data = data_buffer.get_bar_field('low', lookback)
            close_data = data_buffer.get_bar_field('close', lookback)
            
            if len(high_data) < min_required_bars:
                return {
                    'k': np.nan,
                    'd': np.nan
                }
            
            # 计算原始%K值序列
            raw_k_values = []
            
            for i in range(self.k_period - 1, len(close_data)):
                # 计算k_period周期内的最高价和最低价
                period_high = np.max(high_data[i - self.k_period + 1:i + 1])
                period_low = np.min(low_data[i - self.k_period + 1:i + 1])
                current_close = close_data[i]
                
                if period_high == period_low:
                    raw_k = 50.0  # 避免除零，设为中间值
                else:
                    raw_k = (current_close - period_low) / (period_high - period_low) * 100.0
                
                raw_k_values.append(raw_k)
            
            if len(raw_k_values) < self.k_slowing:
                return {
                    'k': np.nan,
                    'd': np.nan
                }
            
            # 计算%K（原始%K的SMA）
            k_value = np.mean(raw_k_values[-self.k_slowing:])
            
            # 为了计算%D，我们需要历史的%K值
            # 简化处理：如果有足够的数据，计算多个%K值，然后取%D
            if len(raw_k_values) >= self.k_slowing + self.d_period - 1:
                k_values = []
                for j in range(self.k_slowing - 1, len(raw_k_values)):
                    k_val = np.mean(raw_k_values[j - self.k_slowing + 1:j + 1])
                    k_values.append(k_val)
                
                if len(k_values) >= self.d_period:
                    d_value = np.mean(k_values[-self.d_period:])
                else:
                    d_value = np.nan
            else:
                d_value = np.nan
            
            return {
                'k': float(k_value),
                'd': float(d_value) if not np.isnan(d_value) else np.nan
            }
            
        except Exception:
            return {
                'k': np.nan,
                'd': np.nan
            }
    
    def __repr__(self) -> str:
        return f"STOCH(k_period={self.k_period}, k_slowing={self.k_slowing}, d_period={self.d_period}, name='{self.name}')"


class OBV(Factor):
    """
    能量潮指标 (On Balance Volume)
    按照talib相同的计算方式实现
    """
    
    def __init__(self, name: str, dependencies: Optional[List[str]] = None):
        """
        初始化OBV因子
        
        Args:
            name: 指标名称（必填）
            dependencies: 依赖列表（可选，如不指定则使用['close', 'volume']）
        """
        if dependencies is None:
            dependencies = ['close', 'volume']
            
        # OBV是单输出因子，使用['default']
        super().__init__(name, dependencies, output_names=['default'])
    
    def calculate(self, data_buffer: DataBuffer) -> float:
        """
        计算OBV值，采用TA-Lib相同的方式：
        1. 如果当前收盘价 > 前一收盘价，OBV = 前一OBV + 当前成交量
        2. 如果当前收盘价 < 前一收盘价，OBV = 前一OBV - 当前成交量
        3. 如果当前收盘价 = 前一收盘价，OBV = 前一OBV
        
        Args:
            data_buffer: 统一数据缓存对象
            
        Returns:
            float: OBV值，数据不足时返回NaN
        """
        bar_size = data_buffer.get_bar_size()
        
        if bar_size < 1:
            return np.nan
        
        try:
            # 获取当前数据
            current_close = data_buffer.get_bar_field('close', 1)[0]
            current_volume = data_buffer.get_bar_field('volume', 1)[0]
            
            # 检查数据有效性
            if np.isnan(current_close) or np.isnan(current_volume):
                return np.nan
            
            if bar_size == 1:
                # 第一个数据点，OBV初始值设为当前成交量
                return float(current_volume)
            
            # 获取前一个收盘价
            prev_close = data_buffer.get_bar_field('close', 2)[0]
            
            if np.isnan(prev_close):
                return float(current_volume)
            
            # 获取前一个OBV值
            previous_obv_data = data_buffer.get_factor_data(self.name, 1, 'default')
            
            if len(previous_obv_data) == 0:
                # 如果没有历史OBV数据，初始化为当前成交量
                if current_close > prev_close:
                    return float(current_volume)
                elif current_close < prev_close:
                    return float(-current_volume)
                else:
                    return 0.0
            
            previous_obv = previous_obv_data[-1]
            
            if np.isnan(previous_obv):
                # 如果前一个OBV值为NaN，初始化
                if current_close > prev_close:
                    return float(current_volume)
                elif current_close < prev_close:
                    return float(-current_volume)
                else:
                    return 0.0
            
            # 计算OBV
            if current_close > prev_close:
                obv = previous_obv + current_volume
            elif current_close < prev_close:
                obv = previous_obv - current_volume
            else:
                obv = previous_obv
            
            return float(obv)
            
        except Exception:
            return np.nan
    
    def __repr__(self) -> str:
        return f"OBV(name='{self.name}')"