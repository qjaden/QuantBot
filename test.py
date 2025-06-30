import talib
import numpy as np
from talib import stream

# 初始化流式计算器
class StreamSMA:
    def __init__(self, period=20):
        self.period = period
        self.prices = []
        
    def update(self, price):
        self.prices.append(price)
        
        # 保持数据长度不超过需要的周期
        if len(self.prices) > self.period:
            self.prices = self.prices[-self.period:]
            
        # 当数据足够时计算SMA
        if len(self.prices) >= self.period:
            return talib.SMA(np.array(self.prices, dtype=float), timeperiod=self.period)[-1]
        else:
            return None

# 使用示例
sma_calculator = StreamSMA(period=5)

# 模拟实时接收价格数据
prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
sma_func_val = talib.SMA(np.array(prices, dtype=float), timeperiod=5)
for price in prices:
    sma_value = sma_calculator.update(price)
    print(f"Price: {price}, SMA(5): {sma_value}")

print(sma_func_val)