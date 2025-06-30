"""
QuantBot使用示例

展示如何使用QuantBot框架进行因子计算
使用新的Factor架构和增强的QuantBotEngine
演示统一的因子接口、缓存管理、性能监控等功能
"""

import numpy as np
from datetime import datetime, timedelta
from quantbot.core.data import BarData
from quantbot.engine import QuantBotEngine
from quantbot.factors.technical import SMA, EMA, RSI, MACD, BollingerBands, ATR, STOCH, OBV


def generate_sample_data(num_bars: int = 100) -> list:
    """
    生成示例数据
    
    Args:
        num_bars: 生成的Bar数量
        
    Returns:
        BarData列表
    """
    bars = []
    base_time = datetime.now()
    base_price = 100.0
    
    for i in range(num_bars):
        # 生成随机价格变化
        price_change = np.random.normal(0, 1) * 0.5
        base_price += price_change
        
        # 确保价格为正
        base_price = max(base_price, 10.0)
        
        # 生成OHLC数据
        high = base_price + abs(np.random.normal(0, 0.5))
        low = base_price - abs(np.random.normal(0, 0.5))
        open_price = base_price + np.random.normal(0, 0.2)
        close = base_price
        
        # 确保OHLC逻辑正确
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        volume = np.random.uniform(1000, 10000)
        amount = volume * close
        
        bar = BarData(
            timestamp=base_time + timedelta(minutes=i),
            open_price=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
            amount=amount
        )
        
        bars.append(bar)
    
    return bars


def main():
    """主函数"""
    print("QuantBot框架示例 - 增强版QuantBotEngine")
    print("=" * 60)
    buffer_size = 60
    
    # 1. 创建QuantBot引擎
    print("1. 创建QuantBot引擎...")
    engine = QuantBotEngine(buffer_size=buffer_size)
    print(f"   引擎状态: {engine}")
    print(f"   初始健康状态: {engine.get_health_status()['status']}")
    
    # 2. 注册技术指标因子
    print("2. 注册技术指标因子...")
    
    try:
        # 基础移动平均线
        sma_5 = SMA(5, 'sma_5_close')
        sma_20 = SMA(20, 'sma_20_close')
        ema_12 = EMA(12, 'ema_12_close')
        ema_26 = EMA(26, 'ema_26_close')
        
        # RSI指标
        rsi_14 = RSI(14, 'rsi_14_close')
        
        # MACD指标系列 (合并为单个因子，输出macd, signal, histogram)
        macd = MACD(12, 26, 9, 'macd_12_26_9_close')
        
        # 布林带指标（多输出因子）
        bb = BollingerBands(20, 2.0, 'bb_20_2')
        
        # 新增技术指标
        atr_14 = ATR(14, 'atr_14')
        stoch = STOCH(14, 3, 3, 'stoch_14_3_3')
        obv = OBV('obv')
        
        # 注册所有因子
        factors = [
            sma_5, sma_20, ema_12, ema_26, rsi_14,
            macd, bb,
            atr_14, stoch, obv
        ]
        
        for factor in factors:
            engine.register_factor(factor)
        
        print(f"   成功注册 {len(factors)} 个因子")
        print(f"   实际因子输出数量: {len(engine.get_registered_factors())} 个")
        print(f"   MACD(3个输出) + 布林带(3个输出) + STOCH(2个输出) + 其他单输出因子")
        
    except Exception as e:
        print(f"   因子注册失败: {e}")
        return
    
    # 3. 检查依赖关系和引擎状态
    print("3. 分析因子依赖关系...")
    calculation_order = engine.get_calculation_order()
    print(f"   计算顺序: {calculation_order}...（显示前5个）")
    
    dependency_info = engine.get_dependency_info()
    print("   主要依赖关系:")
    for factor_name, deps in list(dependency_info.items()):
        if deps:
            print(f"     {factor_name} <- {deps}")
    
    # 检查循环依赖
    if engine.has_circular_dependency():
        print("   ⚠️  警告: 检测到循环依赖!")
    else:
        print("   ✓ 依赖关系正常，无循环依赖")
    
    # 4. 生成和添加数据
    print("\n4. 生成示例数据并开始计算...")
    sample_data = generate_sample_data(100)  # 增加数据量以确保所有指标都有输出
    
    print("   开始逐个添加数据并计算因子...")
    results_history = []
    
    for i, bar in enumerate(sample_data):
        try:
            # 添加Bar数据
            engine.add_bar(bar)
            
            # 计算所有因子
            results = engine.calculate_factors()
            
            results_history.append(results.copy())
            
            # 每15个Bar打印一次结果
            if (i + 1) % 10 == 0:
                print(f"\n   第 {i + 1} 个Bar的因子计算结果:")
                print(f"     时间: {bar.timestamp.strftime('%H:%M:%S')}")
                print(f"     收盘价: {bar.close:.2f}")

                # 显示主要指标
                key_indicators = ['sma_5_close', 'ema_12_close', 'rsi_14_close', 'bb_20_2', 'atr_14', 'obv']
                for name in key_indicators:
                    value = results.get(name)
                    print(f" {name}: {value}")
                        
        except Exception as e:
            print(f"   第 {i + 1} 个Bar处理失败: {e}")
            continue
    
    # 5. 展示最终结果和性能统计
    print("\n5. 最终计算结果和性能统计:")
    print("-" * 60)
    
    final_results = results_history[-1] if results_history else {}
    
    if final_results:
        print("   最新因子值 (显示前10个):")
        for i, (name, value) in enumerate(sorted(final_results.items())):
            if i >= 10:  # 只显示前10个
                break
            print(f"     {name}: {value}")
    
    # 显示引擎统计信息
    stats = engine.get_statistics()
    print(f"\n   引擎性能统计:")
    print(f"     总计算次数: {stats['calculation_count']}")
    print(f"     平均计算时间: {stats['average_calculation_time']*1000:.2f}ms")
    print(f"     数据缓存大小: {stats['current_data_size']}")
    print(f"     注册因子数量: {stats['registered_factors']}")
    
    if 'performance' in stats:
        perf = stats['performance']
        print(f"     串行计算次数: {perf['sequential_count']}")
        print(f"     并行计算次数: {perf['parallel_count']}")
        if perf['sequential_avg_time'] > 0 and perf['parallel_avg_time'] > 0:
            efficiency = perf['sequential_avg_time'] / perf['parallel_avg_time']
            print(f"     并行效率: {efficiency:.2f}x")
    
    # 6. 展示增强的因子访问接口
    print("\n6. 增强的因子访问接口演示:")
    print("-" * 60)
    
    # 通过引擎访问单输出因子
    print("   单输出因子示例 (SMA):")
    try:
        sma_latest = engine.get_factor_value('sma_5_close', 1)
        if len(sma_latest) > 0 and not np.isnan(sma_latest[0]):
            print(f"     SMA(5)最新值: {sma_latest[0]:.4f}")
            
            # 获取SMA历史数据
            sma_history = engine.get_factor_value('sma_5_close', 5)
            if len(sma_history) > 0:
                valid_history = [f'{x:.2f}' for x in sma_history if not np.isnan(x)]
                print(f"     SMA(5)最近{len(valid_history)}个值: {valid_history}")
        else:
            print("     SMA(5): 暂无有效数据")
    except Exception as e:
        print(f"     SMA(5)获取失败: {e}")
    
    # 通过引擎访问多输出因子
    print("\n   多输出因子示例 (布林带):")
    try:
        # 获取布林带各条线（使用基础因子名称和output_key）
        bb_upper = engine.get_factor_value('bb_20_2', 1, 'upper')
        bb_middle = engine.get_factor_value('bb_20_2', 1, 'middle')
        bb_lower = engine.get_factor_value('bb_20_2', 1, 'lower')
        
        if (len(bb_upper) > 0 and len(bb_middle) > 0 and len(bb_lower) > 0 and
            not np.isnan(bb_upper[0]) and not np.isnan(bb_middle[0]) and not np.isnan(bb_lower[0])):
            print(f"     布林带上轨: {bb_upper[0]:.4f}")
            print(f"     布林带中轨: {bb_middle[0]:.4f}")
            print(f"     布林带下轨: {bb_lower[0]:.4f}")
            
            # 布林带历史数据
            bb_upper_history = engine.get_factor_value('bb_20_2', 3, 'upper')
            if len(bb_upper_history) > 0:
                valid_history = [f'{x:.2f}' for x in bb_upper_history if not np.isnan(x)]
                print(f"     布林带上轨历史{len(valid_history)}个值: {valid_history}")
        else:
            print("     布林带: 暂无有效数据")
    except Exception as e:
        print(f"     布林带获取失败: {e}")
    
    # 展示新技术指标
    print("\n   新技术指标示例:")
    try:
        atr_data = engine.get_factor_value('atr_14', 1)
        if len(atr_data) > 0 and not np.isnan(atr_data[0]):
            print(f"     ATR(14): {atr_data[0]:.4f}")
        else:
            print("     ATR(14): 暂无有效数据")
    except Exception as e:
        print(f"     ATR获取失败: {e}")
    
    try:
        stoch_k_data = engine.get_factor_value('stoch_14_3_3', 1, 'k')
        stoch_d_data = engine.get_factor_value('stoch_14_3_3', 1, 'd')
        if (len(stoch_k_data) > 0 and len(stoch_d_data) > 0 and 
            not np.isnan(stoch_k_data[0]) and not np.isnan(stoch_d_data[0])):
            print(f"     STOCH %K: {stoch_k_data[0]:.2f}")
            print(f"     STOCH %D: {stoch_d_data[0]:.2f}")
        else:
            print("     STOCH: 暂无有效数据")
    except Exception as e:
        print(f"     STOCH获取失败: {e}")
    
    try:
        obv_data = engine.get_factor_value('obv', 1)
        if len(obv_data) > 0 and not np.isnan(obv_data[0]):
            print(f"     OBV: {obv_data[0]:.0f}")
        else:
            print("     OBV: 暂无有效数据")
    except Exception as e:
        print(f"     OBV获取失败: {e}")
    
    # 7. 健康状态和内存监控
    print("\n7. 引擎健康状态和内存监控:")
    print("-" * 60)
    
    health = engine.get_health_status()
    print(f"   健康状态: {health['status']}")
    print(f"   健康评分: {health['health_score']:.2f}")
    print(f"   错误率: {health['error_rate']:.2%}")
    
    if 'memory_usage' in health:
        memory = health['memory_usage']
        if 'total_bytes' in memory:
            total_kb = memory['total_bytes'] / 1024
            print(f"   估计内存使用: {total_kb:.1f} KB")
    
    # 展示注册的因子
    try:
        registered_factors = engine.get_registered_factors()
        print(f"\n   注册的因子:")
        print(f"     因子数量: {len(registered_factors)}")
        for i, name in enumerate(list(registered_factors)[:5]):
            print(f"     {i+1}. {name}")
    except Exception as e:
        print(f"\n   获取注册因子失败: {e}")
    
    # 8. 数据获取和缓存演示
    print("\n8. 数据获取和缓存演示:")
    print("-" * 60)
    
    # 获取最近的市场数据
    try:
        recent_closes = engine.get_basic_data('close', 5)
        print(f"   最近5个收盘价: {[f'{x:.2f}' for x in recent_closes]}")
    except Exception as e:
        print(f"   获取收盘价失败: {e}")
    
    # 获取最近的因子值
    try:
        recent_sma = engine.get_factor_value('sma_5_close', 5)
        if len(recent_sma) > 0:
            print(f"   最近5个SMA(5)值: {[f'{x:.2f}' for x in recent_sma]}")
    except Exception as e:
        print(f"   获取SMA历史失败: {e}")
    
    # 性能基准测试（直接使用引擎统计信息）
    print("\n   性能基准测试:")
    engine_stats = engine.get_statistics()
    
    print(f"     总计算次数: {engine_stats['calculation_count']}")
    print(f"     平均计算时间: {engine_stats['average_calculation_time']*1000:.2f}ms")
    print(f"     最后计算时间: {engine_stats['last_calculation_time']*1000:.2f}ms")
    print(f"     总计算时间: {engine_stats['total_calculation_time']*1000:.2f}ms")
    
    # 9. MACD指标系列验证
    print("\n9. MACD指标系列验证:")
    print("-" * 60)
    
    macd_values = {
        'MACD线': final_results.get('macd_12_26_9_close_macd'),
        'MACD信号线': final_results.get('macd_12_26_9_close_signal'),
        'MACD柱状图': final_results.get('macd_12_26_9_close_histogram')
    }
    
    print("   通过引擎获取的MACD结果:")
    for name, value in macd_values.items():
        try:
            if value is not None and isinstance(value, (int, float)) and not np.isnan(value):
                print(f"     {name}: {value:.6f}")
            else:
                print(f"     {name}: 暂无数据")
        except (TypeError, ValueError):
            print(f"     {name}: 暂无数据")
    
    # 通过引擎直接获取MACD结果
    print("\n   通过引擎直接获取的MACD结果:")
    try:
        macd_data = engine.get_factor_value('macd_12_26_9_close', 1, 'macd')
        macd_signal_data = engine.get_factor_value('macd_12_26_9_close', 1, 'signal')
        macd_hist_data = engine.get_factor_value('macd_12_26_9_close', 1, 'histogram')
        
        macd_latest = macd_data[0] if len(macd_data) > 0 and not np.isnan(macd_data[0]) else None
        macd_signal_latest = macd_signal_data[0] if len(macd_signal_data) > 0 and not np.isnan(macd_signal_data[0]) else None
        macd_hist_latest = macd_hist_data[0] if len(macd_hist_data) > 0 and not np.isnan(macd_hist_data[0]) else None
        
        if macd_latest is not None:
            print(f"     MACD线: {macd_latest:.6f}")
        if macd_signal_latest is not None:
            print(f"     MACD信号线: {macd_signal_latest:.6f}")
        if macd_hist_latest is not None:
            print(f"     MACD柱状图: {macd_hist_latest:.6f}")
        
        # 验证MACD关系
        if all(x is not None for x in [macd_latest, macd_signal_latest, macd_hist_latest]):
            expected_hist = macd_latest - macd_signal_latest
            print(f"\n   MACD关系验证:")
            print(f"     计算的柱状图: {macd_hist_latest:.6f}")
            print(f"     期望的柱状图: {expected_hist:.6f}")
            print(f"     差异: {abs(macd_hist_latest - expected_hist):.8f}")
    except Exception as e:
        print(f"   MACD获取失败: {e}")
    
    # 10. 缓存管理和清理演示
    print("\n10. 缓存管理和清理演示:")
    print("-" * 60)
    
    print(f"   当前状态:")
    print(f"     引擎总计算次数: {engine.calculation_count}")
    print(f"     数据缓存大小: {engine.data_buffer.get_bar_size()}")
    print(f"     总处理Bar数量: {engine.data_buffer.bar_buffer.total_bars}")
    
    # 显示各因子的total_factor
    print(f"\n   各因子的总计算次数:")
    factor_buffers = engine.data_buffer.factor_buffer
    for factor_name, factor_buffer in factor_buffers.items():
        total_factor = getattr(factor_buffer, 'total_factor', 0)
        print(f"     {factor_name}: {total_factor}")
    
    # 最终健康检查
    try:
        final_health = engine.get_health_status()
        print(f"\n   最终健康状态: {final_health['status']}")
    except Exception as e:
        print(f"\n   健康状态获取失败: {e}")
    
    print("\n" + "=" * 60)
    print("增强版QuantBot框架完成！主要特性:")
    print("1. ✓ 统一的Factor类设计，支持单输出和多输出")
    print("2. ✓ 每个因子拥有独立的缓存器和状态管理") 
    print("3. ✓ 增强的QuantBotEngine，支持并行计算和性能监控")
    print("4. ✓ 完善的错误处理、健康监控和内存管理")
    print("5. ✓ 丰富的技术指标库（SMA, EMA, RSI, MACD, BB, ATR, STOCH, OBV）")
    print("6. ✓ 灵活的因子结果访问接口和数据获取方法")
    print("7. ✓ 智能并行计算策略和性能基准测试")
    print("8. ✓ 全面的单元测试覆盖和数据验证")
    
    print("\n示例运行完成！")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断执行")
    except Exception as e:
        print(f"\n示例执行出错: {e}")
        import traceback
        traceback.print_exc()