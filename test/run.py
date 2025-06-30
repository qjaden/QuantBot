import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from buffer_test import TestDataBuffer
from factor_test import TestFactor
from technical_test import TestTechnicalIndicators


def run_tests():
    """运行所有测试"""
    print("运行QuantBot框架测试...")
    print("=" * 50)
    
    # 创建测试套件
    test_classes = [
        TestDataBuffer,
        TestFactor,
        TestTechnicalIndicators,
    ]
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 打印结果摘要
    print("\n" + "=" * 50)
    print("测试结果摘要:")
    print(f"总共运行: {result.testsRun} 个测试")
    print(f"失败: {len(result.failures)} 个")
    print(f"错误: {len(result.errors)} 个")
    
    if result.failures:
        print("\n失败的测试:")
        for test, trace in result.failures:
            print(f"- {test}: {trace}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, trace in result.errors:
            print(f"- {test}: {trace}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n测试结果: {'通过' if success else '失败'}")
    
    return success

if __name__ == "__main__":
    run_tests()