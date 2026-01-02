"""
代码悖论验证脚本
运行：python verification.py
"""

import numpy as np
import time
import hashlib
import json
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

@dataclass
class TestResult:
    """测试结果数据类"""
    function_name: str
    is_deterministic: bool
    time_sensitivity: float  # 时间变异系数CV
    collision_rate: float    # 碰撞率
    paradox_exists: bool

def test_determinism(func, n_tests: int = 100) -> bool:
    """
    测试函数确定性
    """
    # 随机测试输入
    np.random.seed(42)
    test_inputs = np.random.randint(0, 1000, n_tests)
    
    for x in test_inputs:
        # 运行多次检查一致性
        results = []
        for _ in range(5):
            results.append(func(x))
        
        if not all(r == results[0] for r in results):
            return False
    
    return True

def test_sensitivity(func, base_input: int = 1000000) -> float:
    """
    测试时间敏感性，返回变异系数(CV)
    """
    execution_times = []
    
    # 测试微小输入变化
    for delta in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        x = base_input + delta
        
        # 多次测量取平均
        times = []
        for _ in range(10):
            start = time.perf_counter_ns()
            _ = func(x)
            end = time.perf_counter_ns()
            times.append(end - start)
        
        execution_times.append(np.mean(times))
    
    # 计算变异系数
    times_array = np.array(execution_times)
    if np.mean(times_array) > 0:
        return np.std(times_array) / np.mean(times_array)
    return 0.0

def test_reversibility(func, n_samples: int = 1000) -> float:
    """
    测试可逆性（单射性），返回碰撞率
    """
    outputs = {}
    collisions = 0
    
    for i in range(n_samples):
        y = func(i)
        if y in outputs:
            collisions += 1
        else:
            outputs[y] = i
    
    return collisions / n_samples

def run_comprehensive_test():
    """
    运行综合测试
    """
    print("=== 代码悖论综合验证 ===")
    print("=" * 50)
    
    # 定义测试函数
    test_functions = [
        ("identity", lambda x: x),
        ("linear", lambda x: (1664525 * x + 1013904223) & 0xFFFFFFFF),
        ("hash_trunc8", lambda x: int(hashlib.sha256(str(x).encode()).hexdigest()[:8], 16)),
        ("constant", lambda x: 42),
    ]
    
    results = []
    
    for name, func in test_functions:
        print(f"\n测试函数: {name}")
        
        # 测试三个特性
        is_det = test_determinism(func)
        sensitivity = test_sensitivity(func)
        collision_rate = test_reversibility(func)
        
        # 判断悖论是否存在
        paradox = (is_det and sensitivity > 0.01 and collision_rate < 0.001)
        
        # 创建结果对象
        result = TestResult(
            function_name=name,
            is_deterministic=is_det,
            time_sensitivity=sensitivity,
            collision_rate=collision_rate,
            paradox_exists=paradox
        )
        
        results.append(result)
        
        # 打印结果
        print(f"  确定性: {'✅' if is_det else '❌'}")
        print(f"  敏感性(CV): {sensitivity:.4f} {'✅' if sensitivity > 0.01 else '❌'}")
        print(f"  碰撞率: {collision_rate:.6f} {'✅' if collision_rate < 0.001 else '❌'}")
        print(f"  代码悖论: {'✅' if paradox else '❌'}")
    
    # 统计总结
    print("\n" + "=" * 50)
    print("测试总结:")
    
    paradox_count = sum(1 for r in results if r.paradox_exists)
    print(f"测试函数总数: {len(test_functions)}")
    print(f"显示悖论的函数数: {paradox_count}")
    print(f"悖论比例: {paradox_count/len(test_functions):.1%}")
    
    # 保存结果到JSON文件
    results_dict = [
        {
            'function': r.function_name,
            'deterministic': r.is_deterministic,
            'sensitivity': r.time_sensitivity,
            'collision_rate': r.collision_rate,
            'paradox': r.paradox_exists
        }
        for r in results
    ]
    
    with open('test_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print("\n详细结果已保存到 test_results.json")
    
    return results

if __name__ == "__main__":
    print("代码悖论验证程序")
    print("开始验证...")
    print("=" * 50)
    
    try:
        results = run_comprehensive_test()
        print("\n✅ 验证完成！")
    except Exception as e:
        print(f"\n❌ 验证出错: {e}")
        print("请确保已安装所需库: pip install numpy")
