"""
代码悖论验证 - 主要验证脚本
作者: baixuanqing749
创建时间: 2024-11-07
"""

import numpy as np
import time
import hashlib
import json
from typing import Dict, List, Tuple, Any

def measure_determinism(func, n_tests: int = 100) -> Tuple[bool, float]:
    """
    测量函数的确定性
    
    参数:
        func: 要测试的函数
        n_tests: 测试次数
    
    返回:
        (是否确定, 不一致比例)
    """
    inconsistencies = 0
    
    # 生成随机测试输入
    np.random.seed(42)  # 固定种子保证可重复
    test_inputs = np.random.randint(0, 2**20, n_tests)
    
    for x in test_inputs:
        # 运行函数两次
        result1 = func(x)
        result2 = func(x)
        
        if result1 != result2:
            inconsistencies += 1
    
    inconsistency_rate = inconsistencies / n_tests
    is_deterministic = inconsistency_rate == 0
    
    return is_deterministic, inconsistency_rate

def measure_sensitivity(func, base_input: int = 1000000, n_variations: int = 100) -> Tuple[float, Dict]:
    """
    测量函数的敏感性（执行时间变化）
    
    参数:
        func: 要测试的函数
        base_input: 基础输入值
        n_variations: 变化次数
    
    返回:
        (变异系数CV, 时间统计信息)
    """
    execution_times = []
    
    # 生成微小变化
    for delta in range(n_variations):
        input_val = base_input + delta
        
        # 测量执行时间（多次取平均）
        times = []
        for _ in range(5):  # 5次测量平均
            start = time.perf_counter_ns()
            _ = func(input_val)
            end = time.perf_counter_ns()
            times.append(end - start)
        
        avg_time = np.mean(times)
        execution_times.append(avg_time)
    
    # 计算统计信息
    times_ns = np.array(execution_times)
    
    stats = {
        'avg_time': float(np.mean(times_ns)),
        'std_time': float(np.std(times_ns)),
        'min_time': float(np.min(times_ns)),
        'max_time': float(np.max(times_ns)),
        'n_measurements': len(execution_times)
    }
    
    # 计算变异系数
    if stats['avg_time'] > 0:
        cv = stats['std_time'] / stats['avg_time']
    else:
        cv = 0.0
    
    return float(cv), stats

def measure_reversibility(func, n_samples: int = 1000) -> Tuple[bool, float]:
    """
    测量函数的信息可逆性（单射性）
    
    参数:
        func: 要测试的函数
        n_samples: 测试样本数
    
    返回:
        (是否单射, 碰撞率)
    """
    outputs = {}
    collisions = 0
    
    np.random.seed(137)  # 固定种子
    for _ in range(n_samples):
        x = np.random.randint(0, 2**31)
        y = func(x)
        
        if y in outputs:
            collisions += 1
        else:
            outputs[y] = x
    
    collision_rate = collisions / n_samples
    is_injective = collisions == 0
    
    return is_injective, collision_rate

def test_paradox(func, func_name: str = "测试函数") -> Dict[str, Any]:
    """
    综合测试函数的代码悖论特性
    
    参数:
        func: 要测试的函数
        func_name: 函数名称
    
    返回:
        测试结果字典
    """
    print(f"\n{'='*60}")
    print(f"测试函数: {func_name}")
    print(f"{'='*60}")
    
    # 1. 测试确定性
    print("\n[1] 确定性测试...")
    is_det, det_rate = measure_determinism(func, 50)
    print(f"   是否确定: {'✅' if is_det else '❌'}")
    print(f"   不一致比例: {det_rate:.6f}")
    
    # 2. 测试敏感性
    print("\n[2] 敏感性测试...")
    cv, time_stats = measure_sensitivity(func, 1000000, 50)
    print(f"   时间变异系数(CV): {cv:.4f}")
    print(f"   平均执行时间: {time_stats['avg_time']:.1f} ns")
    print(f"   敏感(CV>0.01): {'✅' if cv > 0.01 else '❌'}")
    
    # 3. 测试可逆性
    print("\n[3] 可逆性测试...")
    is_inj, coll_rate = measure_reversibility(func, 500)
    print(f"   是否单射: {'✅' if is_inj else '❌'}")
    print(f"   碰撞率: {coll_rate:.6f}")
    print(f"   可逆(碰撞率<0.001): {'✅' if coll_rate < 0.001 else '❌'}")
    
    # 4. 判断悖论是否存在
    paradox_exists = is_det and (cv > 0.01) and (coll_rate < 0.001)
    
    print(f"\n{'='*60}")
    print(f"结论: 代码悖论 {'存在 ✅' if paradox_exists else '不存在 ❌'}")
    print(f"{'='*60}")
    
    return {
        'function_name': func_name,
        'deterministic': is_det,
        'determinism_rate': det_rate,
        'sensitivity_cv': cv,
        'time_statistics': time_stats,
        'injective': is_inj,
        'collision_rate': coll_rate,
        'paradox_exists': paradox_exists,
        'timestamp': time.time()
    }

def main():
    """主函数：测试多个函数"""
    print("代码悖论验证程序")
    print("版本: 1.0.0")
    print("作者: baixuanqing749")
    print("=" * 60)
    
    # 定义测试函数
    test_functions = [
        ("identity", lambda x: x),
        ("constant", lambda x: 42),
        ("linear", lambda x: (1664525 * x + 1013904223) & 0xFFFFFFFF),
        ("hash_trunc8", lambda x: int(hashlib.sha256(str(x).encode()).hexdigest()[:8], 16)),
    ]
    
    results = []
    
    for name, func in test_functions:
        result = test_paradox(func, name)
        results.append(result)
    
    # 保存结果
    with open('results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 统计总结
    print("\n" + "=" * 60)
    print("测试总结:")
    print("=" * 60)
    
    n_paradox = sum(1 for r in results if r['paradox_exists'])
    print(f"测试函数总数: {len(results)}")
    print(f"显示悖论的函数数: {n_paradox}")
    print(f"悖论比例: {n_paradox/len(results)*100:.1f}%")
    
    for r in results:
        status = '✅' if r['paradox_exists'] else '❌'
        print(f"{r['function_name']}: {status}")
    
    print("\n结果已保存到 results.json")
    print("验证完成！")

if __name__ == "__main__":
    main()
