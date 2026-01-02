"""
代码悖论验证脚本
用于验证确定性单射函数同时具有三个特性：
1. 宏观确定性
2. 微观敏感性
3. 信息可逆性
"""

import numpy as np
import time
import hashlib
import json
from typing import Dict, List, Any, Tuple
import sys

class CodeParadoxValidator:
    def __init__(self):
        self.results = {}
    
    def test_determinism(self, func, test_inputs: List[Any], n_repetitions: int = 5) -> bool:
        """
        测试函数的确定性
        """
        is_deterministic = True
        
        for x in test_inputs:
            # 多次运行相同输入
            results = []
            for _ in range(n_repetitions):
                results.append(func(x))
            
            # 检查是否一致
            if not all(r == results[0] for r in results):
                is_deterministic = False
                break
        
        return is_deterministic
    
    def test_sensitivity(self, func, base_input: int = 1000000, n_variations: int = 10) -> Dict[str, float]:
        """
        测试函数的敏感性（执行时间变化）
        """
        execution_times = []
        
        # 测试微小输入变化
        for delta in range(n_variations):
            x = base_input + delta * 0.0001  # 微小变化
            
            # 多次测量取平均
            times = []
            for _ in range(5):
                start = time.perf_counter_ns()
                _ = func(x)
                end = time.perf_counter_ns()
                times.append(end - start)
            
            execution_times.append(np.mean(times))
        
        # 计算统计量
        times_ns = np.array(execution_times)
        
        stats = {
            'mean_ns': np.mean(times_ns),
            'std_ns': np.std(times_ns),
            'cv': np.std(times_ns) / np.mean(times_ns) if np.mean(times_ns) > 0 else 0,
            'min_ns': np.min(times_ns),
            'max_ns': np.max(times_ns),
            'range_ratio': np.max(times_ns) / np.min(times_ns) if np.min(times_ns) > 0 else 0
        }
        
        return stats
    
    def test_reversibility(self, func, n_samples: int = 1000) -> Dict[str, Any]:
        """
        测试函数的可逆性（单射性）
        """
        outputs = {}
        collisions = 0
        collision_details = []
        
        # 生成测试输入
        test_inputs = np.random.randint(0, 2**31, n_samples)
        
        for x in test_inputs:
            y = func(x)
            
            if y in outputs:
                collisions += 1
                if len(collision_details) < 3:  # 记录前3个碰撞
                    collision_details.append({
                        'input1': outputs[y],
                        'input2': x,
                        'output': y
                    })
            else:
                outputs[y] = x
        
        collision_rate = collisions / n_samples
        
        return {
            'collisions': collisions,
            'collision_rate': collision_rate,
            'is_injective': collisions == 0,
            'collision_details': collision_details[:3]
        }
    
    def test_function(self, func, func_name: str) -> Dict[str, Any]:
        """
        全面测试一个函数
        """
        print(f"\n=== 测试函数: {func_name} ===")
        
        # 生成测试输入
        test_inputs = list(range(10, 20))  # 简单测试
        
        # 1. 测试确定性
        is_det = self.test_determinism(func, test_inputs)
        print(f"确定性: {'✅' if is_det else '❌'}")
        
        # 2. 测试敏感性
        sens_stats = self.test_sensitivity(func)
        is_sens = sens_stats['cv'] > 0.01
        print(f"敏感性: {'✅' if is_sens else '❌'} (CV: {sens_stats['cv']:.4f})")
        
        # 3. 测试可逆性
        rev_stats = self.test_reversibility(func, n_samples=500)
        is_rev = rev_stats['is_injective']
        print(f"可逆性: {'✅' if is_rev else '❌'} (碰撞率: {rev_stats['collision_rate']:.6f})")
        
        # 综合判断
        paradox_exists = is_det and is_sens and is_rev
        
        result = {
            'function_name': func_name,
            'determinism': is_det,
            'sensitivity': {
                'value': sens_stats['cv'],
                'is_sensitive': is_sens
            },
            'reversibility': {
                'collisions': rev_stats['collisions'],
                'collision_rate': rev_stats['collision_rate'],
                'is_injective': is_rev
            },
            'paradox_exists': paradox_exists,
            'detailed_stats': {
                'sensitivity_stats': sens_stats,
                'reversibility_stats': rev_stats
            }
        }
        
        if paradox_exists:
            print(f"🎯 代码悖论: {'存在' if paradox_exists else '不存在'}")
        
        return result

def main():
    """主测试函数"""
    print("=" * 60)
    print("代码悖论验证程序")
    print("=" * 60)
    
    validator = CodeParadoxValidator()
    all_results = {}
    
    # 定义测试函数
    test_functions = [
        ("identity", lambda x: x),
        ("linear", lambda x: (1664525 * x + 1013904223) & 0xFFFFFFFF),
        ("quadratic", lambda x: (x * x) & 0xFFFFFFFF),
        ("sha256_trunc8", lambda x: int(
            hashlib.sha256(str(x).encode()).hexdigest()[:8], 16
        )),
    ]
    
    # 测试所有函数
    for func_name, func in test_functions:
        result = validator.test_function(func, func_name)
        all_results[func_name] = result
    
    # 统计结果
    print("\n" + "=" * 60)
    print("综合结果统计")
    print("=" * 60)
    
    paradox_count = sum(1 for r in all_results.values() if r['paradox_exists'])
    total_count = len(all_results)
    
    print(f"测试函数总数: {total_count}")
    print(f"显示悖论的函数: {paradox_count}")
    print(f"悖论比例: {paradox_count/total_count*100:.1f}%")
    
    # 保存结果
    with open('data/experimental_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print("\n结果已保存到: data/experimental_results.json")
    
    return all_results

if __name__ == "__main__":
    results = main()
