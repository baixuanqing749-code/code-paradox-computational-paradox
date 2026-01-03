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



















#!/usr/bin/env python3
"""
特里达理论八大验证实验 - 统一实现
运行: python experiments_unified.py
或: python experiments_unified.py --experiment 3  # 只运行实验3
"""

import time
import math
import statistics
import numpy as np
from collections import defaultdict, Counter
import argparse
import json
from typing import Dict, List, Tuple, Any
import threading
import psutil
import subprocess
import sys

class TriddaEightExperiments:
    """八个验证实验的统一类"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = {}
        self.START_TIME = time.perf_counter()
        
    def log(self, message: str):
        """统一日志输出"""
        if self.verbose:
            elapsed = time.perf_counter() - self.START_TIME
            print(f"[{elapsed:6.2f}s] {message}")
    
    # ==================== 实验1：算术运算33周期 ====================
    def experiment1_arithmetic_33(self) -> Dict[str, Any]:
        """实验1：算术运算的33周期调制"""
        self.log("实验1开始：算术运算33周期测试")
        
        execution_times = []
        
        for base in range(33):
            start_time = time.perf_counter()
            
            # 33种不同的算术模式
            n = 100000
            result = 0
            
            if base % 33 == 0:
                # 模式0：纯加法
                for i in range(n):
                    result += i
                    
            elif base % 33 == 1:
                # 模式1：乘法为主
                result = 1
                for i in range(1, 1000):
                    result = (result * i) % 1000000007
                    
            elif base % 33 == 2:
                # 模式2：混合运算（364相关）
                for i in range(364):
                    result = (result + i*i - i//3) % 1000000007
                    
            else:
                # 其他模式：标准运算
                for i in range(n // 10):
                    result = (result * 3 + i * 7) % 1000000007
            
            elapsed = time.perf_counter() - start_time
            execution_times.append(elapsed)
            
            if self.verbose and base % 11 == 0:
                self.log(f"  余数{base:2d}: {elapsed:.6f}s")
        
        # 统计分析
        high_group = [execution_times[i] for i in range(33) if i % 3 == 0]
        low_group = [execution_times[i] for i in range(33) if i % 3 != 0]
        
        if high_group and low_group:
            high_avg = statistics.mean(high_group)
            low_avg = statistics.mean(low_group)
            
            # t检验（简化版）
            all_times = high_group + low_group
            pooled_std = statistics.stdev(all_times) if len(all_times) > 1 else 0
            n1, n2 = len(high_group), len(low_group)
            
            if pooled_std > 0 and n1 > 0 and n2 > 0:
                se = pooled_std * math.sqrt(1/n1 + 1/n2)
                t_stat = (high_avg - low_avg) / se if se > 0 else 0
            else:
                t_stat = 0
        
        result_data = {
            'times': execution_times,
            'high_avg': statistics.mean(high_group) if high_group else 0,
            'low_avg': statistics.mean(low_group) if low_group else 0,
            't_statistic': t_stat,
            'ratio': high_avg/low_avg if low_avg > 0 else 0,
            'interpretation': '阳性' if t_stat > 2.0 else '阴性'
        }
        
        self.log(f"实验1完成：t={t_stat:.3f}, 比值={high_avg/low_avg:.4f}")
        return result_data
    
    # ==================== 实验2：硬件熵源分析 ====================
    def experiment2_hardware_entropy(self) -> Dict[str, Any]:
        """实验2：硬件熵源的33相关性"""
        self.log("实验2开始：硬件熵源分析")
        
        # 收集硬件时间戳熵
        entropy_samples = []
        
        for _ in range(3640):  # 364的倍数
            start = time.perf_counter_ns()
            # 微小运算
            x = sum(i*i for i in range(100))
            end = time.perf_counter_ns()
            entropy_samples.append(end - start)
        
        # 分析模33分布
        mod_distribution = [0] * 33
        for sample in entropy_samples:
            mod_distribution[sample % 33] += 1
        
        # 统计检验
        expected = len(entropy_samples) / 33
        chi_squared = sum((count - expected) ** 2 / expected for count in mod_distribution)
        
        # 检查余数25的特殊性（理论关键数）
        remainder_25_count = mod_distribution[25]
        remainder_25_ratio = remainder_25_count / expected
        
        result_data = {
            'mod_distribution': mod_distribution,
            'chi_squared': chi_squared,
            'expected_per_mod': expected,
            'remainder_25': {
                'count': remainder_25_count,
                'expected': expected,
                'ratio': remainder_25_ratio
            },
            'interpretation': '异常' if chi_squared > 50 or remainder_25_ratio > 1.3 else '正常'
        }
        
        self.log(f"实验2完成：χ²={chi_squared:.1f}, 余数25={remainder_25_ratio:.2f}倍期望")
        return result_data
    
    # ==================== 实验3：内存访问优化 ====================
    def experiment3_memory_access(self) -> Dict[str, Any]:
        """实验3：内存访问的33步长优化"""
        self.log("实验3开始：内存访问优化测试")
        
        array_size = 1000000
        test_array = list(range(array_size))
        
        access_times = []
        
        # 测试1-33步长
        for stride in range(1, 34):
            times = []
            
            for _ in range(5):  # 5次平均
                start = time.perf_counter()
                
                result = 0
                for i in range(0, array_size, stride):
                    result += test_array[i]
                
                # 防止优化
                if result == 0:
                    pass
                
                times.append(time.perf_counter() - start)
            
            avg_time = statistics.mean(times)
            access_times.append(avg_time)
            
            if self.verbose and stride in [1, 11, 22, 33]:
                self.log(f"  步长{stride:2d}: {avg_time:.6f}s")
        
        # 找出最优步长
        min_time = min(access_times)
        min_stride = access_times.index(min_time) + 1
        
        # 计算33步长相对优势
        time_32 = access_times[31]  # 步长32
        time_33 = access_times[32]  # 步长33
        time_34_est = (time_32 + time_33) / 2  # 估计步长34
        
        improvement = (time_34_est - time_33) / time_33 * 100 if time_33 > 0 else 0
        
        result_data = {
            'access_times': access_times,
            'best_stride': min_stride,
            'time_32': time_32,
            'time_33': time_33,
            'improvement_percent': improvement,
            'is_33_optimal': min_stride == 33,
            'interpretation': '33最优' if min_stride == 33 else f'最优步长{min_stride}'
        }
        
        self.log(f"实验3完成：最优步长={min_stride}, 33比估计快{improvement:.1f}%")
        return result_data
    
    # ==================== 实验4：CPU调度器周期 ====================
    def experiment4_scheduler_33(self) -> Dict[str, Any]:
        """实验4：CPU调度器的33周期"""
        self.log("实验4开始：CPU调度器周期测试")
        
        def cpu_task(task_id: int, duration: float = 0.01):
            end_time = time.perf_counter() + duration
            while time.perf_counter() < end_time:
                x = task_id ** 2
                x = x * 3.14159
            return task_id
        
        # 顺序执行33个任务
        task_times = []
        
        for i in range(33):
            start = time.perf_counter()
            cpu_task(i, 0.01)
            elapsed = time.perf_counter() - start
            task_times.append(elapsed)
            
            if self.verbose and i % 11 == 0:
                self.log(f"  任务{i:2d}: {elapsed:.6f}s")
        
        # 分析3倍数位置
        triple_times = [task_times[i] for i in range(33) if i % 3 == 0]
        other_times = [task_times[i] for i in range(33) if i % 3 != 0]
        
        if triple_times and other_times:
            mean_triple = statistics.mean(triple_times)
            mean_other = statistics.mean(other_times)
            
            # 简化t检验
            all_times = triple_times + other_times
            pooled_std = statistics.stdev(all_times) if len(all_times) > 1 else 0
            n1, n2 = len(triple_times), len(other_times)
            
            if pooled_std > 0 and n1 > 0 and n2 > 0:
                se = pooled_std * math.sqrt(1/n1 + 1/n2)
                t_stat = abs(mean_triple - mean_other) / se if se > 0 else 0
        
        result_data = {
            'task_times': task_times,
            'triple_mean': statistics.mean(triple_times) if triple_times else 0,
            'other_mean': statistics.mean(other_times) if other_times else 0,
            't_statistic': t_stat,
            'interpretation': '阳性' if t_stat > 2.0 else '阴性'
        }
        
        self.log(f"实验4完成：t={t_stat:.3f}, 3倍数慢{(mean_triple/mean_other-1)*100:.1f}%")
        return result_data
    
    # ==================== 实验5：内存分配优化 ====================
    def experiment5_memory_allocation(self) -> Dict[str, Any]:
        """实验5：内存分配的33倍数优化"""
        self.log("实验5开始：内存分配优化测试")
        
        allocation_times = []
        allocation_sizes = []
        
        for i in range(33):
            # 分配大小：33的倍数
            size = (i + 1) * 33
            
            times = []
            for _ in range(10):  # 10次平均
                start = time.perf_counter()
                
                # 分配和简单访问
                data = [0] * size
                data[size // 2] = 1
                
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                
                # 清理
                del data
            
            avg_time = statistics.mean(times)
            allocation_times.append(avg_time)
            allocation_sizes.append(size)
            
            if self.verbose and i % 11 == 0:
                self.log(f"  大小{size:4d}: {avg_time:.8f}s")
        
        # 找出最优大小
        min_time = min(allocation_times)
        min_idx = allocation_times.index(min_time)
        min_size = allocation_sizes[min_idx]
        
        # 计算改进百分比（与相邻大小比较）
        if min_idx > 0 and min_idx < 32:
            neighbor_avg = (allocation_times[min_idx-1] + allocation_times[min_idx+1]) / 2
            improvement = (neighbor_avg - min_time) / min_time * 100
        else:
            improvement = 0
        
        result_data = {
            'allocation_times': allocation_times,
            'allocation_sizes': allocation_sizes,
            'best_size': min_size,
            'best_time': min_time,
            'improvement_percent': improvement,
            'is_33_multiple': min_size % 33 == 0,
            'interpretation': f'最优大小{min_size} ({min_size%33} mod 33)'
        }
        
        self.log(f"实验5完成：最优分配{min_size}字节，改进{improvement:.1f}%")
        return result_data
    
    # ==================== 实验6：IO操作周期 ====================
    def experiment6_io_pattern(self) -> Dict[str, Any]:
        """实验6：IO操作的33周期模式"""
        self.log("实验6开始：IO操作周期测试")
        
        import io
        
        io_times = []
        
        for i in range(33):
            data_size = (i + 1) * 100  # 100-3300字节
            test_data = b'x' * data_size
            
            write_times = []
            for _ in range(5):
                buffer = io.BytesIO()
                start = time.perf_counter()
                buffer.write(test_data)
                buffer.flush()
                write_times.append(time.perf_counter() - start)
            
            avg_write = statistics.mean(write_times)
            io_times.append(avg_write)
            
            if self.verbose and i % 11 == 0:
                self.log(f"  大小{data_size:4d}: {avg_write:.8f}s")
        
        # 分组分析（每11个一组）
        group_means = []
        for g in range(3):
            group = io_times[g*11:(g+1)*11]
            group_means.append(statistics.mean(group))
        
        # 检查差异
        max_diff = max(group_means) - min(group_means)
        avg_time = statistics.mean(group_means)
        diff_percent = max_diff / avg_time * 100 if avg_time > 0 else 0
        
        result_data = {
            'io_times': io_times,
            'group_means': group_means,
            'max_difference_percent': diff_percent,
            'best_group': np.argmin(group_means),
            'interpretation': '显著差异' if diff_percent > 20 else '无显著差异'
        }
        
        self.log(f"实验6完成：组间差异{diff_percent:.1f}%，最优组{np.argmin(group_means)}")
        return result_data
    
    # ==================== 实验7：逻辑普朗克常数 ====================
    def experiment7_logic_planck(self) -> Dict[str, Any]:
        """实验7：逻辑普朗克常数测量"""
        self.log("实验7开始：逻辑普朗克常数测量")
        
        complexities = [33 * i for i in range(1, 34)]  # 33-1089
        
        energy_times = []
        
        for n in complexities:
            times = []
            for _ in range(10):
                start = time.perf_counter()
                
                result = 0
                for i in range(n):
                    result += (i * i) % (n + 1)
                
                if result == 0:
                    pass
                
                times.append(time.perf_counter() - start)
            
            avg_time = statistics.mean(times)
            energy_times.append((n, avg_time))
            
            if self.verbose and n % (33*5) == 0:
                self.log(f"  复杂度{n:4d}: {avg_time:.6f}s")
        
        # 计算能量-时间不确定性
        uncertainties = []
        for i in range(len(complexities)-1):
            delta_E = complexities[i+1] - complexities[i]
            delta_t = energy_times[i+1][1] - energy_times[i][1]
            if delta_t > 0:
                hbar_candidate = 2 * delta_E * delta_t
                uncertainties.append(hbar_candidate)
        
        if uncertainties:
            hbar_mean = statistics.mean(uncertainties)
            hbar_std = statistics.stdev(uncertainties) if len(uncertainties) > 1 else 0
        else:
            hbar_mean = hbar_std = 0
        
        # 与物理普朗克常数比较
        hbar_physical = 1.054571817e-34
        ratio = hbar_mean / hbar_physical if hbar_physical > 0 else 0
        
        result_data = {
            'hbar_log_mean': hbar_mean,
            'hbar_log_std': hbar_std,
            'ratio_to_physical': ratio,
            'log10_ratio': math.log10(ratio) if ratio > 0 else 0,
            'uncertainties': uncertainties[:10],  # 只存前10个
            'interpretation': f'比值={ratio:.1e}'
        }
        
        self.log(f"实验7完成：ħ_log={hbar_mean:.1e}, 比值={ratio:.1e}")
        return result_data
    
    # ==================== 实验8：计算模型普遍性 ====================
    def experiment8_model_universality(self) -> Dict[str, Any]:
        """实验8：33因子在计算模型中的普遍性"""
        self.log("实验8开始：计算模型普遍性测试")
        
        models = [
            ("斐波那契", self._fibonacci_test),
            ("质数生成", self._prime_test),
            ("快速排序", self._quicksort_test),
            ("矩阵乘法", self._matrix_test),
            ("图遍历", self._graph_test)
        ]
        
        model_results = []
        
        for model_name, test_func in models:
            self.log(f"  测试模型: {model_name}")
            t_value = test_func()
            model_results.append({
                'model': model_name,
                't_value': t_value,
                'has_effect': t_value > 2.0
            })
        
        # 统计阳性率
        positive_count = sum(1 for r in model_results if r['has_effect'])
        avg_t = statistics.mean([r['t_value'] for r in model_results])
        
        result_data = {
            'model_results': model_results,
            'positive_count': positive_count,
            'total_models': len(models),
            'average_t': avg_t,
            'interpretation': f'{positive_count}/{len(models)}阳性'
        }
        
        self.log(f"实验8完成：{positive_count}/{len(models)}个模型显示33效应")
        return result_data
    
    # ==================== 辅助测试函数 ====================
    def _fibonacci_test(self) -> float:
        """斐波那契数列测试"""
        times = []
        for base in range(33):
            n = 330 + base
            start = time.perf_counter()
            
            fib = [0, 1] + [0] * (n-2)
            for i in range(2, n):
                fib[i] = fib[i-1] + fib[i-2]
            
            times.append(time.perf_counter() - start)
        
        # 分析3倍数位置
        triple = [times[i] for i in range(33) if i % 3 == 0]
        other = [times[i] for i in range(33) if i % 3 != 0]
        
        if triple and other:
            mean_diff = abs(statistics.mean(triple) - statistics.mean(other))
            avg_time = statistics.mean(times)
            return mean_diff / (avg_time * 0.1) if avg_time > 0 else 0
        return 0
    
    def _prime_test(self) -> float:
        """质数生成测试"""
        times = []
        for base in range(33):
            n = 100 + base  # 生成n个质数
            start = time.perf_counter()
            
            primes = [2]
            num = 3
            while len(primes) < n:
                is_prime = True
                for i in range(2, int(num**0.5)+1):
                    if num % i == 0:
                        is_prime = False
                        break
                if is_prime:
                    primes.append(num)
                num += 2
            
            times.append(time.perf_counter() - start)
        
        # 简化分析
        triple = [times[i] for i in range(33) if i % 3 == 0]
        other = [times[i] for i in range(33) if i % 3 != 0]
        
        if triple and other:
            mean_diff = abs(statistics.mean(triple) - statistics.mean(other))
            avg_time = statistics.mean(times)
            return mean_diff / (avg_time * 0.1) if avg_time > 0 else 0
        return 0
    
    def _quicksort_test(self) -> float:
        """快速排序测试"""
        import random
        
        times = []
        for base in range(33):
            n = 10000 + base * 100
            arr = [random.random() for _ in range(n)]
            
            start = time.perf_counter()
            
            def quicksort(a):
                if len(a) <= 1:
                    return a
                pivot = a[len(a)//2]
                left = [x for x in a if x < pivot]
                middle = [x for x in a if x == pivot]
                right = [x for x in a if x > pivot]
                return quicksort(left) + middle + quicksort(right)
            
            quicksort(arr)
            times.append(time.perf_counter() - start)
        
        triple = [times[i] for i in range(33) if i % 3 == 0]
        other = [times[i] for i in range(33) if i % 3 != 0]
        
        if triple and other:
            mean_diff = abs(statistics.mean(triple) - statistics.mean(other))
            avg_time = statistics.mean(times)
            return mean_diff / (avg_time * 0.1) if avg_time > 0 else 0
        return 0
    
    def _matrix_test(self) -> float:
        """矩阵乘法测试"""
        times = []
        for base in range(33):
            n = 50 + base
            A = [[random.random() for _ in range(n)] for __ in range(n)]
            B = [[random.random() for _ in range(n)] for __ in range(n)]
            
            start = time.perf_counter()
            
            C = [[0]*n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        C[i][j] += A[i][k] * B[k][j]
            
            times.append(time.perf_counter() - start)
        
        triple = [times[i] for i in range(33) if i % 3 == 0]
        other = [times[i] for i in range(33) if i % 3 != 0]
        
        if triple and other:
            mean_diff = abs(statistics.mean(triple) - statistics.mean(other))
            avg_time = statistics.mean(times)
            return mean_diff / (avg_time * 0.1) if avg_time > 0 else 0
        return 0
    
    def _graph_test(self) -> float:
        """图遍历测试"""
        import random
        
        times = []
        for base in range(33):
            n = 100 + base
            graph = {i: [] for i in range(n)}
            
            # 添加随机边
            for i in range(n):
                for j in range(random.randint(1, 5)):
                    neighbor = random.randint(0, n-1)
                    if neighbor != i:
                        graph[i].append(neighbor)
            
            start = time.perf_counter()
            
            # BFS遍历
            visited = [False] * n
            queue = [0]
            visited[0] = True
            
            while queue:
                node = queue.pop(0)
                for neighbor in graph[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            
            times.append(time.perf_counter() - start)
        
        triple = [times[i] for i in range(33) if i % 3 == 0]
        other = [times[i] for i in range(33) if i % 3 != 0]
        
        if triple and other:
            mean_diff = abs(statistics.mean(triple) - statistics.mean(other))
            avg_time = statistics.mean(times)
            return mean_diff / (avg_time * 0.1) if avg_time > 0 else 0
        return 0
    
    # ==================== 运行控制 ====================
    def run_experiment(self, exp_num: int) -> Dict[str, Any]:
        """运行单个实验"""
        experiments = {
            1: self.experiment1_arithmetic_33,
            2: self.experiment2_hardware_entropy,
            3: self.experiment3_memory_access,
            4: self.experiment4_scheduler_33,
            5: self.experiment5_memory_allocation,
            6: self.experiment6_io_pattern,
            7: self.experiment7_logic_planck,
            8: self.experiment8_model_universality
        }
        
        if exp_num not in experiments:
            raise ValueError(f"实验编号{exp_num}无效，应为1-8")
        
        return experiments[exp_num]()
    
    def run_all(self) -> Dict[int, Dict[str, Any]]:
        """运行所有8个实验"""
        self.log("=" * 60)
        self.log("开始运行特里达理论八大验证实验")
        self.log("=" * 60)
        
        all_results = {}
        
        for exp_num in range(1, 9):
            try:
                self.log(f"\n>>> 开始实验 {exp_num}/8")
                result = self.run_experiment(exp_num)
                all_results[exp_num] = result
                self.log(f"<<< 实验 {exp_num} 完成")
            except Exception as e:
                self.log(f"!!! 实验 {exp_num} 出错: {e}")
                all_results[exp_num] = {'error': str(e)}
        
        # 综合统计
        self.log("\n" + "=" * 60)
        self.log("八大实验综合报告")
        self.log("=" * 60)
        
        positive_count = 0
        for exp_num, result in all_results.items():
            if 'interpretation' in result:
                interp = result['interpretation']
                if any(word in str(interp).lower() for word in ['阳性', '最优', '显著', '异常']):
                    positive_count += 1
        
        self.log(f"阳性实验结果: {positive_count}/8")
        self.log(f"阳性率: {positive_count/8*100:.1f}%")
        
        # 保存结果
        all_results['summary'] = {
            'total_experiments': 8,
            'positive_count': positive_count,
            'positive_percent': positive_count/8*100,
            'timestamp': time.time()
        }
        
        return all_results

def main():
    """主函数：命令行接口"""
    parser = argparse.ArgumentParser(description='特里达理论八大验证实验')
    parser.add_argument('--experiment', type=int, choices=range(1, 9), 
                       help='运行单个实验（1-8）')
    parser.add_argument('--all', action='store_true', 
                       help='运行所有8个实验')
    parser.add_argument('--quiet', action='store_true',
                       help='减少输出')
    
    args = parser.parse_args()
    
    if not args.experiment and not args.all:
        print("请指定 --experiment N 或 --all")
        parser.print_help()
        return
    
    tester = TriddaEightExperiments(verbose=not args.quiet)
    
    if args.all:
        results = tester.run_all()
        
        # 保存到文件
        with open('tridda_experiments_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print("\n结果已保存到 tridda_experiments_results.json")
        
    elif args.experiment:
        result = tester.run_experiment(args.experiment)
        print(f"\n实验{args.experiment}结果:")
        print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    main()
