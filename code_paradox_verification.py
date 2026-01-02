"""
代码悖论的严格验证
Version: 2.0
Author: Code Paradox Discovery Project
Date: 2024
License: MIT
"""

import numpy as np
import time
import hashlib
import matplotlib.pyplot as plt
from scipy import stats
import json
from typing import Callable, Dict, Any, List
import sys

class CodeParadoxValidator:
    """代码悖论验证器"""
    
    def __init__(self, func: Callable = None):
        """
        初始化验证器
        
        Args:
            func: 要验证的函数，默认为截断SHA256
        """
        if func is None:
            self.func = lambda x: int(
                hashlib.sha256(str(x).encode()).hexdigest()[:8], 
                16
            ) & 0xFFFFFFFF  # 32位保证
        else:
            self.func = func
        
        # 验证函数是纯函数
        self._verify_pure_function()
    
    def _verify_pure_function(self):
        """验证函数是纯函数"""
        test_val = 42
        results = [self.func(test_val) for _ in range(10)]
        if not all(r == results[0] for r in results):
            raise ValueError("函数不是纯函数：相同输入产生不同输出")
    
    def test_determinism(self, n_tests: int = 1000) -> Dict[str, Any]:
        """
        验证确定性
        
        Returns:
            包含测试结果的字典
        """
        print("=== 确定性验证 ===")
        
        np.random.seed(42)
        test_inputs = np.random.randint(0, 2**31, n_tests)
        
        inconsistencies = 0
        inconsistency_details = []
        
        for x in test_inputs:
            results = [self.func(x) for _ in range(5)]  # 5次重复
            
            if not all(r == results[0] for r in results):
                inconsistencies += 1
                inconsistency_details.append({
                    'input': x,
                    'outputs': results[:3]  # 记录前3个
                })
                
                if inconsistencies >= 3:  # 找到3个不一致就停止
                    break
        
        is_deterministic = (inconsistencies == 0)
        
        result = {
            'is_deterministic': is_deterministic,
            'n_tests': n_tests,
            'inconsistencies': inconsistencies,
            'inconsistency_rate': inconsistencies / n_tests,
            'test_samples': inconsistency_details[:2] if inconsistency_details else []
        }
        
        print(f"测试输入数: {n_tests}")
        print(f"发现不一致: {inconsistencies}")
        print(f"确定性: {'✅' if is_deterministic else '❌'}")
        
        return result
    
    def test_sensitivity(self, 
                        base_input: int = 1000000,
                        n_perturbations: int = 100) -> Dict[str, Any]:
        """
        验证敏感性（执行时间对微小输入变化的响应）
        
        Args:
            base_input: 基础输入值
            n_perturbations: 扰动测试次数
            
        Returns:
            包含敏感性测试结果的字典
        """
        print("\n=== 敏感性验证 ===")
        
        # 生成微小扰动
        np.random.seed(137)
        
        # 三种扰动类型：
        # 1. 算术扰动（±1）
        # 2. 位级扰动（翻转一个bit）
        # 3. 随机微小扰动
        perturbations = []
        
        # 类型1：算术微小变化
        perturbations.extend([base_input + i for i in range(-n_perturbations//3, n_perturbations//3)])
        
        # 类型2：位级变化
        for i in range(0, 32, 2):  # 每2位翻转一个
            mask = 1 << i
            perturbations.append(base_input ^ mask)
        
        # 类型3：随机微小扰动
        perturbations.extend([
            base_input + int(np.random.uniform(-100, 100))
            for _ in range(n_perturbations//3)
        ])
        
        # 去重并限制数量
        perturbations = list(set(perturbations))
        perturbations = perturbations[:n_perturbations]
        
        # 测量执行时间（多次测量减少噪声）
        execution_times = []
        
        for x in perturbations:
            times = []
            for _ in range(7):  # 7次测量
                start = time.perf_counter_ns()
                _ = self.func(x)
                end = time.perf_counter_ns()
                times.append(end - start)
            
            # 使用中位数减少异常值影响
            execution_times.append(np.median(times))
        
        # 计算统计特性
        times_ns = np.array(execution_times)
        
        stats_dict = {
            'mean_ns': float(np.mean(times_ns)),
            'median_ns': float(np.median(times_ns)),
            'std_ns': float(np.std(times_ns)),
            'cv': float(np.std(times_ns) / np.mean(times_ns) if np.mean(times_ns) > 0 else 0),
            'min_ns': float(np.min(times_ns)),
            'max_ns': float(np.max(times_ns)),
            'range_ratio': float(np.max(times_ns) / np.min(times_ns) if np.min(times_ns) > 0 else 0),
            'n_measurements': len(execution_times)
        }
        
        # 敏感性判断：CV > 0.01 且统计显著
        cv = stats_dict['cv']
        
        # 统计显著性检验
        # 零假设：时间变化是随机噪声（CV接近0）
        if len(execution_times) >= 10:
            # 自举法估计CV的置信区间
            n_bootstraps = 1000
            bootstrap_cvs = []
            
            for _ in range(n_bootstraps):
                sample = np.random.choice(times_ns, size=len(times_ns), replace=True)
                sample_cv = np.std(sample) / np.mean(sample) if np.mean(sample) > 0 else 0
                bootstrap_cvs.append(sample_cv)
            
            ci_lower = np.percentile(bootstrap_cvs, 2.5)
            ci_upper = np.percentile(bootstrap_cvs, 97.5)
            
            # 敏感性判断：置信区间下限 > 0.01
            is_sensitive = ci_lower > 0.01
            
            stats_dict.update({
                'cv_ci_lower': float(ci_lower),
                'cv_ci_upper': float(ci_upper),
                'is_sensitive': bool(is_sensitive),
                'sensitivity_threshold': 0.01
            })
        else:
            is_sensitive = cv > 0.01
            stats_dict['is_sensitive'] = bool(is_sensitive)
        
        print(f"测试扰动数: {len(perturbations)}")
        print(f"时间变异系数(CV): {cv:.4f}")
        if 'cv_ci_lower' in stats_dict:
            print(f"CV 95%置信区间: [{stats_dict['cv_ci_lower']:.4f}, {stats_dict['cv_ci_upper']:.4f}]")
        print(f"敏感性 (CV > 0.01): {'✅' if is_sensitive else '❌'}")
        
        return stats_dict
    
    def test_reversibility(self, 
                          n_samples: int = 10000) -> Dict[str, Any]:
        """
        验证可逆性（单射性）
        
        Args:
            n_samples: 测试样本数
            
        Returns:
            包含可逆性测试结果的字典
        """
        print("\n=== 可逆性验证 ===")
        
        np.random.seed(271828)
        
        # 生成测试输入
        inputs = np.random.randint(0, 2**31, n_samples)
        
        # 测试单射性
        outputs_dict = {}
        collisions = 0
        collision_details = []
        
        for x in inputs:
            y = self.func(x)
            
            if y in outputs_dict:
                collisions += 1
                if len(collision_details) < 3:  # 记录前3个碰撞
                    collision_details.append({
                        'input1': int(outputs_dict[y]),
                        'input2': int(x),
                        'output': int(y)
                    })
            else:
                outputs_dict[y] = x
        
        collision_rate = collisions / n_samples
        
        # 统计显著性：与随机函数的预期碰撞比较
        # 假设输出空间大小为 M = 2^32
        M = 2**32
        expected_collisions = n_samples * (n_samples - 1) / (2 * M)
        expected_rate = expected_collisions / n_samples
        
        # 二项检验：观察到的碰撞是否显著少于随机预期
        if expected_collisions > 0:
            # 使用泊松近似
            p_value = stats.poisson.cdf(collisions, expected_collisions)
        else:
            p_value = 1.0
        
        # 可逆性判断：碰撞率低且统计显著
        is_reversible = (collision_rate < 0.001) and (p_value < 0.05)
        
        result = {
            'n_samples': n_samples,
            'collisions': int(collisions),
            'collision_rate': float(collision_rate),
            'expected_collisions': float(expected_collisions),
            'expected_collision_rate': float(expected_rate),
            'p_value': float(p_value),
            'is_injective': bool(collisions == 0),
            'is_reversible': bool(is_reversible),
            'collision_threshold': 0.001,
            'collision_details': collision_details[:2]
        }
        
        print(f"测试样本数: {n_samples}")
        print(f"发现碰撞: {collisions}")
        print(f"碰撞率: {collision_rate:.6f}")
        print(f"随机预期碰撞: {expected_collisions:.2f}")
        print(f"统计p值: {p_value:.4f}")
        print(f"可逆性 (碰撞率<0.001且p<0.05): {'✅' if is_reversible else '❌'}")
        
        return result
    
    def comprehensive_test(self, 
                          save_results: bool = True) -> Dict[str, Any]:
        """
        综合测试：验证代码悖论的三个特性
        
        Returns:
            包含所有测试结果的字典
        """
        print("="*60)
        print("代码悖论综合验证")
        print("="*60)
        
        # 执行所有测试
        results = {
            'determinism': self.test_determinism(),
            'sensitivity': self.test_sensitivity(),
            'reversibility': self.test_reversibility()
        }
        
        # 综合判断
        paradox_exists = (
            results['determinism']['is_deterministic'] and
            results['sensitivity'].get('is_sensitive', False) and
            results['reversibility']['is_reversible']
        )
        
        # 计算综合置信度
        confidences = []
        if results['determinism']['is_deterministic']:
            confidences.append(1.0 - results['determinism']['inconsistency_rate'])
        
        if 'cv_ci_lower' in results['sensitivity']:
            conf_ci = min(1.0, results['sensitivity']['cv_ci_lower'] / 0.01)
            confidences.append(conf_ci)
        
        if results['reversibility']['is_reversible']:
            conf_rev = 1.0 - results['reversibility']['p_value']
            confidences.append(conf_rev)
        
        overall_confidence = np.mean(confidences) if confidences else 0.0
        
        final_result = {
            'paradox_exists': bool(paradox_exists),
            'overall_confidence': float(overall_confidence),
            'results': results,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform
            }
        }
        
        print("\n" + "="*60)
        print("验证结果总结")
        print("="*60)
        
        print(f"确定性: {'✅' if results['determinism']['is_deterministic'] else '❌'}")
        print(f"敏感性: {'✅' if results['sensitivity'].get('is_sensitive', False) else '❌'}")
        print(f"可逆性: {'✅' if results['reversibility']['is_reversible'] else '❌'}")
        print(f"综合置信度: {overall_confidence:.3f}")
        print(f"\n代码悖论存在: {'✅' if paradox_exists else '❌'}")
        
        if paradox_exists:
            print("\n🎯 发现确认：这个函数同时具有三个特性！")
            print("   1. 逻辑确定性")
            print("   2. 实现敏感性")
            print("   3. 信息可逆性")
        
        # 保存结果
        if save_results:
            filename = f"paradox_results_{int(time.time())}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, indent=2, ensure_ascii=False)
            print(f"\n结果已保存到: {filename}")
        
        return final_result

# ============================================================================
# 演示函数和测试用例
# ============================================================================

def demo_identity_function():
    """恒等函数演示"""
    print("\n" + "="*60)
    print("测试：恒等函数 f(x) = x")
    print("="*60)
    
    validator = CodeParadoxValidator(func=lambda x: x)
    return validator.comprehensive_test()

def demo_hash_function():
    """哈希函数演示"""
    print("\n" + "="*60)
    print("测试：SHA256截断函数")
    print("="*60)
    
    validator = CodeParadoxValidator()
    return validator.comprehensive_test()

def demo_linear_function():
    """线性函数演示（修复溢出问题）"""
    print("\n" + "="*60)
    print("测试：线性同余生成器")
    print("="*60)
    
    # 使用32位安全参数
    a = 1664525
    c = 1013904223
    m = 2**32
    
    def linear_func(x):
        # 确保使用Python的大整数，避免溢出
        x = int(x) & 0xFFFFFFFF  # 确保32位
        return (a * x + c) & 0xFFFFFFFF
    
    validator = CodeParadoxValidator(func=linear_func)
    return validator.comprehensive_test()

# ============================================================================
# 主程序
# ============================================================================

def main():
    """主函数"""
    print("代码悖论验证程序 v2.0")
    print("="*60)
    print("验证确定性单射函数是否同时具有：")
    print("1. 逻辑确定性")
    print("2. 实现敏感性")
    print("3. 信息可逆性")
    print("="*60)
    
    # 运行演示测试
    tests = [
        ("恒等函数", demo_identity_function),
        ("哈希函数", demo_hash_function),
        ("线性函数", demo_linear_function)
    ]
    
    all_results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"运行测试: {test_name}")
        print(f"{'='*60}")
        
        try:
            result = test_func()
            all_results[test_name] = {
                'paradox': result['paradox_exists'],
                'confidence': result['overall_confidence']
            }
        except Exception as e:
            print(f"测试失败: {e}")
            all_results[test_name] = {'error': str(e)}
    
    # 总结
    print("\n" + "="*60)
    print("所有测试总结")
    print("="*60)
    
    paradox_count = sum(1 for r in all_results.values() 
                       if isinstance(r, dict) and r.get('paradox', False))
    
    print(f"测试函数数: {len(tests)}")
    print(f"显示悖论的函数: {paradox_count}")
    
    for name, result in all_results.items():
        if 'error' in result:
            print(f"{name}: ❌ 错误 - {result['error']}")
        else:
            status = '✅' if result['paradox'] else '❌'
            print(f"{name}: {status} 悖论={result['paradox']}, 置信度={result['confidence']:.3f}")
    
    return all_results

if __name__ == "__main__":
    results = main()
    
    # 保存总结合果
    with open('test_summary.json', 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            'results': results,
            'conclusion': '代码悖论验证完成'
        }, f, indent=2, ensure_ascii=False)
    
    print("\n✅ 验证完成！")
