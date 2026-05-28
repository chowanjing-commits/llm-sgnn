"""
测试 Qwen 改进效果
对比三种配置：
1. 原始 Mean Pooling (旧版本，需手动回退代码)
2. Last Token Pooling (新版本)
3. Last Token Pooling + PCA 384D (新版本 + 降维)
"""
import subprocess
import os

def run_experiment(name, command):
    print("\n" + "="*70)
    print(f"Running: {name}")
    print("="*70)
    print(f"Command: {command}")
    print()
    
    result = subprocess.run(command, shell=True, cwd=os.path.dirname(__file__))
    
    if result.returncode != 0:
        print(f"Warning: {name} failed with return code {result.returncode}")
    
    return result.returncode == 0

if __name__ == '__main__':
    print("="*70)
    print("Qwen 改进效果测试")
    print("="*70)
    
    experiments = [
        ("Qwen - Last Token Pooling (改进版)", 
         "python train_20news.py --use-hf-qwen --hf-qwen-model Qwen/Qwen2.5-1.5B --force-regenerate"),
        
        ("Qwen - Last Token Pooling + PCA 384D (改进版 + 降维)", 
         "python train_20news.py --use-hf-qwen --hf-qwen-model Qwen/Qwen2.5-1.5B --use-pca --force-regenerate"),
    ]
    
    results = {}
    for name, cmd in experiments:
        success = run_experiment(name, cmd)
        results[name] = "成功" if success else "失败"
    
    print("\n" + "="*70)
    print("实验完成总结")
    print("="*70)
    for name, status in results.items():
        print(f"  {name}: {status}")
    
    print("\n结果文件:")
    print("  - logs/20news_results_hf_qwen.csv (Last Token Pooling)")
    print("  - 查看嵌入文件名确认 PCA 版本")
    print("\n对比基线:")
    print("  - logs/20news_results.csv (Sentence-BERT)")
