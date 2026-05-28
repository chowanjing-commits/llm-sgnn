"""
快速运行 20 Newsgroups 数据集上的嵌入对比实验
自动运行 Sentence-BERT 和 HuggingFace Qwen 的对比实验
"""
import os
import subprocess
import sys
import argparse

def run_command(cmd, description):
    """运行命令并显示进度"""
    print("\n" + "="*80)
    print(f"Running: {description}")
    print("="*80)
    print(f"Command: {cmd}")
    print()
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error running: {description}")
        return False
    else:
        print(f"\n✅ Completed: {description}")
        return True

def main():
    parser = argparse.ArgumentParser(description='Run comparison experiments on 20 Newsgroups')
    parser.add_argument('--skip-sentence-bert', action='store_true',
                       help='Skip Sentence-BERT baseline (if already run)')
    parser.add_argument('--skip-hf-qwen', action='store_true',
                       help='Skip HuggingFace Qwen experiment')
    parser.add_argument('--also-run-hf-qwen-no-pca', action='store_true',
                       help='Also run HuggingFace Qwen experiment without PCA')
    parser.add_argument('--hf-qwen-model', type=str, default='Qwen/Qwen2.5-1.5B',
                       help='HuggingFace Qwen model name (default: Qwen/Qwen2.5-1.5B)')
    parser.add_argument('--qwen-max-length', type=int, default=256,
                       help='HF Qwen tokenizer max_length (default: 256)')
    parser.add_argument('--qwen-batch-size', type=int, default=4,
                       help='HF Qwen embedding batch size (default: 4)')
    parser.add_argument('--pca-dim', type=int, default=384,
                       help='PCA target dimension (default: 384)')
    parser.add_argument('--include-ollama', action='store_true',
                       help='Also run Ollama Qwen experiment')
    parser.add_argument('--ollama-model', type=str, default='qwen2.5:4b',
                       help='Ollama model name (default: qwen2.5:4b)')
    parser.add_argument('--force-regenerate', action='store_true',
                       help='Force regenerate embeddings')
    
    args = parser.parse_args()
    
    print("="*80)
    print("20 Newsgroups: LLM Embedding Comparison Experiments")
    print("="*80)
    print("\nThis script will run experiments with different LLM embeddings:")
    if not args.skip_sentence_bert:
        print("  1. Sentence-BERT (baseline)")
    if not args.skip_hf_qwen:
        print(f"  2. HuggingFace Qwen + PCA ({args.hf_qwen_model})")
        if args.also_run_hf_qwen_no_pca:
            print(f"  3. HuggingFace Qwen (no PCA) ({args.hf_qwen_model})")
    if args.include_ollama:
        print(f"  4. Ollama Qwen ({args.ollama_model})")
    print("\nEach experiment includes:")
    print("  - Data preprocessing and embedding extraction")
    print("  - Training with multiple drop rates [0%, 25%, 50%, 75%, 90%]")
    print("  - Multiple model baselines (MLP, GCN, LLM-GNN)")
    print("  - 3 runs per configuration for statistical significance")
    print()
    
    force_flag = '--force-regenerate' if args.force_regenerate else ''
    success_count = 0
    total_count = 0
    
    # 1. Sentence-BERT baseline
    if not args.skip_sentence_bert:
        total_count += 1
        print("\n" + "🔵 "*40)
        print("Experiment 1/3: Sentence-BERT (Baseline)")
        print("🔵 "*40)
        
        # Preprocess
        if run_command(
            f"python preprocess_20news.py {force_flag}",
            "Preprocessing with Sentence-BERT"
        ):
            # Train
            if run_command(
                "python train_20news.py",
                "Training with Sentence-BERT embeddings"
            ):
                success_count += 1
    
    # 2. HuggingFace Qwen
    if not args.skip_hf_qwen:
        total_count += 1
        print("\n" + "🟢 "*40)
        print(f"Experiment 2/3: HuggingFace Qwen ({args.hf_qwen_model})")
        print("🟢 "*40)
        
        # Preprocess
        if run_command(
            f"python preprocess_20news.py --use-hf-qwen --hf-qwen-model {args.hf_qwen_model} --qwen-max-length {args.qwen_max_length} --qwen-batch-size {args.qwen_batch_size} --use-pca --pca-dim {args.pca_dim} {force_flag}",
            f"Preprocessing with HuggingFace Qwen ({args.hf_qwen_model})"
        ):
            # Train
            if run_command(
                f"python train_20news.py --use-hf-qwen --hf-qwen-model {args.hf_qwen_model} --qwen-max-length {args.qwen_max_length} --qwen-batch-size {args.qwen_batch_size} --use-pca --pca-dim {args.pca_dim}",
                f"Training with HuggingFace Qwen embeddings"
            ):
                success_count += 1

        if args.also_run_hf_qwen_no_pca:
            total_count += 1
            print("\n" + "🟣 "*40)
            print(f"Experiment 3/4: HuggingFace Qwen without PCA ({args.hf_qwen_model})")
            print("🟣 "*40)

            if run_command(
                f"python preprocess_20news.py --use-hf-qwen --hf-qwen-model {args.hf_qwen_model} --qwen-max-length {args.qwen_max_length} --qwen-batch-size {args.qwen_batch_size} {force_flag}",
                f"Preprocessing with HuggingFace Qwen without PCA ({args.hf_qwen_model})"
            ):
                if run_command(
                    f"python train_20news.py --use-hf-qwen --hf-qwen-model {args.hf_qwen_model} --qwen-max-length {args.qwen_max_length} --qwen-batch-size {args.qwen_batch_size}",
                    f"Training with HuggingFace Qwen embeddings without PCA"
                ):
                    success_count += 1
    
    # 3. Ollama Qwen (optional)
    if args.include_ollama:
        total_count += 1
        print("\n" + "🟡 "*40)
        print(f"Experiment 3/3: Ollama Qwen ({args.ollama_model})")
        print("🟡 "*40)
        
        # Preprocess
        if run_command(
            f"python preprocess_20news.py --use-ollama --ollama-model {args.ollama_model} {force_flag}",
            f"Preprocessing with Ollama Qwen ({args.ollama_model})"
        ):
            # Train
            if run_command(
                f"python train_20news.py --use-ollama --ollama-model {args.ollama_model}",
                f"Training with Ollama Qwen embeddings"
            ):
                success_count += 1
    
    # Compare results
    print("\n" + "📊 "*40)
    print("Comparing Results")
    print("📊 "*40)
    
    if success_count >= 2:
        run_command(
            "python compare_embeddings_20news.py --compare-all",
            "Generating comparison report and plots"
        )
    else:
        print("\n⚠️  Not enough experiments completed for comparison")
        print(f"Completed: {success_count}/{total_count}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Completed experiments: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("\n✅ All experiments completed successfully!")
        print("\nResults saved in:")
        print("  - logs/20news_results.csv (Sentence-BERT)")
        print(f"  - logs/20news_results_hf-qwen-len{args.qwen_max_length}-pca{args.pca_dim}.csv (HuggingFace Qwen + PCA)")
        if args.also_run_hf_qwen_no_pca:
            print(f"  - logs/20news_results_hf-qwen-len{args.qwen_max_length}.csv (HuggingFace Qwen, no PCA)")
        if args.include_ollama:
            print("  - logs/20news_results_ollama_qwen.csv (Ollama Qwen)")
        print("  - logs/20news_embedding_comparison.png (Comparison plot)")
    else:
        print(f"\n⚠️  Some experiments failed. Check the output above for errors.")
    
    print("\nTo view detailed comparison:")
    print("  python compare_embeddings_20news.py --compare-all")
    print("\nTo compare two specific methods:")
    print("  python compare_embeddings_20news.py --method1 sentence-bert --method2 hf-qwen")
    print("="*80)

if __name__ == '__main__':
    main()
