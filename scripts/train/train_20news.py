"""
20 Newsgroups 实验
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc

from src import config
from src.models import MLP, GCN, LLM_GNN
from scripts.preprocess.preprocess_20news import preprocess_20news

import torch.serialization
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage, NodeStorage, EdgeStorage
torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, Data, GlobalStorage, NodeStorage, EdgeStorage])


def random_edge_drop(edge_index, drop_rate):
    if drop_rate <= 0:
        return edge_index
    num_edges = edge_index.size(1)
    keep_mask = torch.rand(num_edges) > drop_rate
    return edge_index[:, keep_mask]


def train_and_eval(data, model_type, use_llm, drop_rate, num_classes, num_epochs=200):
    device = config.DEVICE
    
    # 模拟稀疏性：在构建好的 k-NN 图上随机丢边
    base_edge_raw = getattr(data, 'edge_index_raw', data.edge_index)
    base_edge_sem = getattr(data, 'edge_index_sem', None)
    sparse_edge = random_edge_drop(base_edge_raw, drop_rate) if drop_rate > 0 else base_edge_raw
    
    # 选择特征：Raw (TF-IDF) 或 LLM 嵌入
    x_input = data.x_llm if use_llm else data.x
    in_dim = x_input.size(1)
    
    if model_type == 'MLP':
        model = MLP(in_dim, config.GNN_HIDDEN_DIM, num_classes)
    elif model_type == 'GCN':
        model = GCN(in_dim, config.GNN_HIDDEN_DIM, num_classes)
    elif model_type == 'LLM_GNN':
        model = LLM_GNN(in_dim, config.GNN_HIDDEN_DIM, num_classes,
                       k_neighbors=config.DEFAULT_K_NEIGHBORS, beta=config.DEFAULT_BETA)
    
    model = model.to(device)
    
    exp_data = Data(
        x=data.x.to(device),
        x_llm=data.x_llm.to(device),
        edge_index=sparse_edge.to(device),
        original_edge_index=base_edge_raw.to(device),
        y=data.y.to(device),
        train_mask=data.train_mask.to(device),
        val_mask=data.val_mask.to(device),
        test_mask=data.test_mask.to(device)
    )
    if base_edge_sem is not None:
        exp_data.edge_index_sem = base_edge_sem.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=config.WEIGHT_DECAY)
    
    best_test = 0
    
    # 选择输入特征
    x_input = exp_data.x_llm if use_llm else exp_data.x
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        if model_type == 'LLM_GNN' and hasattr(exp_data, 'edge_index_sem'):
            out = model(
                x_input,
                exp_data.edge_index,
                semantic_edge_index=exp_data.edge_index_sem,
                original_edge_index=exp_data.original_edge_index
            )
        else:
            out = model(x_input, exp_data.edge_index)
        loss = F.cross_entropy(out[exp_data.train_mask], exp_data.y[exp_data.train_mask])
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                if model_type == 'LLM_GNN' and hasattr(exp_data, 'edge_index_sem'):
                    out = model(
                        x_input,
                        exp_data.edge_index,
                        semantic_edge_index=exp_data.edge_index_sem,
                        original_edge_index=exp_data.original_edge_index
                    )
                else:
                    out = model(x_input, exp_data.edge_index)
                pred = out.argmax(dim=1)
                test_acc = (pred[exp_data.test_mask] == exp_data.y[exp_data.test_mask]).float().mean().item()
                best_test = max(best_test, test_acc)
    
    del model, exp_data
    torch.cuda.empty_cache()
    gc.collect()
    
    return best_test


def run_experiments(data, num_classes, llm_type='sentence-bert'):
    """
    Args:
        data: 预处理后的数据
        num_classes: 类别数
        llm_type: LLM 类型，'sentence-bert', 'ollama-qwen', 或 'hf-qwen'
    """
    print("\n" + "="*60)
    print(f"20 Newsgroups Experiments (LLM: {llm_type})")
    print("="*60)
    
    # 统一 drop rates 与其他数据集一致
    drop_rates = [0.0, 0.25, 0.5, 0.75, 0.9]
    
    # 完整基线：与 arxiv/WikiCS 一致
    models_config = [
        ('MLP (LLM)', 'MLP', True),       # MLP + LLM 嵌入
        ('GCN (Raw)', 'GCN', False),      # GCN + TF-IDF (原始特征)
        ('GCN (LLM)', 'GCN', True),       # GCN + LLM 嵌入
        ('Ours (LLM-GNN)', 'LLM_GNN', True),  # LLM-GNN + 动态结构学习
    ]
    
    results = []
    
    for drop_rate in drop_rates:
        print(f"\nDrop Rate: {drop_rate*100:.0f}%")
        
        for name, mtype, use_llm in models_config:
            print(f"  {name}...", end=" ", flush=True)
            
            accs = []
            for seed in range(3):
                config.set_seed(seed)
                acc = train_and_eval(data, mtype, use_llm, drop_rate, num_classes)
                accs.append(acc)
            
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            print(f"Acc: {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%")
            
            results.append({
                'dataset': '20News',
                'drop_rate': drop_rate,
                'model': name,
                'accuracy': mean_acc,
                'std': std_acc,
                'llm_type': llm_type
            })

    # 按 llm_type 分文件保存，避免不同实验互相覆盖
    safe_llm_type = str(llm_type).replace('/', '_').replace(':', '_').replace(' ', '_')
    result_file = f'logs/20news_results_{safe_llm_type}.csv'
    
    df = pd.DataFrame(results)
    df.to_csv(result_file, index=False)
    
    print("\nSummary:")
    pivot = df.pivot(index='model', columns='drop_rate', values='accuracy') * 100
    print(pivot.round(2).to_string())
    
    return df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='20 Newsgroups Experiments')
    parser.add_argument('--use-ollama', action='store_true', 
                       help='Use Ollama qwen2.5:4b instead of Sentence-BERT')
    parser.add_argument('--ollama-model', type=str, default='qwen2.5:4b',
                       help='Ollama model name (default: qwen2.5:4b)')
    parser.add_argument('--use-hf-qwen', action='store_true',
                       help='Use HuggingFace Qwen model instead of Sentence-BERT')
    parser.add_argument('--hf-qwen-model', type=str, default='Qwen/Qwen2.5-1.5B',
                       help='HuggingFace Qwen model name (default: Qwen/Qwen2.5-1.5B)')
    parser.add_argument('--qwen-max-length', type=int, default=256,
                       help='HF Qwen tokenizer max_length (default: 256)')
    parser.add_argument('--qwen-batch-size', type=int, default=4,
                       help='HF Qwen embedding batch size (default: 4)')
    parser.add_argument('--use-pca', action='store_true',
                       help='Use PCA to reduce Qwen embeddings to 384D (same as Sentence-BERT)')
    parser.add_argument('--pca-dim', type=int, default=384,
                       help='PCA target dimension (default: 384)')
    parser.add_argument('--force-regenerate', action='store_true',
                       help='Force regenerate embeddings and graph')
    
    args = parser.parse_args()
    
    print("="*60)
    if args.use_hf_qwen:
        print(f"LLM-GNN Experiments on 20 Newsgroups (HuggingFace Qwen: {args.hf_qwen_model})")
        llm_type = f"hf-qwen-len{args.qwen_max_length}"
        if args.use_pca:
            llm_type += f"-pca{args.pca_dim}"
    elif args.use_ollama:
        print(f"LLM-GNN Experiments on 20 Newsgroups (Ollama: {args.ollama_model})")
        llm_type = 'ollama-qwen'
    else:
        print("LLM-GNN Experiments on 20 Newsgroups (Sentence-BERT)")
        llm_type = 'sentence-bert'
    print("="*60)
    
    config.set_seed()
    
    print("\nLoading/Processing 20 Newsgroups...")
    data, num_classes = preprocess_20news(
        force_regenerate=args.force_regenerate,
        use_ollama=args.use_ollama,
        ollama_model=args.ollama_model,
        use_hf_qwen=args.use_hf_qwen,
        hf_qwen_model=args.hf_qwen_model,
        qwen_batch_size=args.qwen_batch_size,
        qwen_max_length=args.qwen_max_length,
        use_pca=args.use_pca,
        pca_dim=args.pca_dim
    )
    print(f"Data: {data.num_nodes} nodes, {data.edge_index.size(1)} edges")
    print(f"LLM embedding shape: {data.x_llm.shape}")
    
    results = run_experiments(data, num_classes, llm_type=llm_type)
