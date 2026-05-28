"""
对比两种大模型 Pooling 方法：
1. Last Token Pooling (Qwen 标准做法，适合 decoder-only 模型)
2. Mean Pooling (通用做法，适合 encoder 模型)

用于分析不同 pooling 策略对嵌入质量的影响
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from src import config

# PyTorch 2.6 兼容性
import torch.serialization
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr, Data
from torch_geometric.data.storage import GlobalStorage, NodeStorage, EdgeStorage
torch.serialization.add_safe_globals([
    DataEdgeAttr, DataTensorAttr, Data,
    GlobalStorage, NodeStorage, EdgeStorage
])


def load_20news_sample(n_samples=2000):
    """加载 20 Newsgroups 数据的子集用于快速测试"""
    print(f"Loading 20 Newsgroups dataset (sample={n_samples})...")
    dataset = fetch_20newsgroups(
        subset='all',
        remove=('headers', 'footers', 'quotes'),
        shuffle=True,
        random_state=42
    )
    
    texts = dataset.data[:n_samples]
    labels = dataset.target[:n_samples]
    
    # 过滤过短文本
    valid_indices = [i for i, t in enumerate(texts) if len(t.strip()) > 10]
    texts = [texts[i] for i in valid_indices]
    labels = torch.tensor([labels[i] for i in valid_indices], dtype=torch.long)
    
    print(f"  Documents: {len(texts)}, Classes: {len(set(labels.tolist()))}")
    return texts, labels


def extract_with_last_token_pooling(texts, model, tokenizer, device, max_length=256, batch_size=4):
    """
    Last Token Pooling (Qwen 标准做法)
    - 适合 decoder-only 模型（GPT, Qwen, LLaMA）
    - 取最后一个有效 token 的隐藏状态
    - 原理：decoder 模型的最后一个 token 聚合了前面所有 token 的信息
    """
    print("Extracting with Last Token Pooling...")
    processed_texts = [t if t.strip() else "empty document" for t in texts]
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(processed_texts), batch_size), desc="Last Token Pooling"):
            batch_texts = processed_texts[i:i + batch_size]
            
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(device)
            
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
            
            # 获取每个序列最后一个有效 token（非 padding）
            seq_lengths = inputs['attention_mask'].sum(dim=1) - 1
            batch_indices = torch.arange(last_hidden_state.size(0), device=device)
            last_token_emb = last_hidden_state[batch_indices, seq_lengths]
            
            embeddings.append(last_token_emb.cpu())
    
    return torch.cat(embeddings, dim=0)


def extract_with_mean_pooling(texts, model, tokenizer, device, max_length=256, batch_size=4):
    """
    Mean Pooling (通用做法)
    - 适合 encoder 模型（BERT, RoBERTa）和部分 decoder 模型
    - 对所有有效 token 的隐藏状态取平均
    - 原理：平均池化可以捕获整个序列的全局信息
    """
    print("Extracting with Mean Pooling...")
    processed_texts = [t if t.strip() else "empty document" for t in texts]
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(processed_texts), batch_size), desc="Mean Pooling"):
            batch_texts = processed_texts[i:i + batch_size]
            
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(device)
            
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
            
            # Mean pooling: 考虑 attention mask，忽略 padding
            attention_mask = inputs['attention_mask'].unsqueeze(-1)  # [batch, seq_len, 1]
            masked_hidden = last_hidden_state * attention_mask
            sum_hidden = masked_hidden.sum(dim=1)  # [batch, hidden_dim]
            sum_mask = attention_mask.sum(dim=1).clamp(min=1e-9)  # [batch, 1]
            mean_emb = sum_hidden / sum_mask
            
            embeddings.append(mean_emb.cpu())
    
    return torch.cat(embeddings, dim=0)


def evaluate_embeddings(embeddings, labels, name):
    """评估嵌入质量"""
    print(f"\n{'='*50}")
    print(f"评估: {name}")
    print(f"{'='*50}")
    
    emb_np = embeddings.numpy()
    labels_np = labels.numpy()
    
    # 1. 基本统计
    print(f"嵌入维度: {embeddings.shape}")
    print(f"嵌入范数 (mean±std): {np.linalg.norm(emb_np, axis=1).mean():.4f} ± {np.linalg.norm(emb_np, axis=1).std():.4f}")
    
    # 2. Silhouette Score (聚类质量)
    # 采样计算以加速
    sample_size = min(1000, len(labels_np))
    indices = np.random.choice(len(labels_np), sample_size, replace=False)
    silhouette = silhouette_score(emb_np[indices], labels_np[indices])
    print(f"Silhouette Score: {silhouette:.4f} (越高越好，范围 [-1, 1])")
    
    # 3. kNN 分类准确率 (k=5)
    k = 5
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(emb_np)
    _, indices_knn = nbrs.kneighbors(emb_np)
    
    correct = 0
    for i in range(len(labels_np)):
        neighbor_labels = labels_np[indices_knn[i, 1:k+1]]  # 排除自身
        pred = np.bincount(neighbor_labels).argmax()
        if pred == labels_np[i]:
            correct += 1
    knn_acc = correct / len(labels_np)
    print(f"kNN 分类准确率 (k={k}): {knn_acc:.4f}")
    
    # 4. 类内/类间距离比
    unique_labels = np.unique(labels_np)
    intra_dists = []
    inter_dists = []
    
    for label in unique_labels[:5]:  # 只取前5个类加速
        mask = labels_np == label
        class_emb = emb_np[mask]
        if len(class_emb) > 1:
            # 类内距离
            from sklearn.metrics.pairwise import cosine_distances
            intra_dist = cosine_distances(class_emb).mean()
            intra_dists.append(intra_dist)
            
            # 类间距离（与其他类的中心）
            other_mask = labels_np != label
            other_center = emb_np[other_mask].mean(axis=0, keepdims=True)
            class_center = class_emb.mean(axis=0, keepdims=True)
            inter_dist = cosine_distances(class_center, other_center)[0, 0]
            inter_dists.append(inter_dist)
    
    if intra_dists and inter_dists:
        ratio = np.mean(inter_dists) / np.mean(intra_dists)
        print(f"类间/类内距离比: {ratio:.4f} (越高越好)")
    
    return {
        'silhouette': silhouette,
        'knn_acc': knn_acc,
        'inter_intra_ratio': ratio if intra_dists else 0
    }


def main():
    from transformers import AutoTokenizer, AutoModel
    
    config.set_seed()
    device = config.DEVICE
    print(f"Device: {device}")
    
    # 加载数据
    texts, labels = load_20news_sample(n_samples=2000)
    
    # 加载模型
    model_name = 'Qwen/Qwen2.5-1.5B'
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()
    
    # 提取嵌入
    emb_last_token = extract_with_last_token_pooling(texts, model, tokenizer, device)
    emb_mean = extract_with_mean_pooling(texts, model, tokenizer, device)
    
    # 评估
    results_last = evaluate_embeddings(emb_last_token, labels, "Last Token Pooling")
    results_mean = evaluate_embeddings(emb_mean, labels, "Mean Pooling")
    
    # 对比总结
    print("\n" + "="*60)
    print("对比总结")
    print("="*60)
    print(f"{'指标':<25} {'Last Token':<15} {'Mean Pooling':<15} {'胜者':<10}")
    print("-"*60)
    
    metrics = [
        ('Silhouette Score', 'silhouette', True),
        ('kNN 准确率', 'knn_acc', True),
        ('类间/类内距离比', 'inter_intra_ratio', True),
    ]
    
    for name, key, higher_better in metrics:
        v1, v2 = results_last[key], results_mean[key]
        if higher_better:
            winner = "Last Token" if v1 > v2 else "Mean" if v2 > v1 else "平局"
        else:
            winner = "Last Token" if v1 < v2 else "Mean" if v2 < v1 else "平局"
        print(f"{name:<25} {v1:<15.4f} {v2:<15.4f} {winner:<10}")
    
    print("\n" + "="*60)
    print("结论")
    print("="*60)
    print("""
对于 Decoder-only 模型（如 Qwen）：
- Last Token Pooling: 利用自回归特性，最后一个 token 聚合了所有前文信息
- Mean Pooling: 平均所有 token，可能稀释重要信息

一般来说：
- Decoder-only 模型推荐 Last Token Pooling
- Encoder 模型（BERT）推荐 Mean Pooling 或 CLS Token
""")
    
    # 清理
    del model, tokenizer
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
