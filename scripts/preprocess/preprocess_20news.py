"""
20 Newsgroups 数据集预处理
领域：新闻/话题分类 (非学术)
特点：原始无图结构，需基于语义构建图，适合展示 LLM-GNN 的建图能力
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
from torch_geometric.data import Data
from src import config

# PyTorch 2.6 兼容性
import torch.serialization
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage, NodeStorage, EdgeStorage
torch.serialization.add_safe_globals([
    DataEdgeAttr, DataTensorAttr, Data,
    GlobalStorage, NodeStorage, EdgeStorage
])


def load_20news_data():
    """
    加载 20 Newsgroups 数据
    """
    print("Loading 20 Newsgroups dataset via sklearn...")
    # 获取所有数据，移除头部、脚部和引用以聚焦内容
    dataset = fetch_20newsgroups(subset='all', 
                                remove=('headers', 'footers', 'quotes'),
                                shuffle=True, random_state=42)
    
    texts = dataset.data
    labels = dataset.target
    target_names = dataset.target_names
    
    # 过滤掉过短的文本
    valid_indices = [i for i, t in enumerate(texts) if len(t.strip()) > 10]
    texts = [texts[i] for i in valid_indices]
    labels = torch.tensor([labels[i] for i in valid_indices], dtype=torch.long)
    
    print(f"  Documents: {len(texts)}")
    print(f"  Classes: {len(target_names)}")
    print(f"  Sample classes: {target_names[:5]}")
    
    return texts, labels, len(target_names)


def extract_tfidf_features(texts, max_features=300):
    """
    提取 TF-IDF 特征作为原始特征基线
    """
    print(f"Extracting TF-IDF features (max_features={max_features})...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # 转换为 dense tensor
    features = torch.tensor(tfidf_matrix.toarray(), dtype=torch.float32)
    print(f"  TF-IDF shape: {features.shape}")
    
    return features


def extract_embeddings_hf_qwen(texts, model_name='Qwen/Qwen2.5-1.5B', batch_size=4, device=None, max_length=256, use_pca=False, pca_dim=384):
    """
    使用 HuggingFace Transformers 加载 Qwen 模型提取隐藏状态作为嵌入
    Args:
        texts: 文本列表
        model_name: HuggingFace 模型名称，默认 'Qwen/Qwen2.5-1.5B'
        batch_size: 批处理大小
        device: 设备
        max_length: tokenizer 最大长度（默认 256，用于加速且保持可控对比）
        use_pca: 是否使用 PCA 降维（默认 False）
        pca_dim: PCA 降维目标维度（默认 384，与 Sentence-BERT 一致）
    """
    from transformers import AutoTokenizer, AutoModel
    import torch
    
    if device is None:
        device = config.DEVICE
    
    print(f"Loading HuggingFace Qwen model: {model_name}")
    print(f"Device: {device}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        model.eval()
        
        print(f"Extracting embeddings for {len(texts)} documents...")
        print(f"Using Last Token Pooling (Qwen standard)")
        print(f"Tokenizer max_length: {max_length}")
        if use_pca:
            print(f"Will apply PCA: {pca_dim}D")
        
        # 处理文本，确保非空（不手动截断，让 tokenizer 处理）
        processed_texts = [t if t.strip() else "empty document" for t in texts]
        
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(processed_texts), batch_size), desc="Extracting embeddings"):
                batch_texts = processed_texts[i:i + batch_size]
                
                # Tokenize
                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                ).to(device)
                
                # Forward pass to get hidden states
                outputs = model(**inputs, output_hidden_states=True)
                
                # Extract last hidden state and perform Last Token Pooling
                # Shape: [batch_size, seq_len, hidden_dim]
                last_hidden_state = outputs.last_hidden_state
                
                # Last Token Pooling (Qwen 标准做法)
                # 获取每个序列的最后一个有效 token（非 padding）
                seq_lengths = inputs['attention_mask'].sum(dim=1) - 1  # [batch_size]
                batch_indices = torch.arange(last_hidden_state.size(0), device=device)
                last_token_embeddings = last_hidden_state[batch_indices, seq_lengths]  # [batch_size, hidden_dim]
                
                embeddings.append(last_token_embeddings.cpu())
        
        embeddings = torch.cat(embeddings, dim=0)
        print(f"Raw embedding shape: {embeddings.shape}")
        
        # 可选：PCA 降维
        if use_pca:
            from sklearn.decomposition import PCA
            print(f"Applying PCA: {embeddings.shape[1]}D -> {pca_dim}D...")
            pca = PCA(n_components=pca_dim)
            embeddings_np = embeddings.numpy()
            embeddings_reduced = pca.fit_transform(embeddings_np)
            embeddings = torch.tensor(embeddings_reduced, dtype=torch.float32)
            print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
            print(f"Final embedding shape: {embeddings.shape}")
        
        # Clean up
        del model, tokenizer
        torch.cuda.empty_cache()
        
        return embeddings
        
    except Exception as e:
        print(f"Error using HuggingFace Transformers: {e}")
        print("Please ensure transformers is installed and the model is available.")
        print(f"You can check by running: huggingface-cli download {model_name}")
        raise


def extract_embeddings_ollama(texts, model_name='qwen2.5:4b', batch_size=32, ollama_base_url='http://localhost:11434'):
    """
    使用 Ollama 本地模型提取嵌入
    Args:
        texts: 文本列表
        model_name: Ollama 模型名称，默认 'qwen2.5:4b'
        batch_size: 批处理大小
        ollama_base_url: Ollama API 基础 URL
    """
    import requests
    
    print(f"Using Ollama model: {model_name}")
    print(f"Ollama API URL: {ollama_base_url}")
    print(f"Extracting embeddings for {len(texts)} documents...")
    
    # 处理文本，确保非空
    processed_texts = [t[:1000] if t.strip() else "empty document" for t in texts]
    
    embeddings = []
    embedding_dim = None
    
    # 使用 Ollama API 获取嵌入
    # 注意：Ollama 的 embeddings API 需要模型支持
    try:
        for i in tqdm(range(0, len(processed_texts), batch_size), desc="Extracting embeddings"):
            batch_texts = processed_texts[i:i + batch_size]
            batch_embeddings = []
            
            for text in batch_texts:
                try:
                    # 使用 Ollama embeddings API
                    response = requests.post(
                        f"{ollama_base_url}/api/embeddings",
                        json={
                            "model": model_name,
                            "prompt": text
                        },
                        timeout=60
                    )
                    response.raise_for_status()
                    embedding = response.json()['embedding']
                    
                    # 记录嵌入维度（第一次获取）
                    if embedding_dim is None:
                        embedding_dim = len(embedding)
                    
                    batch_embeddings.append(embedding)
                except Exception as e:
                    print(f"Error processing text {i}: {e}")
                    # 如果失败，使用零向量作为占位符
                    if embedding_dim is None:
                        embedding_dim = 1024  # 默认维度
                    batch_embeddings.append([0.0] * embedding_dim)
            
            embeddings.extend(batch_embeddings)
        
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
        print(f"Embedding shape: {embeddings.shape}")
        return embeddings
        
    except Exception as e:
        print(f"Error using Ollama API: {e}")
        print("Please ensure Ollama is running and the model is available.")
        print("You can check by running: curl http://localhost:11434/api/tags")
        raise


def extract_embeddings(texts, model_name=None, batch_size=64, device=None, use_ollama=False, ollama_model='qwen2.5:4b', use_hf_qwen=False, hf_qwen_model='Qwen/Qwen2.5-1.5B', qwen_max_length=256, use_pca=False, pca_dim=384):
    """
    提取 LLM 嵌入
    Args:
        texts: 文本列表
        model_name: Sentence-BERT 模型名称（当 use_ollama=False 且 use_hf_qwen=False 时使用）
        batch_size: 批处理大小
        device: 设备
        use_ollama: 是否使用 Ollama（True 使用 qwen2.5:4b，False 使用 Sentence-BERT）
        ollama_model: Ollama 模型名称
        use_hf_qwen: 是否使用 HuggingFace Qwen 模型
        hf_qwen_model: HuggingFace Qwen 模型名称
        qwen_max_length: HF Qwen tokenizer 最大长度
        use_pca: 是否对 HF Qwen 使用 PCA 降维
        pca_dim: PCA 降维目标维度
    """
    if use_hf_qwen:
        return extract_embeddings_hf_qwen(texts, model_name=hf_qwen_model, batch_size=batch_size, device=device, max_length=qwen_max_length, use_pca=use_pca, pca_dim=pca_dim)
    
    if use_ollama:
        return extract_embeddings_ollama(texts, model_name=ollama_model, batch_size=batch_size)
    
    # 原有 Sentence-BERT 方式
    if model_name is None:
        model_name = config.LLM_MODEL_NAME
    if device is None:
        device = config.DEVICE
    
    print(f"Loading embedding model: {model_name}")
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer(model_name, device=str(device))
    
    print(f"Extracting embeddings for {len(texts)} documents...")
    
    # 处理文本，确保非空
    processed_texts = [t[:1000] if t.strip() else "empty document" for t in texts]
    
    embeddings = model.encode(
        processed_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True
    )
    
    embeddings = embeddings.cpu()
    del model
    torch.cuda.empty_cache()
    
    return embeddings


def build_knn_graph(embeddings, k=10):
    """
    基于嵌入构建初始 k-NN 图
    由于 20NG 没有原生图结构，我们需要初始化一个
    """
    print(f"Building k-NN graph (k={k})...")
    from sklearn.neighbors import NearestNeighbors
    
    X = embeddings.numpy()
    # 使用余弦相似度
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='cosine', n_jobs=-1).fit(X)
    _, indices = nbrs.kneighbors(X)
    
    # 构建边索引
    row = np.repeat(np.arange(X.shape[0]), k)
    col = indices[:, 1:].flatten()  # 排除自身
    
    edge_index = torch.tensor([row, col], dtype=torch.long)
    
    print(f"  Edges: {edge_index.size(1)}")
    return edge_index


def preprocess_20news(force_regenerate=False, use_ollama=False, ollama_model='qwen2.5:4b', use_hf_qwen=False, hf_qwen_model='Qwen/Qwen2.5-1.5B', qwen_batch_size=4, qwen_max_length=256, use_pca=False, pca_dim=384):
    """
    预处理流程：加载文本 -> 提取嵌入 -> 构建图 -> 保存
    Args:
        force_regenerate: 是否强制重新生成
        use_ollama: 是否使用 Ollama（True 使用 qwen2.5:4b，False 使用 Sentence-BERT）
        ollama_model: Ollama 模型名称
        use_hf_qwen: 是否使用 HuggingFace Qwen 模型
        hf_qwen_model: HuggingFace Qwen 模型名称
        qwen_batch_size: HF Qwen 抽取嵌入 batch size
        qwen_max_length: HF Qwen tokenizer 最大长度
        use_pca: 是否对 HF Qwen 使用 PCA 降维
        pca_dim: PCA 降维目标维度
    """
    # 根据使用的模型选择不同的文件名
    if use_hf_qwen:
        model_suffix = hf_qwen_model.replace('/', '_').replace('.', '_')
        pca_suffix = f'_pca{pca_dim}' if use_pca else ''
        emb_path = os.path.join(config.EMBEDDINGS_DIR, f'20news_llm_emb_hf_{model_suffix}_len{qwen_max_length}{pca_suffix}.pt')
    elif use_ollama:
        emb_path = os.path.join(config.EMBEDDINGS_DIR, f'20news_llm_emb_{ollama_model.replace(":", "_")}.pt')
    else:
        emb_path = os.path.join(config.EMBEDDINGS_DIR, '20news_llm_emb.pt')
    
    tfidf_path = os.path.join(config.EMBEDDINGS_DIR, '20news_tfidf.pt')
    
    # 20News 没有原始图结构：
    # - edge_index_raw: 基于 TF-IDF 构建的 kNN 图，作为 E_raw（用于模拟稀疏性 drop）
    # - edge_index_sem: 基于 LLM 嵌入构建的 kNN 图，作为 E_sem（用于结构增强）
    raw_graph_path = os.path.join(config.EMBEDDINGS_DIR, '20news_graph_raw_tfidf.pt')
    if use_hf_qwen:
        model_suffix = hf_qwen_model.replace('/', '_').replace('.', '_')
        pca_suffix = f'_pca{pca_dim}' if use_pca else ''
        sem_graph_path = os.path.join(config.EMBEDDINGS_DIR, f'20news_graph_sem_hf_{model_suffix}_len{qwen_max_length}{pca_suffix}.pt')
    elif use_ollama:
        sem_graph_path = os.path.join(config.EMBEDDINGS_DIR, f'20news_graph_sem_{ollama_model.replace(":", "_")}.pt')
    else:
        sem_graph_path = os.path.join(config.EMBEDDINGS_DIR, '20news_graph_sem_sentence_bert.pt')
    
    texts, labels, num_classes = load_20news_data()
    
    # 1. 获取 LLM 嵌入
    if os.path.exists(emb_path) and not force_regenerate:
        print(f"Loading cached LLM embeddings from {emb_path}")
        x_llm = torch.load(emb_path)
        if x_llm.size(0) != len(texts):
            print("Size mismatch, regenerating...")
            x_llm = extract_embeddings(texts, batch_size=qwen_batch_size, use_ollama=use_ollama, ollama_model=ollama_model, 
                                     use_hf_qwen=use_hf_qwen, hf_qwen_model=hf_qwen_model, qwen_max_length=qwen_max_length,
                                     use_pca=use_pca, pca_dim=pca_dim)
            torch.save(x_llm, emb_path)
    else:
        x_llm = extract_embeddings(texts, batch_size=qwen_batch_size, use_ollama=use_ollama, ollama_model=ollama_model,
                                 use_hf_qwen=use_hf_qwen, hf_qwen_model=hf_qwen_model, qwen_max_length=qwen_max_length,
                                 use_pca=use_pca, pca_dim=pca_dim)
        torch.save(x_llm, emb_path)
        print(f"Saved LLM embeddings to {emb_path}")
    
    # 2. 获取 TF-IDF 特征（原始特征基线）
    if os.path.exists(tfidf_path) and not force_regenerate:
        print(f"Loading cached TF-IDF features from {tfidf_path}")
        x_raw = torch.load(tfidf_path)
    else:
        x_raw = extract_tfidf_features(texts, max_features=300)
        torch.save(x_raw, tfidf_path)
        print(f"Saved TF-IDF features to {tfidf_path}")
    
    # 3. 构建 raw kNN 图（基于 TF-IDF，作为 E_raw）
    if os.path.exists(raw_graph_path) and not force_regenerate:
        print(f"Loading cached raw graph from {raw_graph_path}")
        edge_index_raw = torch.load(raw_graph_path)
    else:
        edge_index_raw = build_knn_graph(x_raw, k=10)
        torch.save(edge_index_raw, raw_graph_path)
        print(f"Saved raw graph to {raw_graph_path}")

    # 4. 构建 semantic kNN 图（基于 LLM embedding，作为 E_sem）
    if os.path.exists(sem_graph_path) and not force_regenerate:
        print(f"Loading cached semantic graph from {sem_graph_path}")
        edge_index_sem = torch.load(sem_graph_path)
    else:
        edge_index_sem = build_knn_graph(x_llm, k=10)
        torch.save(edge_index_sem, sem_graph_path)
        print(f"Saved semantic graph to {sem_graph_path}")
    
    # 5. 划分数据集 (60/20/20) - 固定种子确保可复现
    num_nodes = len(texts)
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(num_nodes, generator=generator)
    
    train_size = int(num_nodes * 0.6)
    val_size = int(num_nodes * 0.2)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size+val_size]] = True
    test_mask[indices[train_size+val_size:]] = True
    
    data = Data(
        x=x_raw,      # TF-IDF 作为原始特征
        x_llm=x_llm,  # LLM 嵌入
        edge_index=edge_index_raw,
        edge_index_raw=edge_index_raw,
        edge_index_sem=edge_index_sem,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    
    return data, num_classes


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess 20 Newsgroups dataset')
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
                       help='Use PCA to reduce HF Qwen embeddings')
    parser.add_argument('--pca-dim', type=int, default=384,
                       help='PCA target dimension (default: 384)')
    parser.add_argument('--force-regenerate', action='store_true',
                       help='Force regenerate embeddings and graph')
    
    args = parser.parse_args()
    
    config.set_seed()
    
    print("="*60)
    if args.use_hf_qwen:
        print(f"Preprocessing 20 Newsgroups (Text-to-Graph) with HuggingFace Qwen: {args.hf_qwen_model}")
    elif args.use_ollama:
        print(f"Preprocessing 20 Newsgroups (Text-to-Graph) with Ollama: {args.ollama_model}")
    else:
        print("Preprocessing 20 Newsgroups (Text-to-Graph) with Sentence-BERT")
    print("="*60)
    
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
    
    print(f"\nProcessed Data:")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.edge_index.size(1)}")
    print(f"  Raw Features (TF-IDF): {data.x.size(1)}")
    print(f"  LLM Embeddings: {data.x_llm.size(1)}")
    print(f"  Classes: {num_classes}")
