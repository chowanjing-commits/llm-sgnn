"""
ogbn-arxiv 数据集预处理
该数据集包含真实的论文标题，适合 LLM 语义增强实验
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# 修复 PyTorch 2.6 兼容性问题
import torch.serialization
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage, NodeStorage, EdgeStorage

# 添加安全全局类
torch.serialization.add_safe_globals([
    DataEdgeAttr, DataTensorAttr, Data,
    GlobalStorage, NodeStorage, EdgeStorage
])

from ogb.nodeproppred import PygNodePropPredDataset

from src import config


def load_arxiv_dataset():
    """
    加载 ogbn-arxiv 数据集
    包含 ~170k 节点，~1.2M 边，40 个类别
    """
    print("Loading ogbn-arxiv dataset...")
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=config.DATA_DIR)
    data = dataset[0]
    
    # 获取划分
    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
    
    # 创建 mask
    num_nodes = data.num_nodes
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.y = data.y.squeeze()
    
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.edge_index.size(1)}")
    print(f"  Features: {data.num_features}")
    print(f"  Classes: {dataset.num_classes}")
    print(f"  Train/Val/Test: {train_mask.sum().item()}/{val_mask.sum().item()}/{test_mask.sum().item()}")
    
    return data, dataset


def download_with_progress(url, filepath, max_retries=3):
    """带进度条和重试的下载函数"""
    import urllib.request
    import time
    
    for attempt in range(max_retries):
        try:
            print(f"Download attempt {attempt + 1}/{max_retries}...")
            
            # 获取文件大小
            response = urllib.request.urlopen(url, timeout=30)
            total_size = int(response.headers.get('content-length', 0))
            
            # 下载并显示进度
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB chunks
            
            with open(filepath, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = downloaded * 100 / total_size
                        print(f"\rDownloading: {downloaded/(1024*1024):.1f}MB / {total_size/(1024*1024):.1f}MB ({pct:.1f}%)", end='')
            
            print()  # 换行
            
            # 验证下载完整性
            if total_size > 0 and downloaded < total_size:
                raise Exception(f"Incomplete download: {downloaded}/{total_size}")
            
            return True
            
        except Exception as e:
            print(f"\nAttempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying in 5 seconds...")
                time.sleep(5)
    
    return False


def download_arxiv_titles():
    """
    下载 ogbn-arxiv 的论文标题文件
    """
    import gzip
    import shutil
    
    mapping_dir = os.path.join(config.DATA_DIR, 'ogbn_arxiv', 'mapping')
    os.makedirs(mapping_dir, exist_ok=True)
    
    # 标题文件 URL (来自 OGB 官方)
    url = 'https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz'
    gz_path = os.path.join(mapping_dir, 'titleabs.tsv.gz')
    tsv_path = os.path.join(mapping_dir, 'titleabs.tsv')
    
    # 检查文件是否已存在且有效
    if os.path.exists(tsv_path) and os.path.getsize(tsv_path) > 100000000:  # >100MB
        print(f"Title file already exists: {tsv_path}")
        return tsv_path
    
    # 删除可能损坏的文件
    if os.path.exists(tsv_path):
        os.remove(tsv_path)
    
    # 检查 gz 文件是否完整
    if os.path.exists(gz_path) and os.path.getsize(gz_path) > 60000000:  # >60MB
        print(f"Using cached gz file: {gz_path}")
    else:
        if os.path.exists(gz_path):
            os.remove(gz_path)
        
        print(f"Downloading title file (~67MB) from {url}...")
        if not download_with_progress(url, gz_path):
            print("\n*** Download failed. Please manually download: ***")
            print(f"URL: {url}")
            print(f"Save to: {gz_path}")
            return None
    
    print("Extracting...")
    try:
        import tarfile
        
        # 尝试作为 tar.gz 解压
        if tarfile.is_tarfile(gz_path):
            print("Detected tar.gz format...")
            with tarfile.open(gz_path, 'r:gz') as tar:
                # 查找 tsv 文件
                for member in tar.getmembers():
                    if member.name.endswith('.tsv'):
                        print(f"Extracting {member.name}...")
                        # 提取并重命名
                        f = tar.extractfile(member)
                        if f:
                            with open(tsv_path, 'wb') as out:
                                out.write(f.read())
                            break
        else:
            # 普通 gzip 文件
            with gzip.open(gz_path, 'rb') as f_in:
                with open(tsv_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        print(f"Extracted to {tsv_path} (size: {os.path.getsize(tsv_path)/(1024*1024):.1f}MB)")
        return tsv_path
    except Exception as e:
        print(f"Extraction failed: {e}")
        return None


def load_arxiv_titles(num_nodes):
    """
    加载 ogbn-arxiv 的论文标题
    通过 nodeidx2paperid 映射获取每个节点对应的标题
    """
    import gzip
    
    mapping_dir = os.path.join(config.DATA_DIR, 'ogbn_arxiv', 'mapping')
    titleabs_path = os.path.join(mapping_dir, 'titleabs.tsv')
    nodeidx_path = os.path.join(mapping_dir, 'nodeidx2paperid.csv.gz')
    
    # 如果标题文件不存在，尝试下载
    if not os.path.exists(titleabs_path):
        titleabs_path = download_arxiv_titles()
        if titleabs_path is None:
            return None
    
    # 1. 加载 paper_id -> title 映射
    print(f"Loading title mapping from {titleabs_path}...")
    paperid_to_title = {}
    with open(titleabs_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading titles"):
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                paper_id = int(parts[0])
                title = parts[1]
                paperid_to_title[paper_id] = title
    
    print(f"Loaded {len(paperid_to_title)} paper titles")
    
    # 2. 加载 node_idx -> paper_id 映射
    print(f"Loading node mapping from {nodeidx_path}...")
    nodeidx_to_paperid = {}
    with gzip.open(nodeidx_path, 'rt') as f:
        next(f)  # 跳过 header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                node_idx = int(parts[0])
                paper_id = int(parts[1])
                nodeidx_to_paperid[node_idx] = paper_id
    
    # 3. 按节点顺序获取标题
    print("Mapping titles to nodes...")
    titles = []
    for i in range(num_nodes):
        paper_id = nodeidx_to_paperid.get(i)
        if paper_id is not None:
            title = paperid_to_title.get(paper_id, f"Paper {paper_id}")
        else:
            title = f"Unknown paper at index {i}"
        titles.append(title)
    
    print(f"Mapped {len(titles)} titles to nodes")
    return titles


def extract_arxiv_embeddings(titles, model_name=None, batch_size=16, device=None):
    """
    为 arxiv 标题提取 LLM 嵌入
    由于数据量大，使用较小的 batch size
    """
    if model_name is None:
        model_name = config.LLM_MODEL_NAME
    if device is None:
        device = config.DEVICE
    
    print(f"Loading embedding model: {model_name}")
    
    from transformers import AutoTokenizer, AutoModel
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    print(f"Extracting embeddings for {len(titles)} titles...")
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(titles), batch_size), desc="Extracting"):
            batch_texts = titles[i:i + batch_size]
            
            # 处理空标题
            batch_texts = [t if t else "Unknown paper" for t in batch_texts]
            
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,  # 标题通常较短
                return_tensors='pt'
            ).to(device)
            
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            embeddings.append(batch_embeddings.cpu())
    
    embeddings = torch.cat(embeddings, dim=0)
    
    # 释放显存
    del model
    torch.cuda.empty_cache()
    print(f"Embedding shape: {embeddings.shape}")
    
    return embeddings


def preprocess_arxiv(force_regenerate=False):
    """
    完整的 arxiv 预处理流程
    """
    # 加载数据集
    data, dataset = load_arxiv_dataset()
    
    # 嵌入缓存路径
    emb_path = os.path.join(config.EMBEDDINGS_DIR, 'arxiv_llm_emb.pt')
    
    if os.path.exists(emb_path) and not force_regenerate:
        print(f"Loading cached embeddings from {emb_path}")
        llm_embeddings = torch.load(emb_path)
    else:
        # 加载标题（传入节点数量用于映射）
        titles = load_arxiv_titles(data.num_nodes)
        
        if titles is not None and len(titles) == data.num_nodes:
            # 提取嵌入
            llm_embeddings = extract_arxiv_embeddings(titles)
            
            # 保存
            torch.save(llm_embeddings, emb_path)
            print(f"Saved embeddings to {emb_path}")
        else:
            print("Failed to load titles, using original features")
            llm_embeddings = data.x.clone()
    
    # 创建增强数据
    enhanced_data = Data(
        x=data.x,
        x_llm=llm_embeddings,
        edge_index=data.edge_index,
        y=data.y,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask
    )
    
    return enhanced_data, dataset


def sample_arxiv_subgraph(data, num_nodes=5000, seed=42):
    """
    对 arxiv 进行子图采样，减少计算量
    用于快速实验
    """
    np.random.seed(seed)
    
    # 随机采样节点
    all_nodes = np.arange(data.num_nodes)
    sampled_nodes = np.random.choice(all_nodes, size=min(num_nodes, len(all_nodes)), replace=False)
    sampled_nodes = torch.from_numpy(sampled_nodes).long()
    
    # 创建节点映射
    node_map = {old.item(): new for new, old in enumerate(sampled_nodes)}
    
    # 筛选边
    edge_index = data.edge_index
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[sampled_nodes] = True
    
    edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
    new_edge_index = edge_index[:, edge_mask]
    
    # 重新映射边索引
    new_edge_index = torch.stack([
        torch.tensor([node_map[idx.item()] for idx in new_edge_index[0]]),
        torch.tensor([node_map[idx.item()] for idx in new_edge_index[1]])
    ])
    
    # 采样数据
    sampled_data = Data(
        x=data.x[sampled_nodes],
        x_llm=data.x_llm[sampled_nodes] if hasattr(data, 'x_llm') else None,
        edge_index=new_edge_index,
        y=data.y[sampled_nodes],
        train_mask=data.train_mask[sampled_nodes],
        val_mask=data.val_mask[sampled_nodes],
        test_mask=data.test_mask[sampled_nodes]
    )
    
    print(f"Sampled subgraph: {sampled_data.num_nodes} nodes, {sampled_data.edge_index.size(1)} edges")
    
    return sampled_data


if __name__ == '__main__':
    config.set_seed()
    
    print("="*50)
    print("Preprocessing ogbn-arxiv with real paper titles")
    print("="*50)
    
    data, dataset = preprocess_arxiv()
    
    print(f"\nEnhanced data:")
    print(f"  x (original): {data.x.shape}")
    print(f"  x_llm: {data.x_llm.shape}")
    print(f"  edge_index: {data.edge_index.shape}")
    
    # 采样子图用于快速实验
    print("\nSampling subgraph for quick experiments...")
    sampled_data = sample_arxiv_subgraph(data, num_nodes=5000)
