"""
WikiCS preprocessing.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import WikiCS
from tqdm import tqdm

from src import config


def load_wikics_dataset():
    """Load WikiCS and preserve all train/val splits."""
    print("Loading WikiCS dataset...")
    dataset = WikiCS(root=os.path.join(config.DATA_DIR, "WikiCS"), is_undirected=False)
    data = dataset[0]

    data.train_mask_all = data.train_mask.clone()
    data.val_mask_all = data.val_mask.clone()
    data.test_mask_all = data.test_mask.clone() if hasattr(data, "test_mask") else None

    data.train_mask = data.train_mask_all[:, 0]
    data.val_mask = data.val_mask_all[:, 0]

    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.edge_index.size(1)}")
    print(f"  Features: {data.num_features}")
    print(f"  Classes: {dataset.num_classes}")
    print(
        "  Train/Val/Test (split 0): "
        f"{int(data.train_mask.sum())}/{int(data.val_mask.sum())}/{int(data.test_mask.sum())}"
    )

    return data, dataset


def download_wikics_texts():
    """Download WikiCS texts if needed."""
    import urllib.request

    text_dir = os.path.join(config.DATA_DIR, "WikiCS", "raw")
    os.makedirs(text_dir, exist_ok=True)

    text_path = os.path.join(text_dir, "texts.json")
    if os.path.exists(text_path):
        print(f"Text file already exists: {text_path}")
        return text_path

    url = "https://raw.githubusercontent.com/pmernyei/wiki-cs-dataset/master/dataset/docs.json"
    print(f"Downloading WikiCS texts from {url}...")
    try:
        urllib.request.urlretrieve(url, text_path)
        print(f"Downloaded to {text_path}")
        return text_path
    except Exception as e:
        print(f"Download failed: {e}")
        return None


def load_wikics_texts():
    """Load WikiCS article texts."""
    text_path = download_wikics_texts()
    if text_path is None:
        return None

    print(f"Loading texts from {text_path}...")
    with open(text_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data.get("nodes", data)
    texts = []
    if isinstance(nodes, list) and len(nodes) > 0 and "title" in nodes[0]:
        for node in tqdm(nodes, desc="Processing texts"):
            title = node.get("title", "")
            tokens = node.get("tokens", [])
            text = " ".join(tokens[:100])
            texts.append(f"{title}. {text}")
    else:
        for doc in tqdm(nodes, desc="Processing texts"):
            title = doc.get("title", "")
            abstract = doc.get("text", "")[:500]
            texts.append(f"{title}. {abstract}")

    print(f"Loaded {len(texts)} texts")
    return texts


def extract_wikics_embeddings(texts, model_name=None, batch_size=32, device=None):
    """Extract LLM embeddings for WikiCS texts."""
    if model_name is None:
        model_name = config.LLM_MODEL_NAME
    if device is None:
        device = config.DEVICE

    print(f"Loading embedding model: {model_name}")
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    print(f"Extracting embeddings for {len(texts)} texts...")
    embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting"):
            batch_texts = [t if t else "Unknown document" for t in texts[i : i + batch_size]]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(device)
            outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state[:, 0, :].cpu())

    embeddings = torch.cat(embeddings, dim=0)
    del model
    torch.cuda.empty_cache()
    print(f"Embedding shape: {embeddings.shape}")
    return embeddings


def preprocess_wikics(force_regenerate=False):
    """Preprocess WikiCS with cached text embeddings."""
    data, dataset = load_wikics_dataset()

    emb_path = os.path.join(config.EMBEDDINGS_DIR, "wikics_llm_emb.pt")
    if os.path.exists(emb_path) and not force_regenerate:
        print(f"Loading cached embeddings from {emb_path}")
        llm_embeddings = torch.load(emb_path)
    else:
        texts = load_wikics_texts()
        if texts is not None and len(texts) == data.num_nodes:
            llm_embeddings = extract_wikics_embeddings(texts)
            torch.save(llm_embeddings, emb_path)
            print(f"Saved embeddings to {emb_path}")
        else:
            print("Text count mismatch or failed. Using original features.")
            llm_embeddings = data.x.clone()

    enhanced_data = Data(
        x=data.x,
        x_llm=llm_embeddings,
        edge_index=data.edge_index,
        y=data.y,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
    )
    enhanced_data.train_mask_all = data.train_mask_all
    enhanced_data.val_mask_all = data.val_mask_all
    enhanced_data.test_mask_all = data.test_mask_all

    return enhanced_data, dataset


def get_wikics_split(data, split_idx):
    """Build a single WikiCS split from preserved split tensors."""
    if not hasattr(data, "train_mask_all") or not hasattr(data, "val_mask_all"):
        raise ValueError("WikiCS split tensors are missing. Call preprocess_wikics() first.")

    if split_idx < 0 or split_idx >= data.train_mask_all.size(1):
        raise IndexError(f"split_idx {split_idx} is out of range")

    return Data(
        x=data.x,
        x_llm=data.x_llm,
        edge_index=data.edge_index,
        y=data.y,
        train_mask=data.train_mask_all[:, split_idx],
        val_mask=data.val_mask_all[:, split_idx],
        test_mask=data.test_mask,
    )


if __name__ == "__main__":
    config.set_seed()

    print("=" * 50)
    print("Preprocessing WikiCS with Wikipedia article texts")
    print("=" * 50)

    data, dataset = preprocess_wikics()
    print("\nEnhanced data:")
    print(f"  x (original): {data.x.shape}")
    print(f"  x_llm: {data.x_llm.shape}")
    print(f"  edge_index: {data.edge_index.shape}")
