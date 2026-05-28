"""
Preprocess Cora and PubMed citation datasets with text-derived embeddings.
"""
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch_geometric.data import Data
from tqdm import tqdm

from src import config


TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


def clean_text(text):
    text = TAG_RE.sub(" ", text or "")
    text = text.replace("#R##N#", " ")
    return SPACE_RE.sub(" ", text).strip()


def extract_tagged_title(citation):
    match = re.search(r"<title>(.*?)</title>", citation or "", flags=re.IGNORECASE)
    return clean_text(match.group(1)) if match else clean_text(citation)


def make_undirected_edge_index(edges, num_nodes):
    edge_set = set()
    for src, dst in edges:
        if src == dst:
            continue
        edge_set.add((src, dst))
        edge_set.add((dst, src))
    if not edge_set:
        return torch.empty((2, 0), dtype=torch.long)
    edge_index = torch.tensor(sorted(edge_set), dtype=torch.long).t().contiguous()
    return edge_index


def build_public_split(y, num_classes, seed=42):
    generator = torch.Generator().manual_seed(seed)
    train_mask = torch.zeros(y.size(0), dtype=torch.bool)
    val_mask = torch.zeros(y.size(0), dtype=torch.bool)
    test_mask = torch.zeros(y.size(0), dtype=torch.bool)

    train_indices = []
    for cls in range(num_classes):
        cls_indices = torch.nonzero(y == cls, as_tuple=False).view(-1)
        cls_indices = cls_indices[torch.randperm(cls_indices.numel(), generator=generator)]
        train_indices.append(cls_indices[:20])

    train_indices = torch.cat(train_indices)
    train_mask[train_indices] = True

    remaining = torch.nonzero(~train_mask, as_tuple=False).view(-1)
    remaining = remaining[torch.randperm(remaining.numel(), generator=generator)]
    val_count = min(500, remaining.numel())
    test_count = min(1000, max(0, remaining.numel() - val_count))
    val_mask[remaining[:val_count]] = True
    test_mask[remaining[val_count : val_count + test_count]] = True

    return train_mask, val_mask, test_mask


def protocol_filename_to_extraction_name(filename):
    for protocol in ("http", "ftp", "file"):
        prefix = f"{protocol}:##"
        if filename.startswith(prefix):
            return f"{protocol}_##{filename[len(prefix):]}"
    return filename


def parse_cora_texts(node_ids):
    text_root = Path(config.DATA_DIR) / "Cora" / "cora_orig" / "cora_orig" / "mccallum" / "cora"
    papers_path = text_root / "papers"
    extraction_dir = text_root / "extractions"

    paper_info = {}
    with papers_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[0] not in paper_info:
                paper_info[parts[0]] = (parts[1], parts[2])

    texts = []
    text_available = []
    for paper_id in node_ids:
        filename, citation = paper_info.get(paper_id, ("", ""))
        extraction_path = extraction_dir / protocol_filename_to_extraction_name(filename)
        title = ""
        abstract = ""
        has_real_text = extraction_path.exists()

        if has_real_text:
            with extraction_path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("Title:") and not title:
                        title = clean_text(line[len("Title:") :])
                    elif line.startswith("Abstract:") and not abstract:
                        abstract = clean_text(line[len("Abstract:") :])

        if not title:
            title = extract_tagged_title(citation)

        text = clean_text(f"{title}. {abstract}")
        texts.append(text if text else f"Paper {paper_id}")
        text_available.append(bool(has_real_text and (title or abstract)))

    return texts, torch.tensor(text_available, dtype=torch.bool)


def load_cora_dataset():
    graph_dir = Path(config.DATA_DIR) / "Cora" / "graph_official" / "cora"
    content_path = graph_dir / "cora.content"
    cites_path = graph_dir / "cora.cites"

    node_ids = []
    features = []
    labels = []
    with content_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split("\t")
            if not parts:
                continue
            node_ids.append(parts[0])
            features.append([float(value) for value in parts[1:-1]])
            labels.append(parts[-1])

    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    label_names = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(label_names)}
    y = torch.tensor([label_to_idx[label] for label in labels], dtype=torch.long)

    edges = []
    with cites_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            cited, citing = parts
            if cited in node_to_idx and citing in node_to_idx:
                edges.append((node_to_idx[citing], node_to_idx[cited]))

    num_classes = len(label_names)
    train_mask, val_mask, test_mask = build_public_split(y, num_classes=num_classes)
    texts, text_available_mask = parse_cora_texts(node_ids)

    data = Data(
        x=torch.tensor(features, dtype=torch.float),
        edge_index=make_undirected_edge_index(edges, len(node_ids)),
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        text_available_mask=text_available_mask,
    )

    print("Loaded Cora citation dataset")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.edge_index.size(1)}")
    print(f"  Features: {data.x.size(1)}")
    print(f"  Classes: {num_classes}")
    print(f"  Text coverage: {int(text_available_mask.sum())}/{data.num_nodes}")
    print(f"  Train/Val/Test: {int(train_mask.sum())}/{int(val_mask.sum())}/{int(test_mask.sum())}")
    return data, texts, {"num_classes": num_classes, "labels": label_names}


def parse_pubmed_attrs(line):
    parts = line.strip().split("\t")
    paper_id = parts[0]
    label = None
    feature_values = {}
    for part in parts[1:]:
        if part.startswith("label="):
            label = int(part.split("=", 1)[1]) - 1
        elif part.startswith("w-") and "=" in part:
            key, value = part.split("=", 1)
            feature_values[key] = float(value)
    return paper_id, label, feature_values


def load_pubmed_texts():
    json_path = Path(config.DATA_DIR) / "PubMed" / "PubMed_orig" / "PubMed_orig" / "pubmed.json"
    with json_path.open("r", encoding="utf-8", errors="ignore") as f:
        records = json.load(f)

    text_by_id = {}
    for record in records:
        pmid = str(record.get("PMID"))
        title = clean_text(record.get("TI", ""))
        abstract = clean_text(record.get("AB", ""))
        text_by_id[pmid] = clean_text(f"{title}. {abstract}")
    return text_by_id


def load_pubmed_dataset():
    graph_dir = Path(config.DATA_DIR) / "PubMed" / "graph_official" / "Pubmed-Diabetes" / "data"
    node_path = graph_dir / "Pubmed-Diabetes.NODE.paper.tab"
    cites_path = graph_dir / "Pubmed-Diabetes.DIRECTED.cites.tab"

    with node_path.open("r", encoding="utf-8", errors="ignore") as f:
        header_1 = f.readline()
        header_2 = f.readline().strip().split("\t")
        feature_names = [
            field.split("numeric:", 1)[1].split(":", 1)[0]
            for field in header_2
            if field.startswith("numeric:w-")
        ]
        feature_to_idx = {feature: idx for idx, feature in enumerate(feature_names)}

        node_ids = []
        labels = []
        features = []
        for line in f:
            if not line.strip():
                continue
            paper_id, label, feature_values = parse_pubmed_attrs(line)
            row = torch.zeros(len(feature_names), dtype=torch.float)
            for key, value in feature_values.items():
                if key in feature_to_idx:
                    row[feature_to_idx[key]] = value
            node_ids.append(paper_id)
            labels.append(label)
            features.append(row)

    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    y = torch.tensor(labels, dtype=torch.long)

    edges = []
    with cites_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("DIRECTED") or line.startswith("NO_FEATURES"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            src = parts[1].replace("paper:", "")
            dst = parts[3].replace("paper:", "")
            if src in node_to_idx and dst in node_to_idx:
                edges.append((node_to_idx[src], node_to_idx[dst]))

    text_by_id = load_pubmed_texts()
    texts = []
    text_available = []
    for paper_id in node_ids:
        text = text_by_id.get(paper_id, "")
        texts.append(text if text else f"Paper {paper_id}")
        text_available.append(bool(text))

    num_classes = int(y.max().item() + 1)
    train_mask, val_mask, test_mask = build_public_split(y, num_classes=num_classes)
    text_available_mask = torch.tensor(text_available, dtype=torch.bool)

    data = Data(
        x=torch.stack(features, dim=0),
        edge_index=make_undirected_edge_index(edges, len(node_ids)),
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        text_available_mask=text_available_mask,
    )

    print("Loaded PubMed citation dataset")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.edge_index.size(1)}")
    print(f"  Features: {data.x.size(1)}")
    print(f"  Classes: {num_classes}")
    print(f"  Text coverage: {int(text_available_mask.sum())}/{data.num_nodes}")
    print(f"  Train/Val/Test: {int(train_mask.sum())}/{int(val_mask.sum())}/{int(test_mask.sum())}")
    return data, texts, {"num_classes": num_classes}


def extract_text_embeddings(texts, model_name=None, batch_size=None, device=None):
    if model_name is None:
        model_name = config.LLM_MODEL_NAME
    if batch_size is None:
        batch_size = config.LLM_BATCH_SIZE
    if device is None:
        device = config.DEVICE

    print(f"Loading embedding model: {model_name}")
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Extracting text embeddings"):
            batch_texts = [text if text else "Unknown paper" for text in texts[start : start + batch_size]]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=config.LLM_MAX_LENGTH,
                return_tensors="pt",
            ).to(device)
            outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state[:, 0, :].cpu())

    llm_embeddings = torch.cat(embeddings, dim=0)
    del model
    torch.cuda.empty_cache()
    print(f"Embedding shape: {tuple(llm_embeddings.shape)}")
    return llm_embeddings


def preprocess_citation(dataset_name, force_regenerate=False):
    dataset_name = dataset_name.lower()
    if dataset_name == "cora":
        data, texts, meta = load_cora_dataset()
        emb_name = "cora_llm_emb.pt"
    elif dataset_name == "pubmed":
        data, texts, meta = load_pubmed_dataset()
        emb_name = "pubmed_llm_emb.pt"
    else:
        raise ValueError(f"Unsupported citation dataset: {dataset_name}")

    emb_path = Path(config.EMBEDDINGS_DIR) / emb_name
    if emb_path.exists() and not force_regenerate:
        print(f"Loading cached embeddings from {emb_path}")
        llm_embeddings = torch.load(emb_path)
    else:
        llm_embeddings = extract_text_embeddings(texts)
        torch.save(llm_embeddings, emb_path)
        print(f"Saved embeddings to {emb_path}")

    data.x_llm = llm_embeddings
    return data, meta


if __name__ == "__main__":
    config.set_seed()
    for name in ("cora", "pubmed"):
        data, _ = preprocess_citation(name)
        print(f"{name}: x={tuple(data.x.shape)}, x_llm={tuple(data.x_llm.shape)}, edge_index={tuple(data.edge_index.shape)}")
