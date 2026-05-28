"""
Run a label-informed synthetic-text recovery ablation.

This script generates class-conditioned fake paper text with a local Qwen model
(HuggingFace cache or Ollama), embeds that text with the same embedding encoder
used by the citation pipeline, and replaces the LLM embeddings of sampled
recovered training nodes with synthetic-text embeddings during repair.

The experiment is intentionally an ablation/upper bound, not a main-method
setting: synthetic text is conditioned on the ground-truth class label of a
dropped training node.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import gc
import gzip
import json
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

from scripts.preprocess.preprocess_arxiv import preprocess_arxiv, sample_arxiv_subgraph
from scripts.preprocess.preprocess_arxiv import load_arxiv_titles
from scripts.preprocess.preprocess_citation import (
    extract_text_embeddings,
    load_cora_dataset,
    load_pubmed_dataset,
    preprocess_citation,
)
from scripts.utils.run_single_dataset_pilot import (
    build_corrupted_graph,
    build_model,
    build_threshold_recovery_edges,
    sample_recovered_nodes,
)
from src import config


def get_dataset(dataset, arxiv_subgraph_size):
    if dataset == "arxiv":
        data, _ = preprocess_arxiv()
        if arxiv_subgraph_size > 0:
            data = sample_arxiv_subgraph(data, num_nodes=arxiv_subgraph_size, seed=42)
            dataset_name = f"ogbn-arxiv-subgraph-{arxiv_subgraph_size}"
        else:
            dataset_name = "ogbn-arxiv-full"
        class_names = load_arxiv_class_names()
    elif dataset in {"cora", "pubmed"}:
        data, meta = preprocess_citation(dataset)
        dataset_name = dataset.upper() if dataset == "cora" else "PubMed"
        if hasattr(data, "text_available_mask"):
            data.repair_target_mask = data.text_available_mask
            data.semantic_candidate_mask = data.text_available_mask
        class_names = meta.get("labels") if dataset == "cora" else pubmed_class_names()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return data, dataset_name, class_names


def pubmed_class_names():
    return [
        "experimental diabetes mellitus",
        "type 1 diabetes mellitus",
        "type 2 diabetes mellitus",
    ]


def load_arxiv_class_names():
    readable_names = {
        "cs.na": "networking and internet architecture",
        "cs.mm": "multimedia",
        "cs.lo": "logic in computer science",
        "cs.cy": "computers and society",
        "cs.cr": "cryptography and security",
        "cs.dc": "distributed computing",
        "cs.hc": "human-computer interaction",
        "cs.ce": "computational engineering, finance, and science",
        "cs.ni": "networking and internet architecture",
        "cs.cc": "computational complexity",
        "cs.ai": "artificial intelligence",
        "cs.ma": "multiagent systems",
        "cs.gl": "general literature in computer science",
        "cs.ne": "neural and evolutionary computing",
        "cs.sc": "symbolic computation",
        "cs.ar": "computer architecture",
        "cs.cv": "computer vision and pattern recognition",
        "cs.gr": "graphics",
        "cs.et": "emerging technologies",
        "cs.sy": "systems and control",
        "cs.cg": "computational geometry",
        "cs.oh": "other computer science",
        "cs.pl": "programming languages",
        "cs.se": "software engineering",
        "cs.lg": "machine learning",
        "cs.sd": "sound",
        "cs.si": "social and information networks",
        "cs.ro": "robotics",
        "cs.it": "information theory",
        "cs.pf": "performance",
        "cs.cl": "computation and language",
        "cs.ir": "information retrieval",
        "cs.ms": "mathematical software",
        "cs.fl": "formal languages and automata theory",
        "cs.ds": "data structures and algorithms",
        "cs.os": "operating systems",
        "cs.gt": "computer science and game theory",
        "cs.db": "databases",
        "cs.dl": "digital libraries",
        "cs.dm": "discrete mathematics",
    }
    label_path = PROJECT_ROOT / "data" / "ogbn_arxiv" / "mapping" / "labelidx2arxivcategeory.csv.gz"
    names = {}
    with gzip.open(label_path, "rt", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                raw_name = parts[1].replace("arxiv ", "").replace(" ", ".").lower()
                names[int(parts[0])] = readable_names.get(raw_name, raw_name.replace(".", " "))
    return [names[i] for i in range(len(names))]


def check_ollama(base_url):
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
        return True
    except Exception as exc:
        raise RuntimeError(
            f"Ollama is not reachable at {base_url}. Start Ollama and pull the Qwen model first, "
            f"for example: `ollama pull qwen2.5:4b`. Original error: {exc}"
        ) from exc


def ollama_generate(prompt, model, base_url, temperature):
    response = requests.post(
        f"{base_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


class HfGenerator:
    def __init__(
        self,
        model_name,
        temperature,
        max_new_tokens,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        model_path = resolve_hf_model_path(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if not torch.cuda.is_available():
            self.model = self.model.to(config.DEVICE)
        self.model.eval()
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size

    def generate(self, prompt):
        messages = [
            {
                "role": "system",
                "content": "You write concise, realistic academic paper titles and abstracts.",
            },
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = f"{messages[0]['content']}\n\n{prompt}\n\nAnswer:"
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def resolve_hf_model_path(model_name):
    model_path = Path(model_name)
    if model_path.exists():
        return str(model_path)

    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    cache_name = "models--" + model_name.replace("/", "--")
    snapshots_dir = cache_root / cache_name / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(
            [path for path in snapshots_dir.iterdir() if path.is_dir()],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if snapshots:
            return str(snapshots[0])

    return model_name


def synthetic_prompt(
    dataset_name,
    class_name,
    item_idx,
    synthetic_text_style,
    target_min_words=None,
    target_max_words=None,
    ban_dataset_name=False,
):
    topic = class_name.replace("_", " ").replace(".", " ")
    if synthetic_text_style == "length_matched":
        dataset_rule = (
            f"Do not mention the benchmark or dataset name '{dataset_name}'. "
            if ban_dataset_name
            else ""
        )
        return (
            "Write a realistic academic paper record. "
            "Do not mention that it is fictional or synthetic. "
            "Use the exact format 'Title: ... Abstract: ...'. "
            f"The paper should be about the research topic: {topic}. "
            f"{dataset_rule}"
            f"Write {target_min_words}-{target_max_words} words total, including title and abstract. "
            "Use one title and a 5-7 sentence abstract with concrete motivation, method, data, findings, and limitations. "
            "Avoid generic benchmark descriptions. "
            "Do not repeat phrases or sentences.\n\n"
            f"Variant id: {item_idx}\n"
            "Title:"
        )
    return (
        "Complete the following fictional academic paper record. "
        "Do not mention that it is fictional or synthetic. "
        "Use one concise title and a two-sentence abstract.\n\n"
        f"Dataset context: {dataset_name}\n"
        f"Paper topic/class: {class_name}\n"
        f"Variant id: {item_idx}\n"
        "Title:"
    )


def fallback_synthetic_text(dataset_name, class_name, item_idx):
    topic = class_name.replace("_", " ").replace(".", " ")
    if "arxiv" in dataset_name.lower():
        return f"Title: Semantic learning methods for {topic} research"
    return (
        f"Title: Learning representations for {topic} in citation graphs. "
        f"Abstract: This paper studies methods for classifying documents related to {topic} "
        f"using text-derived semantic representations and sparse citation neighborhoods. "
        f"The study analyzes how semantic similarity and graph structure interact under "
        f"limited supervision in the {dataset_name} setting."
    )


def clean_generated_text(text, dataset_name, class_name, item_idx):
    text = " ".join((text or "").replace("\r", " ").replace("\n", " ").split())
    lower = text.lower()
    for marker in ["title:", "the title is:", "paper title:"]:
        pos = lower.find(marker)
        if pos >= 0:
            text = text[pos:]
            break
    bad_markers = [
        "write one realistic",
        "complete the following",
        "ascript",
        "dataset context",
        "variant id",
    ]
    min_words = 4 if "arxiv" in dataset_name.lower() else 18
    if len(text.split()) < min_words or any(marker in lower for marker in bad_markers):
        return fallback_synthetic_text(dataset_name, class_name, item_idx)
    return text


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def text_tokens(text):
    return TOKEN_RE.findall((text or "").lower())


def has_repeated_ngram(tokens, n=4, max_count=2):
    if len(tokens) < n:
        return False
    counts = {}
    for idx in range(len(tokens) - n + 1):
        ngram = tuple(tokens[idx : idx + n])
        counts[ngram] = counts.get(ngram, 0) + 1
        if counts[ngram] > max_count:
            return True
    return False


def lexical_jaccard(text_a, text_b):
    toks_a = set(text_tokens(text_a))
    toks_b = set(text_tokens(text_b))
    if not toks_a or not toks_b:
        return 0.0
    return len(toks_a & toks_b) / len(toks_a | toks_b)


def dataset_name_aliases(dataset_name):
    lower = dataset_name.lower()
    aliases = {lower}
    if "full" in lower:
        aliases.add(lower.replace("-full", ""))
    if "cora" in lower:
        aliases.add("cora")
    if "pubmed" in lower:
        aliases.add("pubmed")
    if "arxiv" in lower:
        aliases.update({"arxiv", "ogbn-arxiv"})
    return aliases


def assess_generated_text(
    text,
    dataset_name,
    existing_class_texts,
    min_words,
    max_words,
    ban_dataset_name,
    max_jaccard_similarity,
):
    lower = (text or "").lower()
    tokens = text_tokens(text)
    word_count = len(tokens)
    prompt_echo = any(
        marker in lower
        for marker in [
            "complete the following",
            "dataset context",
            "variant id",
            "fictional",
            "synthetic",
            "write a realistic",
        ]
    )
    repeated = has_repeated_ngram(tokens)
    dataset_leak = ban_dataset_name and any(alias in lower for alias in dataset_name_aliases(dataset_name))
    length_ok = min_words <= word_count <= max_words
    max_jaccard = max((lexical_jaccard(text, prev) for prev in existing_class_texts), default=0.0)
    diversity_ok = max_jaccard <= max_jaccard_similarity
    quality_pass = length_ok and not prompt_echo and not repeated and not dataset_leak and diversity_ok
    return {
        "word_count": word_count,
        "length_ok": length_ok,
        "prompt_echo": prompt_echo,
        "repeated_ngram": repeated,
        "dataset_name_leak": dataset_leak,
        "max_class_jaccard": max_jaccard,
        "diversity_ok": diversity_ok,
        "quality_pass": quality_pass,
    }


def real_text_length_targets(dataset, class_names):
    if dataset == "cora":
        data, texts, _ = load_cora_dataset()
    elif dataset == "pubmed":
        data, texts, _ = load_pubmed_dataset()
    elif dataset == "arxiv":
        data, _ = preprocess_arxiv()
        texts = load_arxiv_titles(data.num_nodes)
        if texts is None:
            return None
    else:
        return None

    train_nodes = torch.nonzero(data.train_mask, as_tuple=False).view(-1).tolist()
    text_available = getattr(data, "text_available_mask", torch.ones(data.num_nodes, dtype=torch.bool))
    targets = {}
    min_real_words = 4 if dataset == "arxiv" else 30
    for label in range(len(class_names)):
        lengths = sorted(
            len(text_tokens(texts[idx]))
            for idx in train_nodes
            if int(data.y[idx].item()) == label and bool(text_available[idx]) and len(text_tokens(texts[idx])) >= min_real_words
        )
        if lengths:
            targets[label] = lengths
    return targets


def target_word_range(dataset, length_targets, label, item_idx, texts_per_class, synthetic_text_style):
    if synthetic_text_style != "length_matched" or not length_targets or label not in length_targets:
        return 18, 90, None
    lengths = length_targets[label]
    if texts_per_class <= 1:
        pos = len(lengths) // 2
    else:
        quantile = (item_idx + 0.5) / texts_per_class
        pos = min(len(lengths) - 1, max(0, int(round(quantile * (len(lengths) - 1)))))
    target = int(lengths[pos])
    if dataset == "arxiv":
        min_words = max(4, int(target * 0.85))
        max_words = min(32, max(min_words + 2, int(target * 1.15)))
    else:
        min_words = max(70, int(target * 0.85))
        max_words = min(config.LLM_MAX_LENGTH, max(min_words + 10, int(target * 1.15)))
    return min_words, max_words, target


def synthetic_prompt_format(dataset, synthetic_text_style):
    if synthetic_text_style != "length_matched":
        return "title_abstract"
    return "title_only" if dataset == "arxiv" else "title_abstract"


def generate_synthetic_texts(
    dataset,
    dataset_name,
    class_names,
    texts_per_class,
    generation_backend,
    ollama_model,
    ollama_base_url,
    hf_generation_model,
    temperature,
    max_new_tokens,
    top_p,
    repetition_penalty,
    no_repeat_ngram_size,
    synthetic_text_style,
    ban_dataset_name,
    max_jaccard_similarity,
    max_generation_attempts,
):
    if generation_backend == "ollama":
        check_ollama(ollama_base_url)
        generator = None
    elif generation_backend == "hf":
        generator = HfGenerator(
            hf_generation_model,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
    else:
        raise ValueError(f"Unknown generation backend: {generation_backend}")

    length_targets = real_text_length_targets(dataset, class_names)
    prompt_format = synthetic_prompt_format(dataset, synthetic_text_style)
    rows = []
    for label, class_name in enumerate(class_names):
        accepted_class_texts = []
        for item_idx in range(texts_per_class):
            min_words, max_words, target_words = target_word_range(
                dataset,
                length_targets,
                label,
                item_idx,
                texts_per_class,
                synthetic_text_style,
            )
            best_text = ""
            best_quality = None
            for attempt in range(max_generation_attempts):
                prompt = synthetic_prompt(
                    dataset_name=dataset_name,
                    class_name=class_name,
                    item_idx=f"{item_idx}-{attempt}",
                    synthetic_text_style=synthetic_text_style,
                    target_min_words=min_words,
                    target_max_words=max_words,
                    ban_dataset_name=ban_dataset_name,
                )
                if prompt_format == "title_only":
                    dataset_rule = (
                        f"Do not mention the benchmark or dataset name '{dataset_name}'. "
                        if ban_dataset_name
                        else ""
                    )
                    prompt = (
                        "Write one realistic academic paper title. "
                        "Do not mention that it is fictional or synthetic. "
                        f"The paper should be about the research topic: {class_name.replace('_', ' ').replace('.', ' ')}. "
                        f"{dataset_rule}"
                        f"Write {min_words}-{max_words} words total. "
                        "Use the exact format 'Title: ...'. "
                        "Do not repeat phrases. "
                        f"Variant id: {item_idx}-{attempt}\n"
                        "Title:"
                    )
                if generation_backend == "ollama":
                    text = ollama_generate(prompt, ollama_model, ollama_base_url, temperature)
                else:
                    text = generator.generate(prompt)
                text = clean_generated_text(text, dataset_name, class_name, item_idx)
                quality = assess_generated_text(
                    text=text,
                    dataset_name=dataset_name,
                    existing_class_texts=accepted_class_texts,
                    min_words=min_words,
                    max_words=max_words,
                    ban_dataset_name=ban_dataset_name,
                    max_jaccard_similarity=max_jaccard_similarity,
                )
                best_text = text
                best_quality = quality
                if quality["quality_pass"]:
                    break
            accepted_class_texts.append(best_text)
            rows.append(
                {
                    "label": label,
                    "class_name": class_name,
                    "item_idx": item_idx,
                    "text": best_text,
                    "synthetic_text_style": synthetic_text_style,
                    "prompt_format": prompt_format,
                    "target_word_count": target_words,
                    "min_words": min_words,
                    "max_words": max_words,
                    **best_quality,
                }
            )
            print(
                f"generated label={label} item={item_idx}: words={best_quality['word_count']} "
                f"pass={best_quality['quality_pass']} {best_text[:90].replace(chr(10), ' ')}"
            )
    return rows


def synthetic_artifact_paths(dataset, generation_backend, generation_model, texts_per_class, synthetic_text_style):
    safe_model = generation_model.replace("/", "_").replace(":", "_").replace(".", "_")
    suffix = "" if synthetic_text_style == "short" else f"_{synthetic_text_style}"
    base = PROJECT_ROOT / "embeddings" / f"{dataset}_synthetic_text_{generation_backend}_{safe_model}_n{texts_per_class}{suffix}"
    return base.with_suffix(".json"), base.with_suffix(".pt")


def load_or_create_synthetic_embeddings(
    dataset,
    dataset_name,
    class_names,
    texts_per_class,
    generation_backend,
    ollama_model,
    ollama_base_url,
    hf_generation_model,
    temperature,
    max_new_tokens,
    top_p,
    repetition_penalty,
    no_repeat_ngram_size,
    synthetic_text_style,
    ban_dataset_name,
    max_jaccard_similarity,
    max_generation_attempts,
    force_regenerate,
):
    generation_model = ollama_model if generation_backend == "ollama" else hf_generation_model
    text_path, emb_path = synthetic_artifact_paths(
        dataset,
        generation_backend,
        generation_model,
        texts_per_class,
        synthetic_text_style,
    )

    if text_path.exists() and not force_regenerate:
        rows = json.loads(text_path.read_text(encoding="utf-8"))
    else:
        rows = generate_synthetic_texts(
            dataset=dataset,
            dataset_name=dataset_name,
            class_names=class_names,
            texts_per_class=texts_per_class,
            generation_backend=generation_backend,
            ollama_model=ollama_model,
            ollama_base_url=ollama_base_url,
            hf_generation_model=hf_generation_model,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            synthetic_text_style=synthetic_text_style,
            ban_dataset_name=ban_dataset_name,
            max_jaccard_similarity=max_jaccard_similarity,
            max_generation_attempts=max_generation_attempts,
        )
        text_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved synthetic texts to {text_path}")

    if emb_path.exists() and not force_regenerate:
        synthetic_embeddings = torch.load(emb_path)
    else:
        synthetic_embeddings = extract_text_embeddings([row["text"] for row in rows])
        torch.save(synthetic_embeddings, emb_path)
        print(f"Saved synthetic embeddings to {emb_path}")

    by_label = {}
    for row_idx, row in enumerate(rows):
        by_label.setdefault(int(row["label"]), []).append(row_idx)
    return synthetic_embeddings.float(), by_label, text_path, emb_path, rows


def replace_recovered_embeddings_with_synthetic(x_llm, y, recovered_mask, synthetic_embeddings, synthetic_by_label, seed):
    x_synth = x_llm.detach().cpu().clone()
    generator = torch.Generator().manual_seed(seed)
    recovered_nodes = torch.nonzero(recovered_mask, as_tuple=False).view(-1).cpu()
    for node in recovered_nodes.tolist():
        label = int(y[node].item())
        choices = synthetic_by_label.get(label)
        if not choices:
            continue
        choice_pos = int(torch.randint(len(choices), (1,), generator=generator).item())
        x_synth[node] = synthetic_embeddings[choices[choice_pos]]
    return x_synth


def train_sparse_baseline_once(
    data,
    model_type,
    use_llm,
    seed,
    drop_rate,
    node_drop_rate,
    num_epochs,
    k_neighbors,
):
    config.set_seed(seed)
    sparse_edge, dropped_node_mask = build_corrupted_graph(
        data,
        edge_drop_rate=drop_rate,
        node_drop_rate=node_drop_rate,
    )
    train_mask = data.train_mask & ~dropped_node_mask
    in_dim = data.x_llm.size(1) if use_llm else data.x.size(1)
    num_classes = int(data.y.max().item() + 1)
    model = build_model(model_type, in_dim, num_classes, k_neighbors, config.DEFAULT_BETA).to(config.DEVICE)

    x = (data.x_llm if use_llm else data.x).to(config.DEVICE)
    edge_index = sparse_edge.to(config.DEVICE)
    y = data.y.to(config.DEVICE)
    train_mask = train_mask.to(config.DEVICE)
    val_mask = data.val_mask.to(config.DEVICE)
    test_mask = data.test_mask.to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    best_val = 0.0
    best_test = 0.0
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            model.eval()
            with torch.no_grad():
                out = model(x, edge_index)
                pred = out.argmax(dim=1)
                val_acc = (pred[val_mask] == y[val_mask]).float().mean().item() if val_mask.any() else 0.0
                test_acc = (pred[test_mask] == y[test_mask]).float().mean().item() if test_mask.any() else 0.0
                if val_acc > best_val:
                    best_val = val_acc
                    best_test = test_acc

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return {
        "accuracy": best_test,
        "sparse_edges": int(sparse_edge.size(1)),
        "dropped_nodes": int(dropped_node_mask.sum().item()),
        "observed_train_nodes": int(train_mask.sum().item()),
        "sampled_recovered_nodes": 0,
        "successful_recovered_nodes": 0,
        "train_nodes": int(train_mask.sum().item()),
        "recovery_edges": 0,
    }


def train_ours_once(
    data,
    synthetic_embeddings,
    synthetic_by_label,
    synthetic_mode,
    seed,
    drop_rate,
    node_drop_rate,
    num_epochs,
    k_neighbors,
    recovery_ratio,
    similarity_threshold,
    max_edges_per_recovered_node,
    repair_policy,
    adaptive_threshold_alpha,
):
    sparse_edge, dropped_node_mask = build_corrupted_graph(
        data,
        edge_drop_rate=drop_rate,
        node_drop_rate=node_drop_rate,
    )
    candidate_node_mask = ~dropped_node_mask
    if hasattr(data, "semantic_candidate_mask"):
        candidate_node_mask = candidate_node_mask & data.semantic_candidate_mask

    observed_train_mask = data.train_mask & ~dropped_node_mask
    sampled_recovered_mask = sample_recovered_nodes(data, dropped_node_mask, recovery_ratio)
    x_llm_for_repair = data.x_llm.detach().cpu()
    if synthetic_mode == "label_text":
        x_llm_for_repair = replace_recovered_embeddings_with_synthetic(
            x_llm=x_llm_for_repair,
            y=data.y.detach().cpu(),
            recovered_mask=sampled_recovered_mask,
            synthetic_embeddings=synthetic_embeddings,
            synthetic_by_label=synthetic_by_label,
            seed=seed,
        )
    elif synthetic_mode != "true_text":
        raise ValueError(f"Unknown synthetic mode: {synthetic_mode}")

    model_edge_index, successful_recovered_mask, added_recovery_edges = build_threshold_recovery_edges(
        x_llm=x_llm_for_repair,
        sparse_edge_index=sparse_edge,
        recovered_node_mask=sampled_recovered_mask,
        candidate_node_mask=candidate_node_mask,
        k_neighbors=k_neighbors,
        similarity_threshold=similarity_threshold,
        max_edges_per_node=max_edges_per_recovered_node,
        repair_policy=repair_policy,
        adaptive_threshold_alpha=adaptive_threshold_alpha,
        observed_node_count=int((~dropped_node_mask).sum().item()),
    )
    train_mask = observed_train_mask | (successful_recovered_mask & data.train_mask)

    device = config.DEVICE
    model = build_model("LLM_GNN", data.x_llm.size(1), int(data.y.max().item() + 1), k_neighbors, config.DEFAULT_BETA).to(device)
    gcn_edge_index, _ = add_self_loops(model_edge_index.to(device), num_nodes=data.num_nodes)
    x = x_llm_for_repair.to(device)
    y = data.y.to(device)
    train_mask = train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    best_val = 0.0
    best_test = 0.0
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model.gnn(x, gcn_edge_index)
        loss = F.cross_entropy(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            model.eval()
            with torch.no_grad():
                out = model.gnn(x, gcn_edge_index)
                pred = out.argmax(dim=1)
                val_acc = (pred[val_mask] == y[val_mask]).float().mean().item() if val_mask.any() else 0.0
                test_acc = (pred[test_mask] == y[test_mask]).float().mean().item() if test_mask.any() else 0.0
                if val_acc > best_val:
                    best_val = val_acc
                    best_test = test_acc

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return {
        "accuracy": best_test,
        "sparse_edges": int(sparse_edge.size(1)),
        "dropped_nodes": int(dropped_node_mask.sum().item()),
        "observed_train_nodes": int(observed_train_mask.sum().item()),
        "sampled_recovered_nodes": int(sampled_recovered_mask.sum().item()),
        "successful_recovered_nodes": int(successful_recovered_mask.sum().item()),
        "train_nodes": int(train_mask.sum().item()),
        "recovery_edges": int(added_recovery_edges),
    }


def main():
    parser = argparse.ArgumentParser(description="Run synthetic-text recovered-node ablation")
    parser.add_argument("--dataset", type=str, default="cora", choices=["cora", "pubmed", "arxiv"])
    parser.add_argument("--node-drop-rates", type=str, default="0.75,0.85,0.95")
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--num-epochs", type=int, default=300)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--k-neighbors", type=int, default=10)
    parser.add_argument("--recovery-ratio", type=float, default=0.5)
    parser.add_argument("--similarity-threshold", type=float, default=0.6)
    parser.add_argument("--max-edges-per-recovered-node", type=int, default=10)
    parser.add_argument("--repair-policy", type=str, default="adaptive", choices=["fixed", "adaptive"])
    parser.add_argument("--adaptive-threshold-alpha", type=float, default=0.0)
    parser.add_argument("--texts-per-class", type=int, default=12)
    parser.add_argument("--generation-backend", type=str, default="hf", choices=["hf", "ollama"])
    parser.add_argument("--hf-generation-model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--ollama-model", type=str, default="qwen2.5:4b")
    parser.add_argument("--ollama-base-url", type=str, default="http://localhost:11434")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--synthetic-text-style", type=str, default="length_matched", choices=["short", "length_matched"])
    parser.add_argument("--ban-dataset-name", action="store_true")
    parser.add_argument("--max-jaccard-similarity", type=float, default=0.72)
    parser.add_argument("--max-generation-attempts", type=int, default=3)
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--include-true-text-control", action="store_true")
    parser.add_argument("--include-sparse-baselines", action="store_true")
    parser.add_argument("--arxiv-subgraph-size", type=int, default=0)
    args = parser.parse_args()

    node_drop_rates = [float(value) for value in args.node_drop_rates.split(",") if value.strip()]
    config.set_seed()
    data, dataset_name, class_names = get_dataset(args.dataset, args.arxiv_subgraph_size)
    synthetic_embeddings, synthetic_by_label, text_path, emb_path, synthetic_rows = load_or_create_synthetic_embeddings(
        dataset=args.dataset,
        dataset_name=dataset_name,
        class_names=class_names,
        texts_per_class=args.texts_per_class,
        generation_backend=args.generation_backend,
        ollama_model=args.ollama_model,
        ollama_base_url=args.ollama_base_url,
        hf_generation_model=args.hf_generation_model,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        synthetic_text_style=args.synthetic_text_style,
        ban_dataset_name=args.ban_dataset_name,
        max_jaccard_similarity=args.max_jaccard_similarity,
        max_generation_attempts=args.max_generation_attempts,
        force_regenerate=args.force_regenerate,
    )
    synthetic_quality = pd.DataFrame(synthetic_rows)
    quality_summary = {}
    if not synthetic_quality.empty and "word_count" in synthetic_quality.columns:
        quality_summary = {
            "synthetic_word_count_mean": float(synthetic_quality["word_count"].mean()),
            "synthetic_word_count_std": float(synthetic_quality["word_count"].std(ddof=0)),
            "synthetic_quality_pass_rate": float(synthetic_quality.get("quality_pass", pd.Series(dtype=float)).mean()),
            "synthetic_length_ok_rate": float(synthetic_quality.get("length_ok", pd.Series(dtype=float)).mean()),
            "synthetic_dataset_leak_rate": float(synthetic_quality.get("dataset_name_leak", pd.Series(dtype=float)).mean()),
            "synthetic_repeated_ngram_rate": float(synthetic_quality.get("repeated_ngram", pd.Series(dtype=float)).mean()),
            "synthetic_max_jaccard_mean": float(synthetic_quality.get("max_class_jaccard", pd.Series(dtype=float)).mean()),
        }

    modes = ["label_text"]
    if args.include_true_text_control:
        modes.insert(0, "true_text")
    baseline_modes = [
        ("MLP (Raw)", "MLP", False),
        ("MLP (LLM)", "MLP", True),
        ("GCN (Raw, Sparse)", "GCN", False),
        ("GCN (LLM, Sparse)", "GCN", True),
        ("GAT (Raw, Sparse)", "GAT", False),
        ("GAT (LLM, Sparse)", "GAT", True),
    ]

    rows = []
    raw_rows = []
    print("=" * 80)
    print(f"Synthetic-text recovery ablation on {dataset_name}")
    print(f"Synthetic texts: {text_path}")
    print(f"Synthetic embeddings: {emb_path}")
    print("=" * 80)

    if args.include_sparse_baselines:
        for mode, model_type, use_llm in baseline_modes:
            for node_drop_rate in node_drop_rates:
                accs = []
                stats = []
                for seed in range(args.num_runs):
                    row = train_sparse_baseline_once(
                        data=data,
                        model_type=model_type,
                        use_llm=use_llm,
                        seed=seed,
                        drop_rate=args.drop_rate,
                        node_drop_rate=node_drop_rate,
                        num_epochs=args.num_epochs,
                        k_neighbors=args.k_neighbors,
                    )
                    accs.append(row["accuracy"])
                    stats.append(row)
                    raw_rows.append({"dataset": dataset_name, "mode": mode, "node_drop_rate": node_drop_rate, "seed": seed, **row})
                    print(
                        f"mode={mode} node_drop={node_drop_rate:.2f} seed={seed} "
                        f"acc={row['accuracy'] * 100:.2f}% observed_train={row['observed_train_nodes']}"
                    )
                rows.append(
                    {
                        "dataset": dataset_name,
                        "mode": mode,
                        "node_drop_rate": node_drop_rate,
                        "accuracy": float(np.mean(accs)),
                        "std": float(np.std(accs)),
                        "num_runs": args.num_runs,
                        "drop_rate": args.drop_rate,
                        "k_neighbors": args.k_neighbors,
                        "recovery_ratio": 0.0,
                        "repair_policy": "none",
                        "adaptive_threshold_alpha": args.adaptive_threshold_alpha,
                        "max_edges_per_recovered_node": 0,
                        "texts_per_class": args.texts_per_class,
                        "generation_backend": args.generation_backend,
                        "generation_model": args.ollama_model if args.generation_backend == "ollama" else args.hf_generation_model,
                        "synthetic_text_style": args.synthetic_text_style,
                        "ban_dataset_name": args.ban_dataset_name,
                        "max_jaccard_similarity": args.max_jaccard_similarity,
                        "max_generation_attempts": args.max_generation_attempts,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "repetition_penalty": args.repetition_penalty,
                        "no_repeat_ngram_size": args.no_repeat_ngram_size,
                        "max_new_tokens": args.max_new_tokens,
                        "synthetic_text_path": str(text_path.relative_to(PROJECT_ROOT)),
                        "synthetic_embedding_path": str(emb_path.relative_to(PROJECT_ROOT)),
                        "sparse_edges_mean": float(np.mean([item["sparse_edges"] for item in stats])),
                        "dropped_nodes_mean": float(np.mean([item["dropped_nodes"] for item in stats])),
                        "observed_train_nodes_mean": float(np.mean([item["observed_train_nodes"] for item in stats])),
                        "sampled_recovered_nodes_mean": 0.0,
                        "successful_recovered_nodes_mean": 0.0,
                        "train_nodes_mean": float(np.mean([item["train_nodes"] for item in stats])),
                        "recovery_edges_mean": 0.0,
                        **quality_summary,
                    }
                )
                print(f"mode={mode} node_drop={node_drop_rate:.2f} mean={np.mean(accs) * 100:.2f}% std={np.std(accs) * 100:.2f}%")

    for mode in modes:
        for node_drop_rate in node_drop_rates:
            accs = []
            stats = []
            for seed in range(args.num_runs):
                config.set_seed(seed)
                row = train_ours_once(
                    data=data,
                    synthetic_embeddings=synthetic_embeddings,
                    synthetic_by_label=synthetic_by_label,
                    synthetic_mode=mode,
                    seed=seed,
                    drop_rate=args.drop_rate,
                    node_drop_rate=node_drop_rate,
                    num_epochs=args.num_epochs,
                    k_neighbors=args.k_neighbors,
                    recovery_ratio=args.recovery_ratio,
                    similarity_threshold=args.similarity_threshold,
                    max_edges_per_recovered_node=args.max_edges_per_recovered_node,
                    repair_policy=args.repair_policy,
                    adaptive_threshold_alpha=args.adaptive_threshold_alpha,
                )
                accs.append(row["accuracy"])
                stats.append(row)
                raw_rows.append({"dataset": dataset_name, "mode": mode, "node_drop_rate": node_drop_rate, "seed": seed, **row})
                print(
                    f"mode={mode} node_drop={node_drop_rate:.2f} seed={seed} "
                    f"acc={row['accuracy'] * 100:.2f}% "
                    f"successful={row['successful_recovered_nodes']} recovery_edges={row['recovery_edges']}"
                )
            rows.append(
                {
                    "dataset": dataset_name,
                    "mode": mode,
                    "node_drop_rate": node_drop_rate,
                    "accuracy": float(np.mean(accs)),
                    "std": float(np.std(accs)),
                    "num_runs": args.num_runs,
                    "drop_rate": args.drop_rate,
                    "k_neighbors": args.k_neighbors,
                    "recovery_ratio": args.recovery_ratio,
                    "repair_policy": args.repair_policy,
                    "adaptive_threshold_alpha": args.adaptive_threshold_alpha,
                    "max_edges_per_recovered_node": args.max_edges_per_recovered_node,
                    "texts_per_class": args.texts_per_class,
                    "generation_backend": args.generation_backend,
                    "generation_model": args.ollama_model if args.generation_backend == "ollama" else args.hf_generation_model,
                    "synthetic_text_style": args.synthetic_text_style,
                    "ban_dataset_name": args.ban_dataset_name,
                    "max_jaccard_similarity": args.max_jaccard_similarity,
                    "max_generation_attempts": args.max_generation_attempts,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "repetition_penalty": args.repetition_penalty,
                    "no_repeat_ngram_size": args.no_repeat_ngram_size,
                    "max_new_tokens": args.max_new_tokens,
                    "synthetic_text_path": str(text_path.relative_to(PROJECT_ROOT)),
                    "synthetic_embedding_path": str(emb_path.relative_to(PROJECT_ROOT)),
                    "sparse_edges_mean": float(np.mean([item["sparse_edges"] for item in stats])),
                    "dropped_nodes_mean": float(np.mean([item["dropped_nodes"] for item in stats])),
                    "observed_train_nodes_mean": float(np.mean([item["observed_train_nodes"] for item in stats])),
                    "sampled_recovered_nodes_mean": float(np.mean([item["sampled_recovered_nodes"] for item in stats])),
                    "successful_recovered_nodes_mean": float(np.mean([item["successful_recovered_nodes"] for item in stats])),
                    "train_nodes_mean": float(np.mean([item["train_nodes"] for item in stats])),
                    "recovery_edges_mean": float(np.mean([item["recovery_edges"] for item in stats])),
                    **quality_summary,
                }
            )
            print(f"mode={mode} node_drop={node_drop_rate:.2f} mean={np.mean(accs) * 100:.2f}% std={np.std(accs) * 100:.2f}%")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = PROJECT_ROOT / "logs" / f"{args.dataset}_synthetic_text_recovery_summary_{timestamp}.csv"
    raw_path = PROJECT_ROOT / "logs" / f"{args.dataset}_synthetic_text_recovery_raw_{timestamp}.csv"
    quality_path = PROJECT_ROOT / "logs" / f"{args.dataset}_synthetic_text_recovery_quality_{timestamp}.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    pd.DataFrame(raw_rows).to_csv(raw_path, index=False)
    synthetic_quality.to_csv(quality_path, index=False)
    print("\nSummary")
    print(pd.DataFrame(rows)[["dataset", "mode", "node_drop_rate", "accuracy", "std", "successful_recovered_nodes_mean", "recovery_edges_mean"]].to_string(index=False))
    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved raw runs to: {raw_path}")
    print(f"Saved synthetic quality to: {quality_path}")


if __name__ == "__main__":
    main()
