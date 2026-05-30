"""
Cold-start node admission and pseudo-label utilities.

These functions are method components rather than experiment runners. They only
use text-derived node embeddings and observed training labels; hidden cold-start
labels are not needed except by downstream diagnostics.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class ColdStartPipelineConfig:
    admission_ratio: float = 0.5
    admission_strategy: str = "cluster_representative"
    pseudo_label_strategy: str = "cluster_majority"
    pseudo_label_k: int = 5
    pseudo_label_confidence: float = 0.0
    min_pseudo_label_support: int = 0
    pseudo_label_agreement: str = "none"
    k_neighbors: int = 10
    similarity_threshold: float = 0.6
    max_edges_per_node: int = 10
    repair_policy: str = "adaptive"
    adaptive_threshold_alpha: float = 0.0


def encode_texts_with_transformer(
    texts,
    model_name=None,
    batch_size=None,
    max_length=None,
    device=None,
):
    """
    Encode raw node texts into frozen Transformer features.

    This is the generic raw-text feature generation entry for text-only
    cold-start nodes. Dataset preprocessors may still cache their own embeddings,
    but external new nodes can use this function before admission.
    """
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer

    from src import config

    if model_name is None:
        model_name = config.LLM_MODEL_NAME
    if batch_size is None:
        batch_size = config.LLM_BATCH_SIZE
    if max_length is None:
        max_length = config.LLM_MAX_LENGTH
    if device is None:
        device = config.DEVICE

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding cold-start texts"):
            batch_texts = [text if str(text).strip() else "Unknown document" for text in texts[start : start + batch_size]]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state[:, 0, :].cpu())

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not embeddings:
        return torch.empty((0, 0), dtype=torch.float)
    return torch.cat(embeddings, dim=0).float()


def normalized_cpu(x):
    return F.normalize(x.detach().cpu().float(), p=2, dim=1)


def fit_kmeans(x, num_clusters, seed):
    from sklearn.cluster import MiniBatchKMeans

    if x.size(0) == 0:
        return torch.empty(0, dtype=torch.long), torch.empty((0, x.size(1)))
    k = max(1, min(int(num_clusters), x.size(0)))
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=min(4096, max(256, x.size(0))),
        n_init="auto",
        max_iter=100,
    )
    labels = torch.tensor(kmeans.fit_predict(x.numpy()), dtype=torch.long)
    centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float)
    return labels, F.normalize(centers, p=2, dim=1)


def admission_budget(num_candidates, admission_ratio):
    if num_candidates == 0 or admission_ratio <= 0:
        return 0
    total = int(num_candidates * admission_ratio)
    if admission_ratio > 0 and total == 0:
        total = 1
    return min(total, num_candidates)


def select_random_admission(cold_start_mask, admission_ratio, seed):
    selected = torch.zeros_like(cold_start_mask)
    candidate_nodes = torch.nonzero(cold_start_mask, as_tuple=False).view(-1).cpu()
    total = admission_budget(candidate_nodes.numel(), admission_ratio)
    center_distance = torch.full((cold_start_mask.size(0),), float("nan"), dtype=torch.float)
    cluster_labels = torch.full((cold_start_mask.size(0),), -1, dtype=torch.long)
    if total == 0:
        return selected, cluster_labels, center_distance

    generator = torch.Generator().manual_seed(seed)
    keep_positions = torch.randperm(candidate_nodes.numel(), generator=generator)[:total]
    selected[candidate_nodes[keep_positions]] = True
    return selected, cluster_labels, center_distance


def select_cluster_representatives(x_llm, cold_start_mask, num_clusters, admission_ratio, seed):
    selected = torch.zeros_like(cold_start_mask)
    candidate_nodes = torch.nonzero(cold_start_mask, as_tuple=False).view(-1).cpu()
    total = admission_budget(candidate_nodes.numel(), admission_ratio)
    if total == 0:
        cluster_labels = torch.full((x_llm.size(0),), -1, dtype=torch.long)
        center_distance = torch.full((x_llm.size(0),), float("nan"), dtype=torch.float)
        return selected, cluster_labels, center_distance

    x = normalized_cpu(x_llm)
    candidate_x = x[candidate_nodes]
    cluster_labels, centers = fit_kmeans(candidate_x, num_clusters=num_clusters, seed=seed)
    sims = (candidate_x * centers[cluster_labels]).sum(dim=1)
    distances = 1.0 - sims

    selected_positions = set()
    for cluster_id in torch.unique(cluster_labels).tolist():
        positions = torch.nonzero(cluster_labels == int(cluster_id), as_tuple=False).view(-1)
        quota = admission_budget(positions.numel(), admission_ratio)
        order = positions[torch.argsort(distances[positions])]
        selected_positions.update(order[:quota].tolist())

    if len(selected_positions) < total:
        remaining = [
            pos
            for pos in torch.argsort(distances).tolist()
            if pos not in selected_positions
        ]
        selected_positions.update(remaining[: total - len(selected_positions)])
    selected_positions = sorted(selected_positions, key=lambda pos: float(distances[pos].item()))
    keep_positions = torch.tensor(selected_positions[:total], dtype=torch.long)
    selected[candidate_nodes[keep_positions]] = True

    all_cluster_labels = torch.full((x_llm.size(0),), -1, dtype=torch.long)
    all_cluster_labels[candidate_nodes] = cluster_labels
    center_distance = torch.full((x_llm.size(0),), float("nan"), dtype=torch.float)
    center_distance[candidate_nodes] = distances
    return selected, all_cluster_labels, center_distance


def select_cold_start_nodes(x_llm, cold_start_mask, num_classes, admission_ratio, strategy, seed):
    if strategy == "cluster_representative":
        return select_cluster_representatives(
            x_llm=x_llm,
            cold_start_mask=cold_start_mask,
            num_clusters=num_classes,
            admission_ratio=admission_ratio,
            seed=seed,
        )
    if strategy == "random":
        return select_random_admission(
            cold_start_mask=cold_start_mask,
            admission_ratio=admission_ratio,
            seed=seed,
        )
    raise ValueError(f"Unknown admission strategy: {strategy}")


def pseudo_label_by_nearest_labeled(x_llm, observed_train_mask, selected_mask, y, num_classes, k):
    pseudo_y = y.clone()
    confidence = torch.full_like(y, float("nan"), dtype=torch.float)
    support = torch.zeros_like(y, dtype=torch.long)
    train_nodes = torch.nonzero(observed_train_mask, as_tuple=False).view(-1).cpu()
    selected_nodes = torch.nonzero(selected_mask, as_tuple=False).view(-1).cpu()
    if train_nodes.numel() == 0 or selected_nodes.numel() == 0:
        return pseudo_y, confidence, support

    x = normalized_cpu(x_llm)
    topk = min(max(1, int(k)), train_nodes.numel())
    sims = x[selected_nodes] @ x[train_nodes].t().contiguous()
    _, nn_idx = torch.topk(sims, k=topk, dim=1, largest=True)
    train_y = y.detach().cpu()[train_nodes]
    for row_idx, node in enumerate(selected_nodes.tolist()):
        labels = train_y[nn_idx[row_idx]]
        counts = torch.bincount(labels, minlength=num_classes).float()
        pseudo_y[node] = int(torch.argmax(counts).item())
        confidence[node] = float(counts.max().item() / max(labels.numel(), 1))
        support[node] = int(counts.max().item())
    return pseudo_y, confidence, support


def pseudo_label_by_class_centroid(x_llm, observed_train_mask, selected_mask, y, num_classes):
    pseudo_y = y.clone()
    confidence = torch.full_like(y, float("nan"), dtype=torch.float)
    support = torch.zeros_like(y, dtype=torch.long)
    selected_nodes = torch.nonzero(selected_mask, as_tuple=False).view(-1).cpu()
    if selected_nodes.numel() == 0:
        return pseudo_y, confidence, support

    x = normalized_cpu(x_llm)
    centroids = []
    valid_labels = []
    y_cpu = y.detach().cpu()
    observed_cpu = observed_train_mask.detach().cpu()
    for label in range(num_classes):
        nodes = torch.nonzero(observed_cpu & (y_cpu == label), as_tuple=False).view(-1)
        if nodes.numel() == 0:
            continue
        centroids.append(F.normalize(x[nodes].mean(dim=0, keepdim=True), p=2, dim=1).squeeze(0))
        valid_labels.append(label)
    if not centroids:
        return pseudo_y, confidence, support

    centroid_x = torch.stack(centroids, dim=0)
    sims = x[selected_nodes] @ centroid_x.t().contiguous()
    best_sims, best_idx = torch.max(sims, dim=1)
    for row_idx, node in enumerate(selected_nodes.tolist()):
        pseudo_y[node] = valid_labels[int(best_idx[row_idx].item())]
        confidence[node] = float((best_sims[row_idx].item() + 1.0) / 2.0)
        support[node] = int((y_cpu[observed_cpu] == pseudo_y[node]).sum().item())
    return pseudo_y, confidence, support


def pseudo_label_by_cluster_majority(
    x_llm,
    observed_train_mask,
    selected_mask,
    y,
    num_classes,
    num_clusters,
    seed,
):
    pseudo_y = y.clone()
    confidence = torch.full_like(y, float("nan"), dtype=torch.float)
    support = torch.zeros_like(y, dtype=torch.long)
    active_mask = observed_train_mask | selected_mask
    active_nodes = torch.nonzero(active_mask, as_tuple=False).view(-1).cpu()
    selected_nodes = torch.nonzero(selected_mask, as_tuple=False).view(-1).cpu()
    if active_nodes.numel() == 0 or selected_nodes.numel() == 0:
        return pseudo_y, confidence, support

    x = normalized_cpu(x_llm)
    cluster_labels, _ = fit_kmeans(x[active_nodes], num_clusters=num_clusters, seed=seed)
    node_to_cluster = torch.full((x_llm.size(0),), -1, dtype=torch.long)
    node_to_cluster[active_nodes] = cluster_labels

    y_cpu = y.detach().cpu()
    observed_cpu = observed_train_mask.detach().cpu()
    fallback_y, fallback_conf, fallback_support = pseudo_label_by_class_centroid(
        x_llm=x_llm,
        observed_train_mask=observed_train_mask,
        selected_mask=selected_mask,
        y=y,
        num_classes=num_classes,
    )
    for node in selected_nodes.tolist():
        cluster_id = int(node_to_cluster[node].item())
        cluster_nodes = active_nodes[cluster_labels == cluster_id]
        labeled_nodes = cluster_nodes[observed_cpu[cluster_nodes]]
        if labeled_nodes.numel() == 0:
            pseudo_y[node] = fallback_y[node]
            confidence[node] = fallback_conf[node]
            support[node] = fallback_support[node]
            continue
        counts = torch.bincount(y_cpu[labeled_nodes], minlength=num_classes).float()
        pseudo_y[node] = int(torch.argmax(counts).item())
        confidence[node] = float(counts.max().item() / max(labeled_nodes.numel(), 1))
        support[node] = int(counts.max().item())
    return pseudo_y, confidence, support


def assign_pseudo_labels(
    x_llm,
    observed_train_mask,
    selected_mask,
    y,
    num_classes,
    strategy,
    pseudo_label_k,
    seed,
):
    if strategy == "cluster_majority":
        return pseudo_label_by_cluster_majority(
            x_llm=x_llm,
            observed_train_mask=observed_train_mask,
            selected_mask=selected_mask,
            y=y,
            num_classes=num_classes,
            num_clusters=num_classes,
            seed=seed,
        )
    if strategy == "nearest_labeled":
        return pseudo_label_by_nearest_labeled(
            x_llm=x_llm,
            observed_train_mask=observed_train_mask,
            selected_mask=selected_mask,
            y=y,
            num_classes=num_classes,
            k=pseudo_label_k,
        )
    if strategy == "class_centroid":
        return pseudo_label_by_class_centroid(
            x_llm=x_llm,
            observed_train_mask=observed_train_mask,
            selected_mask=selected_mask,
            y=y,
            num_classes=num_classes,
        )
    raise ValueError(f"Unknown pseudo-label strategy: {strategy}")


def build_pseudo_label_agreement_mask(
    x_llm,
    observed_train_mask,
    selected_mask,
    y,
    num_classes,
    primary_pseudo_y,
    agreement,
    pseudo_label_k,
    seed,
):
    agreement = str(agreement or "none")
    if agreement == "none":
        return torch.ones_like(selected_mask, dtype=torch.bool)

    helper_strategies = []
    if agreement == "nearest_labeled":
        helper_strategies = ["nearest_labeled"]
    elif agreement == "class_centroid":
        helper_strategies = ["class_centroid"]
    elif agreement == "nearest_or_centroid":
        helper_strategies = ["nearest_labeled", "class_centroid"]
    elif agreement == "nearest_and_centroid":
        helper_strategies = ["nearest_labeled", "class_centroid"]
    else:
        raise ValueError(f"Unknown pseudo-label agreement mode: {agreement}")

    selected_ready = selected_mask.clone()
    helper_matches = []
    for strategy in helper_strategies:
        helper_y, helper_confidence, _ = assign_pseudo_labels(
            x_llm=x_llm,
            observed_train_mask=observed_train_mask,
            selected_mask=selected_mask,
            y=y,
            num_classes=num_classes,
            strategy=strategy,
            pseudo_label_k=pseudo_label_k,
            seed=seed,
        )
        helper_valid = selected_mask & torch.isfinite(helper_confidence)
        helper_matches.append(helper_valid & (helper_y == primary_pseudo_y))

    if agreement == "nearest_or_centroid":
        agreement_mask = helper_matches[0] | helper_matches[1]
    else:
        agreement_mask = helper_matches[0]
        for match in helper_matches[1:]:
            agreement_mask = agreement_mask & match
    return selected_ready & agreement_mask


def build_semantic_recovery_edges(
    x_llm,
    sparse_edge_index,
    recovered_node_mask,
    candidate_node_mask,
    k_neighbors,
    similarity_threshold,
    max_edges_per_node,
    repair_policy="fixed",
    adaptive_threshold_alpha=0.0,
    observed_node_count=None,
):
    if not recovered_node_mask.any() or max_edges_per_node <= 0:
        return sparse_edge_index, torch.zeros_like(recovered_node_mask), 0

    recovered_nodes = torch.nonzero(recovered_node_mask, as_tuple=False).view(-1)
    candidate_nodes = torch.nonzero(candidate_node_mask, as_tuple=False).view(-1)
    if recovered_nodes.numel() == 0 or candidate_nodes.numel() == 0:
        return sparse_edge_index, torch.zeros_like(recovered_node_mask), 0

    x = normalized_cpu(x_llm)
    recovered_nodes = recovered_nodes.cpu()
    candidate_nodes = candidate_nodes.cpu()
    candidate_x = x[candidate_nodes].t().contiguous()
    added_edges = []
    successful_recovered = torch.zeros_like(recovered_node_mask)
    edge_budget = int(max_edges_per_node)
    if repair_policy == "adaptive":
        if observed_node_count is None:
            observed_nodes = torch.unique(sparse_edge_index.cpu()) if sparse_edge_index.numel() > 0 else torch.empty(0)
            observed_node_count = int(observed_nodes.numel())
        avg_observed_degree = (
            float(sparse_edge_index.size(1)) / max(int(observed_node_count), 1)
            if observed_node_count > 0
            else 0.0
        )
        edge_budget = min(int(max_edges_per_node), max(1, int(np.ceil(avg_observed_degree))))

    existing_edges = set()
    if sparse_edge_index.numel() > 0:
        existing_edges = set(zip(sparse_edge_index[0].cpu().tolist(), sparse_edge_index[1].cpu().tolist()))

    batch_size = 256
    topk = min(int(k_neighbors), candidate_nodes.numel()) if k_neighbors > 0 else candidate_nodes.numel()
    if topk <= 0:
        return sparse_edge_index, successful_recovered, 0

    for start in range(0, recovered_nodes.numel(), batch_size):
        batch_nodes = recovered_nodes[start : start + batch_size]
        batch_sims = x[batch_nodes] @ candidate_x
        if topk < candidate_nodes.numel():
            topk_sims, topk_idx = torch.topk(batch_sims, k=topk, dim=1, largest=True)
        else:
            topk_sims = batch_sims
            topk_idx = torch.arange(candidate_nodes.numel()).view(1, -1).expand(batch_sims.size(0), -1)

        for row_idx, node in enumerate(batch_nodes.tolist()):
            candidate_sims = topk_sims[row_idx]
            candidate_idx = topk_idx[row_idx]
            if repair_policy == "adaptive":
                threshold = candidate_sims.mean()
                if candidate_sims.numel() > 1:
                    threshold = threshold + adaptive_threshold_alpha * candidate_sims.std(unbiased=False)
                valid_mask = candidate_sims >= threshold
            else:
                valid_mask = candidate_sims >= similarity_threshold

            if not valid_mask.any():
                continue

            valid = candidate_idx[valid_mask]
            valid_sims = candidate_sims[valid_mask]
            order = torch.argsort(valid_sims, descending=True)
            repaired = 0
            for candidate_pos in valid[order].tolist():
                candidate = int(candidate_nodes[candidate_pos].item())
                if candidate == node:
                    continue
                src, dst = int(node), candidate
                if (src, dst) in existing_edges or (dst, src) in existing_edges:
                    continue
                added_edges.append((src, dst))
                added_edges.append((dst, src))
                existing_edges.add((src, dst))
                existing_edges.add((dst, src))
                repaired += 1
                if repaired >= edge_budget:
                    break

            if repaired > 0:
                successful_recovered[node] = True

        del batch_sims, topk_sims, topk_idx

    if not added_edges:
        return sparse_edge_index, successful_recovered, 0

    added_edge_index = torch.tensor(added_edges, dtype=torch.long).t().contiguous()
    repaired_edge_index = torch.cat([sparse_edge_index.cpu(), added_edge_index], dim=1)
    return repaired_edge_index, successful_recovered, len(added_edges) // 2


def build_cold_start_training_state(
    x_llm,
    y,
    sparse_edge_index,
    cold_start_mask,
    observed_train_mask,
    candidate_node_mask,
    num_classes,
    admission_ratio,
    admission_strategy,
    pseudo_label_strategy,
    pseudo_label_k,
    pseudo_label_confidence,
    min_pseudo_label_support,
    pseudo_label_agreement,
    k_neighbors,
    similarity_threshold,
    max_edges_per_node,
    repair_policy,
    adaptive_threshold_alpha,
    seed,
    observed_node_count=None,
):
    selected_mask, cluster_labels, center_distance = select_cold_start_nodes(
        x_llm=x_llm,
        cold_start_mask=cold_start_mask,
        num_classes=num_classes,
        admission_ratio=admission_ratio,
        strategy=admission_strategy,
        seed=seed,
    )
    pseudo_y, pseudo_confidence, pseudo_label_support = assign_pseudo_labels(
        x_llm=x_llm,
        observed_train_mask=observed_train_mask,
        selected_mask=selected_mask,
        y=y,
        num_classes=num_classes,
        strategy=pseudo_label_strategy,
        pseudo_label_k=pseudo_label_k,
        seed=seed,
    )
    pseudo_label_agreement_mask = build_pseudo_label_agreement_mask(
        x_llm=x_llm,
        observed_train_mask=observed_train_mask,
        selected_mask=selected_mask,
        y=y,
        num_classes=num_classes,
        primary_pseudo_y=pseudo_y,
        agreement=pseudo_label_agreement,
        pseudo_label_k=pseudo_label_k,
        seed=seed,
    )
    support_ready_mask = pseudo_label_support >= max(0, int(min_pseudo_label_support))
    label_ready_mask = (
        selected_mask
        & torch.isfinite(pseudo_confidence)
        & (pseudo_confidence >= pseudo_label_confidence)
        & support_ready_mask
        & pseudo_label_agreement_mask
    )
    repaired_edge_index, successful_recovered_mask, added_recovery_edges = build_semantic_recovery_edges(
        x_llm=x_llm,
        sparse_edge_index=sparse_edge_index,
        recovered_node_mask=selected_mask,
        candidate_node_mask=candidate_node_mask,
        k_neighbors=k_neighbors,
        similarity_threshold=similarity_threshold,
        max_edges_per_node=max_edges_per_node,
        repair_policy=repair_policy,
        adaptive_threshold_alpha=adaptive_threshold_alpha,
        observed_node_count=observed_node_count,
    )
    pseudo_train_mask = label_ready_mask & successful_recovered_mask
    train_mask = observed_train_mask | pseudo_train_mask
    return {
        "edge_index": repaired_edge_index,
        "pseudo_y": pseudo_y,
        "train_mask": train_mask,
        "selected_mask": selected_mask,
        "cluster_labels": cluster_labels,
        "center_distance": center_distance,
        "pseudo_confidence": pseudo_confidence,
        "pseudo_label_support": pseudo_label_support,
        "pseudo_label_agreement_mask": pseudo_label_agreement_mask,
        "label_ready_mask": label_ready_mask,
        "successful_recovered_mask": successful_recovered_mask,
        "pseudo_train_mask": pseudo_train_mask,
        "added_recovery_edges": int(added_recovery_edges),
    }


def build_cold_start_training_state_from_config(
    x_llm,
    y,
    sparse_edge_index,
    cold_start_mask,
    observed_train_mask,
    candidate_node_mask,
    num_classes,
    pipeline_config,
    seed,
    observed_node_count=None,
):
    return build_cold_start_training_state(
        x_llm=x_llm,
        y=y,
        sparse_edge_index=sparse_edge_index,
        cold_start_mask=cold_start_mask,
        observed_train_mask=observed_train_mask,
        candidate_node_mask=candidate_node_mask,
        num_classes=num_classes,
        admission_ratio=pipeline_config.admission_ratio,
        admission_strategy=pipeline_config.admission_strategy,
        pseudo_label_strategy=pipeline_config.pseudo_label_strategy,
        pseudo_label_k=pipeline_config.pseudo_label_k,
        pseudo_label_confidence=pipeline_config.pseudo_label_confidence,
        min_pseudo_label_support=pipeline_config.min_pseudo_label_support,
        pseudo_label_agreement=pipeline_config.pseudo_label_agreement,
        k_neighbors=pipeline_config.k_neighbors,
        similarity_threshold=pipeline_config.similarity_threshold,
        max_edges_per_node=pipeline_config.max_edges_per_node,
        repair_policy=pipeline_config.repair_policy,
        adaptive_threshold_alpha=pipeline_config.adaptive_threshold_alpha,
        seed=seed,
        observed_node_count=observed_node_count,
    )


def weighted_supervised_loss(
    out,
    train_y,
    train_mask,
    pseudo_train_mask=None,
    pseudo_label_loss_weight=1.0,
):
    loss_values = F.cross_entropy(out[train_mask], train_y[train_mask], reduction="none")
    weights = torch.ones_like(loss_values)
    if pseudo_train_mask is not None and pseudo_label_loss_weight != 1.0:
        pseudo_train_mask = pseudo_train_mask.to(device=train_mask.device)
        weights[pseudo_train_mask[train_mask]] = max(0.0, float(pseudo_label_loss_weight))
    if weights.sum().item() <= 0:
        raise RuntimeError("No positive-weight training nodes remain after pseudo-label loss weighting.")
    return (loss_values * weights).sum() / weights.sum()


def safe_nanmean(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.isnan(arr).all():
        return float("nan")
    return float(np.nanmean(arr))
