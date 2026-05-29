"""
Cold-start node admission and pseudo-label utilities.

These functions are method components rather than experiment runners. They only
use text-derived node embeddings and observed training labels; hidden cold-start
labels are not needed except by downstream diagnostics.
"""
import numpy as np
import torch
import torch.nn.functional as F


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
    confidence = torch.zeros_like(y, dtype=torch.float)
    train_nodes = torch.nonzero(observed_train_mask, as_tuple=False).view(-1).cpu()
    selected_nodes = torch.nonzero(selected_mask, as_tuple=False).view(-1).cpu()
    if train_nodes.numel() == 0 or selected_nodes.numel() == 0:
        return pseudo_y, confidence

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
    return pseudo_y, confidence


def pseudo_label_by_class_centroid(x_llm, observed_train_mask, selected_mask, y, num_classes):
    pseudo_y = y.clone()
    confidence = torch.zeros_like(y, dtype=torch.float)
    selected_nodes = torch.nonzero(selected_mask, as_tuple=False).view(-1).cpu()
    if selected_nodes.numel() == 0:
        return pseudo_y, confidence

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
        return pseudo_y, confidence

    centroid_x = torch.stack(centroids, dim=0)
    sims = x[selected_nodes] @ centroid_x.t().contiguous()
    best_sims, best_idx = torch.max(sims, dim=1)
    for row_idx, node in enumerate(selected_nodes.tolist()):
        pseudo_y[node] = valid_labels[int(best_idx[row_idx].item())]
        confidence[node] = float((best_sims[row_idx].item() + 1.0) / 2.0)
    return pseudo_y, confidence


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
    confidence = torch.zeros_like(y, dtype=torch.float)
    active_mask = observed_train_mask | selected_mask
    active_nodes = torch.nonzero(active_mask, as_tuple=False).view(-1).cpu()
    selected_nodes = torch.nonzero(selected_mask, as_tuple=False).view(-1).cpu()
    if active_nodes.numel() == 0 or selected_nodes.numel() == 0:
        return pseudo_y, confidence

    x = normalized_cpu(x_llm)
    cluster_labels, _ = fit_kmeans(x[active_nodes], num_clusters=num_clusters, seed=seed)
    node_to_cluster = torch.full((x_llm.size(0),), -1, dtype=torch.long)
    node_to_cluster[active_nodes] = cluster_labels

    y_cpu = y.detach().cpu()
    observed_cpu = observed_train_mask.detach().cpu()
    fallback_y, fallback_conf = pseudo_label_by_class_centroid(
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
            continue
        counts = torch.bincount(y_cpu[labeled_nodes], minlength=num_classes).float()
        pseudo_y[node] = int(torch.argmax(counts).item())
        confidence[node] = float(counts.max().item() / max(labeled_nodes.numel(), 1))
    return pseudo_y, confidence


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


def safe_nanmean(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.isnan(arr).all():
        return float("nan")
    return float(np.nanmean(arr))
