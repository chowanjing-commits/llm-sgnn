"""
Model definitions for LLM-Augmented GNN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.utils import add_self_loops


class MLP(nn.Module):
    """Baseline MLP that ignores graph structure."""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels))
        self.layers.append(nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index=None):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layers[-1](x)


class GCN(nn.Module):
    """Standard multi-layer GCN."""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)

    def get_embeddings(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        return x


class GAT(nn.Module):
    """Standard multi-layer GAT."""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, heads=4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
            )
        self.convs.append(
            GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)

    def get_embeddings(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
        return x


class GraphSAGE(nn.Module):
    """Standard multi-layer GraphSAGE."""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)

    def get_embeddings(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        return x


class StructureLearner(nn.Module):
    """
    Repair missing topology with semantic neighbors.

    The learner does not merge a full semantic graph anymore. It only repairs
    nodes whose undirected degree dropped after edge deletion, and only adds
    semantic edges that are not already present in the sparse graph.
    """

    def __init__(self, k_neighbors=5, beta=0.5):
        super().__init__()
        self.k = k_neighbors
        self.repair_budget = float(beta)
        self.last_repaired_node_mask = None

    def compute_cosine_similarity(self, features):
        features_norm = F.normalize(features, p=2, dim=1)
        return torch.mm(features_norm, features_norm.t())

    def build_knn_candidates(self, features, device="cpu"):
        """
        Return top-k semantic neighbor indices for every node.
        """
        features = features.to("cpu")
        n_nodes = features.size(0)
        if n_nodes > 50000:
            return self._build_knn_candidates_sklearn(features)

        features = features.to(device)
        similarity = self.compute_cosine_similarity(features)
        similarity.fill_diagonal_(-float("inf"))
        _, indices = similarity.topk(self.k, dim=1)
        return indices.cpu()

    def _build_knn_candidates_sklearn(self, features):
        from sklearn.neighbors import NearestNeighbors

        features_np = features.numpy()
        nn = NearestNeighbors(n_neighbors=self.k + 1, algorithm="auto", metric="cosine", n_jobs=-1)
        nn.fit(features_np)
        _, indices = nn.kneighbors(features_np)
        return torch.from_numpy(indices[:, 1:]).long()

    def _coalesce_undirected_edges(self, edge_index, device):
        if edge_index is None or edge_index.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        edge_index = edge_index.to(device)
        row = edge_index[0].long()
        col = edge_index[1].long()
        mask = row != col
        row = row[mask]
        col = col[mask]
        if row.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        low = torch.minimum(row, col)
        high = torch.maximum(row, col)
        return torch.unique(torch.stack([low, high], dim=0), dim=1)

    def _expand_undirected_edges(self, undirected_edge_index, device):
        if undirected_edge_index.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        return torch.cat([undirected_edge_index, undirected_edge_index.flip(0)], dim=1)

    def _compute_undirected_degree(self, undirected_edge_index, num_nodes, device):
        degree = torch.zeros(num_nodes, dtype=torch.long, device=device)
        if undirected_edge_index.numel() == 0:
            return degree

        row, col = undirected_edge_index
        ones_row = torch.ones_like(row, dtype=torch.long, device=device)
        ones_col = torch.ones_like(col, dtype=torch.long, device=device)
        degree.scatter_add_(0, row, ones_row)
        degree.scatter_add_(0, col, ones_col)
        return degree

    def _build_sparse_adjacency(self, undirected_edge_index, num_nodes):
        adjacency = [set() for _ in range(num_nodes)]
        if undirected_edge_index.numel() == 0:
            return adjacency

        rows = undirected_edge_index[0].cpu().tolist()
        cols = undirected_edge_index[1].cpu().tolist()
        for src, dst in zip(rows, cols):
            adjacency[src].add(dst)
            adjacency[dst].add(src)
        return adjacency

    def _candidate_lists_from_edge_index(self, semantic_edge_index, num_nodes):
        candidates = [[] for _ in range(num_nodes)]
        seen = [set() for _ in range(num_nodes)]
        if semantic_edge_index is None or semantic_edge_index.numel() == 0:
            return candidates

        rows = semantic_edge_index[0].cpu().tolist()
        cols = semantic_edge_index[1].cpu().tolist()
        for src, dst in zip(rows, cols):
            if src == dst or dst in seen[src]:
                continue
            candidates[src].append(dst)
            seen[src].add(dst)
        return candidates

    def _candidate_lists_from_features(self, features, damaged_nodes=None, candidate_node_mask=None, device="cpu"):
        if damaged_nodes is None:
            return self.build_knn_candidates(features, device=device).cpu().tolist()

        damaged_nodes = [int(node) for node in damaged_nodes]
        candidates = [[] for _ in range(features.size(0))]
        if not damaged_nodes:
            return candidates

        features_cpu = features.detach().to("cpu")
        damaged_tensor = torch.tensor(damaged_nodes, dtype=torch.long)
        query_features = features_cpu[damaged_tensor]

        from sklearn.neighbors import NearestNeighbors

        n_neighbors = min(max(self.k * 20, self.k + 1), features_cpu.size(0))
        nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto", metric="cosine", n_jobs=-1)
        nn.fit(features_cpu.numpy())
        _, indices = nn.kneighbors(query_features.numpy())

        for node, node_candidates in zip(damaged_nodes, indices):
            candidates[node] = [int(candidate) for candidate in node_candidates if int(candidate) != node]
        return candidates

    def repair_graph(
        self,
        features,
        sparse_edge_index,
        original_edge_index,
        num_nodes,
        device="cpu",
        semantic_edge_index=None,
        repair_node_mask=None,
        candidate_node_mask=None,
    ):
        if original_edge_index is None:
            original_edge_index = sparse_edge_index

        sparse_undirected = self._coalesce_undirected_edges(sparse_edge_index, device)
        original_undirected = self._coalesce_undirected_edges(original_edge_index, device)

        degree_sparse = self._compute_undirected_degree(sparse_undirected, num_nodes, device)
        degree_original = self._compute_undirected_degree(original_undirected, num_nodes, device)
        degree_drop = torch.clamp(degree_original - degree_sparse, min=0)

        repair_budget = max(0.0, min(1.0, self.repair_budget))
        if repair_budget <= 0.0 or degree_drop.sum().item() == 0:
            self.last_repaired_node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            return self._expand_undirected_edges(sparse_undirected, device)

        sparse_adjacency = self._build_sparse_adjacency(sparse_undirected, num_nodes)
        added_edges = []
        repair_completed = torch.zeros(num_nodes, dtype=torch.bool, device=device)

        if repair_node_mask is not None:
            repair_node_mask = repair_node_mask.to(device).bool()
            degree_drop = degree_drop * repair_node_mask.long()

        damaged_nodes = torch.nonzero(degree_drop > 0, as_tuple=False).view(-1).cpu().tolist()
        candidate_node_mask = (
            candidate_node_mask.to(device).bool().clone()
            if candidate_node_mask is not None
            else torch.ones(num_nodes, dtype=torch.bool, device=device)
        )
        if semantic_edge_index is not None:
            candidate_lists = self._candidate_lists_from_edge_index(semantic_edge_index.to(device), num_nodes)
        else:
            candidate_lists = self._candidate_lists_from_features(
                features,
                damaged_nodes=damaged_nodes,
                candidate_node_mask=candidate_node_mask,
                device=device,
            )

        for node in damaged_nodes:
            missing_degree = int(degree_drop[node].item())
            if missing_degree <= 0:
                continue

            repair_quota = max(0, min(missing_degree, int((missing_degree * repair_budget) + 0.999999)))
            if repair_quota == 0:
                continue

            repaired = 0
            for candidate in candidate_lists[node]:
                if (
                    candidate == node
                    or candidate in sparse_adjacency[node]
                    or not candidate_node_mask[candidate].item()
                ):
                    continue
                sparse_adjacency[node].add(candidate)
                sparse_adjacency[candidate].add(node)
                added_edges.append((min(node, candidate), max(node, candidate)))
                repaired += 1
                if repaired >= repair_quota:
                    break
            if repaired > 0:
                repair_completed[node] = True
                candidate_node_mask[node] = True

        if not added_edges:
            self.last_repaired_node_mask = repair_completed
            return self._expand_undirected_edges(sparse_undirected, device)

        added_edge_index = torch.tensor(added_edges, dtype=torch.long, device=device).t().contiguous()
        repaired_undirected = torch.cat([sparse_undirected, added_edge_index], dim=1)
        repaired_undirected = torch.unique(repaired_undirected, dim=1)
        self.last_repaired_node_mask = repair_completed
        return self._expand_undirected_edges(repaired_undirected, device)

    def forward(
        self,
        features,
        sparse_edge_index,
        num_nodes,
        device="cpu",
        semantic_edge_index=None,
        original_edge_index=None,
        repair_node_mask=None,
        candidate_node_mask=None,
    ):
        repaired_edge_index = self.repair_graph(
            features=features,
            sparse_edge_index=sparse_edge_index,
            original_edge_index=original_edge_index,
            num_nodes=num_nodes,
            device=device,
            semantic_edge_index=semantic_edge_index,
            repair_node_mask=repair_node_mask,
            candidate_node_mask=candidate_node_mask,
        )
        repaired_edge_index, _ = add_self_loops(repaired_edge_index, num_nodes=num_nodes)
        return repaired_edge_index


class LLM_GNN(nn.Module):
    """Complete LLM-GNN model with targeted topology repair."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        k_neighbors=5,
        beta=0.5,
        backbone="gcn",
        num_layers=2,
        dropout=0.5,
        heads=4,
    ):
        super().__init__()
        self.structure_learner = StructureLearner(k_neighbors=k_neighbors, beta=beta)

        if backbone == "gcn":
            self.gnn = GCN(in_channels, hidden_channels, out_channels, num_layers, dropout)
        elif backbone == "gat":
            self.gnn = GAT(in_channels, hidden_channels, out_channels, num_layers, dropout, heads)
        elif backbone == "sage":
            self.gnn = GraphSAGE(in_channels, hidden_channels, out_channels, num_layers, dropout)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.backbone_type = backbone
        self._cached_edge_index = None
        self._cache_key = None

    def _make_cache_key(self, edge_index, original_edge_index, semantic_edge_index, repair_node_mask, candidate_node_mask, num_nodes):
        return (
            num_nodes,
            edge_index.data_ptr(),
            None if original_edge_index is None else original_edge_index.data_ptr(),
            None if semantic_edge_index is None else semantic_edge_index.data_ptr(),
            None if repair_node_mask is None else repair_node_mask.data_ptr(),
            None if candidate_node_mask is None else candidate_node_mask.data_ptr(),
        )

    def precompute_structure(
        self,
        x,
        edge_index,
        num_nodes=None,
        semantic_edge_index=None,
        original_edge_index=None,
        repair_node_mask=None,
        candidate_node_mask=None,
    ):
        if num_nodes is None:
            num_nodes = x.size(0)
        print(f"  Precomputing repaired graph for {num_nodes} nodes...", flush=True)
        self._cached_edge_index = self.structure_learner(
            x,
            edge_index,
            num_nodes,
            x.device,
            semantic_edge_index=semantic_edge_index,
            original_edge_index=original_edge_index,
            repair_node_mask=repair_node_mask,
            candidate_node_mask=candidate_node_mask,
        )
        self._cache_key = self._make_cache_key(
            edge_index,
            original_edge_index,
            semantic_edge_index,
            repair_node_mask,
            candidate_node_mask,
            num_nodes,
        )
        print(f"  Done. Enhanced edges: {self._cached_edge_index.size(1)}")
        return self._cached_edge_index

    def forward(
        self,
        x,
        edge_index,
        num_nodes=None,
        use_structure_learning=True,
        semantic_edge_index=None,
        original_edge_index=None,
        repair_node_mask=None,
        candidate_node_mask=None,
    ):
        if num_nodes is None:
            num_nodes = x.size(0)

        if use_structure_learning:
            cache_key = self._make_cache_key(
                edge_index,
                original_edge_index,
                semantic_edge_index,
                repair_node_mask,
                candidate_node_mask,
                num_nodes,
            )
            if self._cached_edge_index is None or self._cache_key != cache_key:
                self._cached_edge_index = self.structure_learner(
                    x,
                    edge_index,
                    num_nodes,
                    x.device,
                    semantic_edge_index=semantic_edge_index,
                    original_edge_index=original_edge_index,
                    repair_node_mask=repair_node_mask,
                    candidate_node_mask=candidate_node_mask,
                )
                self._cache_key = cache_key
            enhanced_edge_index = self._cached_edge_index
        else:
            enhanced_edge_index = edge_index

        return self.gnn(x, enhanced_edge_index)

    def get_embeddings(
        self,
        x,
        edge_index,
        num_nodes=None,
        use_structure_learning=True,
        semantic_edge_index=None,
        original_edge_index=None,
        repair_node_mask=None,
        candidate_node_mask=None,
    ):
        if num_nodes is None:
            num_nodes = x.size(0)

        if use_structure_learning:
            enhanced_edge_index = self.structure_learner(
                x,
                edge_index,
                num_nodes,
                x.device,
                semantic_edge_index=semantic_edge_index,
                original_edge_index=original_edge_index,
                repair_node_mask=repair_node_mask,
                candidate_node_mask=candidate_node_mask,
            )
        else:
            enhanced_edge_index = edge_index

        return self.gnn.get_embeddings(x, enhanced_edge_index)
