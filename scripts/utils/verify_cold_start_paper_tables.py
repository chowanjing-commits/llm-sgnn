"""
Verify cold-start appendix tables in docs/paper_draft_zh.md.

The script reconstructs Tables D1-D8 from the curated CSV summaries under
logs/ and compares the rendered markdown cells in the Chinese paper draft.
It is intentionally read-only and does not write regenerated tables.
"""
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_PATH = PROJECT_ROOT / "docs" / "paper_draft_zh.md"

DATASETS = ["Cora", "PubMed", "WikiCS", "ogbn-arxiv-full"]
BACKBONES = [
    ("Ours-GCN", "GCN"),
    ("Ours-GAT", "GAT"),
    ("Ours-GraphSAGE", "GraphSAGE"),
]

BUDGET_FILES = [
    ("Cora", "logs/conda_cold_start_cora_budget_summary_20260528_235539.csv"),
    ("PubMed", "logs/conda_cold_start_pubmed_budget_summary_20260528_235624.csv"),
    ("WikiCS", "logs/conda_cold_start_wikics_budget_summary_20260528_235716.csv"),
    ("ogbn-arxiv-10k", "logs/conda_cold_start_arxiv10k_budget_summary_20260529_000157.csv"),
    ("ogbn-arxiv-full", "logs/conda_cold_start_arxiv_full_budget_summary_20260529_001323.csv"),
]

MAIN_DEFAULT_FILES = {
    "Cora": "logs/cora_single_pilot_coldstart_drop0_node75_20260529_212853.csv",
    "PubMed": "logs/pubmed_single_pilot_coldstart_drop0_node75_20260529_212946.csv",
    "WikiCS": "logs/wikics_single_pilot_coldstart_drop0_node75_20260529_213036.csv",
    "ogbn-arxiv-full": "logs/arxiv_single_pilot_coldstart_drop0_node75_20260529_213555.csv",
}

MAIN_EDGE_ONLY_FILES = {
    "Cora": "logs/cora_single_pilot_coldstart_drop0_node75_20260530_162230.csv",
    "PubMed": "logs/pubmed_single_pilot_coldstart_drop0_node75_20260530_162323.csv",
    "WikiCS": "logs/wikics_single_pilot_coldstart_drop0_node75_20260530_162614.csv",
    "ogbn-arxiv-full": "logs/arxiv_single_pilot_coldstart_drop0_node75_20260530_163630.csv",
}

MAIN_NO_COLD_START_FILES = {
    "Cora": "logs/cora_single_pilot_drop0_node75_20260530_164206.csv",
    "PubMed": "logs/pubmed_single_pilot_drop0_node75_20260530_164245.csv",
    "WikiCS": "logs/wikics_single_pilot_drop0_node75_20260530_164326.csv",
    "ogbn-arxiv-full": "logs/arxiv_single_pilot_drop0_node75_20260530_164516.csv",
}

MAIN_HIGH_RELIABILITY_FILES = {
    "Cora": "logs/cora_single_pilot_coldstart_drop0_node75_20260530_171132.csv",
    "PubMed": "logs/pubmed_single_pilot_coldstart_drop0_node75_20260530_171213.csv",
    "WikiCS": "logs/wikics_single_pilot_coldstart_drop0_node75_20260530_171309.csv",
    "ogbn-arxiv-full": "logs/arxiv_single_pilot_coldstart_drop0_node75_20260530_172042.csv",
}

FORMAL_CONTROL_FILES = {
    "Cora": "logs/cold_start_formal_cora_sage_summary_20260530_142118.csv",
    "PubMed": "logs/cold_start_formal_pubmed_sage_summary_20260530_142216.csv",
    "WikiCS": "logs/cold_start_formal_wikics_sage_summary_20260530_142300.csv",
    "ogbn-arxiv-full": "logs/cold_start_formal_arxiv_full_sage_summary_20260530_143237.csv",
}

AGREE_OR_FILES = {
    "Cora": "logs/cold_start_agree_or_cora_sage_summary_20260530_145743.csv",
    "PubMed": "logs/cold_start_agree_or_pubmed_sage_summary_20260530_145827.csv",
    "WikiCS": "logs/cold_start_agree_or_wikics_sage_summary_20260530_154715.csv",
    "ogbn-arxiv-full": "logs/cold_start_agree_or_arxiv_full_sage_summary_20260530_155008.csv",
}

AGREE_W03_FILES = {
    "Cora": "logs/cold_start_agree_w03_cora_sage_summary_20260530_161119.csv",
    "PubMed": "logs/cold_start_agree_w03_pubmed_sage_summary_20260530_161145.csv",
    "WikiCS": "logs/cold_start_agree_w03_wikics_sage_summary_20260530_161210.csv",
    "ogbn-arxiv-full": "logs/cold_start_agree_w03_arxiv_full_sage_summary_20260530_161438.csv",
}

AGREE_EDGE_ONLY_FILES = {
    "Cora": "logs/cold_start_agree_edgeonly_cora_sage_summary_20260530_160337.csv",
    "PubMed": "logs/cold_start_agree_edgeonly_pubmed_sage_summary_20260530_160611.csv",
    "WikiCS": "logs/cold_start_agree_edgeonly_wikics_sage_summary_20260530_160634.csv",
    "ogbn-arxiv-full": "logs/cold_start_agree_edgeonly_arxiv_full_sage_summary_20260530_161051.csv",
}

AGREE_AND_HIGH_FILES = {
    "Cora": "logs/cold_start_agree_and_w1_cora_sage_summary_20260530_165913.csv",
    "PubMed": "logs/cold_start_agree_and_w1_pubmed_sage_summary_20260530_165954.csv",
    "WikiCS": "logs/cold_start_agree_and_w1_wikics_sage_summary_20260530_170023.csv",
    "ogbn-arxiv-full": "logs/cold_start_agree_and_w1_arxiv_full_sage_summary_20260530_170256.csv",
}


def read_csv(rel_path):
    path = PROJECT_ROOT / rel_path
    if not path.exists():
        raise AssertionError(f"Missing CSV: {rel_path}")
    return pd.read_csv(path)


def pct(value):
    return f"{float(value) * 100.0:.2f}"


def pct_pm(row):
    return f"{pct(row['accuracy'])} ± {pct(row['std'])}"


def mean1(value):
    return f"{float(value):.1f}"


def markdown_rows(table_id):
    text = PAPER_PATH.read_text(encoding="utf-8")
    marker = f"**表 {table_id}."
    start = text.find(marker)
    if start < 0:
        raise AssertionError(f"Table {table_id} caption not found")
    rows = []
    in_table = False
    for line in text[start:].splitlines():
        if line.startswith("|"):
            in_table = True
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if all(set(cell) <= {"-", ":"} for cell in cells):
                continue
            rows.append(cells)
        elif in_table:
            break
    if not rows:
        raise AssertionError(f"Table {table_id} body not found")
    return rows


def single_row(df, **filters):
    mask = pd.Series(True, index=df.index)
    for column, expected in filters.items():
        if isinstance(expected, float):
            mask &= (df[column].astype(float) - expected).abs() < 1e-9
        else:
            mask &= df[column] == expected
    rows = df[mask]
    if len(rows) != 1:
        raise AssertionError(f"Expected one row for filters {filters}, found {len(rows)}")
    return rows.iloc[0]


def compare_table(table_id, expected_header, expected_rows):
    actual = markdown_rows(table_id)
    actual_header, actual_rows = actual[0], actual[1:]
    errors = []
    if actual_header != expected_header:
        errors.append(f"header expected {expected_header}, got {actual_header}")
    if len(actual_rows) != len(expected_rows):
        errors.append(f"row count expected {len(expected_rows)}, got {len(actual_rows)}")
    for idx, expected in enumerate(expected_rows):
        if idx >= len(actual_rows):
            break
        if actual_rows[idx] != expected:
            errors.append(f"row {idx + 1} expected {expected}, got {actual_rows[idx]}")
    if errors:
        raise AssertionError(f"Table {table_id} mismatch:\n" + "\n".join(errors))


def budget_rows(metric):
    rows = []
    for dataset, rel_path in BUDGET_FILES:
        df = read_csv(rel_path)
        for ratio in [0.25, 0.50, 0.75, 1.00]:
            cluster = single_row(
                df,
                admission_ratio=ratio,
                admission_strategy="cluster_representative",
            )
            random = single_row(df, admission_ratio=ratio, admission_strategy="random")
            cluster_value = float(cluster[metric]) * 100.0
            random_value = float(random[metric]) * 100.0
            rows.append(
                [
                    dataset,
                    f"{ratio:.2f}",
                    f"{cluster_value:.2f}",
                    f"{random_value:.2f}",
                    f"{cluster_value - random_value:+.2f}",
                ]
            )
    return rows


def main_entry_rows(file_map, include_recovery_edges=True):
    rows = []
    for dataset in DATASETS:
        df = read_csv(file_map[dataset])
        for model_name, backbone in BACKBONES:
            row = single_row(df, model=model_name)
            cells = [dataset, backbone, pct_pm(row)]
            if include_recovery_edges:
                cells.extend(
                    [
                        mean1(row["selected_cold_start_nodes_mean"]),
                        mean1(row["pseudo_train_nodes_mean"]),
                        mean1(row["recovery_edges_mean"]),
                    ]
                )
            else:
                cells.append(mean1(row["pseudo_train_nodes_mean"]))
            cells.append(pct(row["pseudo_label_accuracy_mean"]))
            rows.append(cells)
    return rows


def graph_sage_row(rel_path):
    df = read_csv(rel_path)
    if "model" in df.columns:
        return single_row(df, model="Ours-GraphSAGE")
    return df.iloc[0]


def no_cold_start_row(dataset):
    df = read_csv(FORMAL_CONTROL_FILES[dataset])
    return single_row(df, admission_ratio=0.0, admission_strategy="random")


def loss_weight_rows():
    rows = []
    for dataset in DATASETS:
        no_cold = no_cold_start_row(dataset)
        w1 = graph_sage_row(AGREE_OR_FILES[dataset])
        w03 = graph_sage_row(AGREE_W03_FILES[dataset])
        w0 = graph_sage_row(AGREE_EDGE_ONLY_FILES[dataset])
        rows.append(
            [
                dataset,
                pct_pm(no_cold),
                pct_pm(w1),
                pct_pm(w03),
                pct_pm(w0),
                pct(w1["pseudo_label_accuracy_mean"]),
            ]
        )
    return rows


def no_vs_edge_rows():
    rows = []
    for dataset in DATASETS:
        no_df = read_csv(MAIN_NO_COLD_START_FILES[dataset])
        edge_df = read_csv(MAIN_EDGE_ONLY_FILES[dataset])
        for model_name, backbone in BACKBONES:
            no_row = single_row(no_df, model=model_name)
            edge_row = single_row(edge_df, model=model_name)
            no_value = float(no_row["accuracy"]) * 100.0
            edge_value = float(edge_row["accuracy"]) * 100.0
            rows.append(
                [
                    dataset,
                    backbone,
                    f"{no_value:.2f}",
                    f"{edge_value:.2f}",
                    f"{edge_value - no_value:+.2f}",
                ]
            )
    return rows


def high_reliability_rows():
    rows = []
    for dataset in DATASETS:
        no_cold = no_cold_start_row(dataset)
        edge = graph_sage_row(AGREE_EDGE_ONLY_FILES[dataset])
        high = graph_sage_row(AGREE_AND_HIGH_FILES[dataset])
        rows.append(
            [
                dataset,
                pct_pm(no_cold),
                pct_pm(edge),
                pct_pm(high),
                mean1(high["pseudo_train_nodes_mean"]),
                pct(high["pseudo_label_accuracy_mean"]),
            ]
        )
    return rows


def main():
    compare_table(
        "D1",
        ["数据集", "准入比例", "聚类代表准入", "随机准入", "差值"],
        budget_rows("accuracy"),
    )
    compare_table(
        "D2",
        ["数据集", "准入比例", "聚类代表准入", "随机准入", "差值"],
        budget_rows("pseudo_label_accuracy_mean"),
    )
    compare_table(
        "D3",
        ["数据集", "骨干", "准确率", "准入冷启动节点", "伪标签训练节点", "新增语义边", "伪标签准确率"],
        main_entry_rows(MAIN_DEFAULT_FILES),
    )
    compare_table(
        "D4",
        ["数据集", "不加入 cold-start", "权重 1.0", "权重 0.3", "权重 0.0 / edge-only", "伪标签准确率"],
        loss_weight_rows(),
    )
    compare_table(
        "D5",
        ["数据集", "骨干", "准确率", "准入冷启动节点", "伪标签诊断节点", "新增语义边", "伪标签准确率"],
        main_entry_rows(MAIN_EDGE_ONLY_FILES),
    )
    compare_table(
        "D6",
        ["数据集", "骨干", "No cold-start", "Edge-only", "差值"],
        no_vs_edge_rows(),
    )
    compare_table(
        "D7",
        ["数据集", "不加入 cold-start", "Edge-only", "高可靠伪标签", "伪标签训练节点", "伪标签准确率"],
        high_reliability_rows(),
    )
    compare_table(
        "D8",
        ["数据集", "骨干", "准确率", "伪标签训练节点", "伪标签准确率"],
        main_entry_rows(MAIN_HIGH_RELIABILITY_FILES, include_recovery_edges=False),
    )
    print("Cold-start paper table verification passed for Tables D1-D8.")


if __name__ == "__main__":
    main()
