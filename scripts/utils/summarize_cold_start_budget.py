"""
Generate paper-ready markdown tables for cold-start budget runs.
"""
import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUTS = [
    ("Cora", "logs/conda_cold_start_cora_budget_summary_20260528_235539.csv"),
    ("PubMed", "logs/conda_cold_start_pubmed_budget_summary_20260528_235624.csv"),
    ("WikiCS", "logs/conda_cold_start_wikics_budget_summary_20260528_235716.csv"),
    ("ogbn-arxiv-10k", "logs/conda_cold_start_arxiv10k_budget_summary_20260529_000157.csv"),
    ("ogbn-arxiv-full", "logs/conda_cold_start_arxiv_full_budget_summary_20260529_001323.csv"),
]


def load_rows(metric):
    rows = []
    for dataset, rel_path in DEFAULT_INPUTS:
        path = PROJECT_ROOT / rel_path
        df = pd.read_csv(path)
        pivot = df.pivot(index="admission_ratio", columns="admission_strategy", values=metric)
        for ratio, values in pivot.sort_index().iterrows():
            cluster = float(values["cluster_representative"]) * 100.0
            random = float(values["random"]) * 100.0
            rows.append((dataset, float(ratio), cluster, random, cluster - random))
    return rows


def print_table(title, metric):
    print(title)
    print()
    print("| Dataset | Admission ratio | Cluster representative | Random | Delta |")
    print("|---|---:|---:|---:|---:|")
    for dataset, ratio, cluster, random, delta in load_rows(metric):
        print(f"| {dataset} | {ratio:.2f} | {cluster:.2f} | {random:.2f} | {delta:+.2f} |")
    print()


def main():
    parser = argparse.ArgumentParser(description="Summarize cold-start budget CSV files as markdown tables.")
    parser.add_argument(
        "--metric",
        type=str,
        default="both",
        choices=["accuracy", "pseudo_label_accuracy_mean", "both"],
    )
    args = parser.parse_args()

    if args.metric in {"accuracy", "both"}:
        print_table("Table D1: Test accuracy", "accuracy")
    if args.metric in {"pseudo_label_accuracy_mean", "both"}:
        print_table("Table D2: Pseudo-label accuracy", "pseudo_label_accuracy_mean")


if __name__ == "__main__":
    main()
