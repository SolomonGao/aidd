#!/usr/bin/env python3
"""
Step 1b: 基于 SAbDab2 splits_final + SAbDab affinity 构建带标签训练集

输入：
    - data/splits_final/abag_split.csv
    - data/splits_final/*.cif
    - data/sabdab_data/sabdab_summary.tsv

输出：
    - processed/sabdab2_labeled_dataset.csv

说明：
    - splits_final 本身不含亲和力标签，需要从 SAbDab summary 的 affinity 列合并。
    - 使用 abag_split（抗原感知的 split），避免抗体/抗原序列相似导致的泄漏。
    - 文件名如 pdb_00001ejo_H_L.cif 对应 PDB 1EJO，H 链 + L 链。
    - 提供两级标签可信度：
        * chain_matched：PDB + 链 都对上，标签最可靠
        * pdb_only：只按 PDB 匹配，标签可能有噪声（用于扩充数据量）
"""

import math
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_CSV = PROJECT_ROOT / "data" / "splits_final" / "abag_split.csv"
SABDAB_TSV = PROJECT_ROOT / "data" / "sabdab_data" / "sabdab_summary.tsv"
OUTPUT_CSV = PROJECT_ROOT / "processed" / "sabdab2_labeled_dataset.csv"


def parse_pdb_id(pdb_id_col: str) -> str:
    """把 'pdb_00001ejo' 转成标准 PDB ID '1EJO'。"""
    return re.sub(r"^pdb_0+", "", pdb_id_col).upper()


def parse_chain_set(value):
    """把 'A | B' 或 'A/B' 解析成 chain 集合。"""
    if pd.isna(value) or str(value).strip() in ("", "NA", "+"):
        return set()
    parts = re.split(r"[|/]", str(value))
    return {p.strip() for p in parts if p.strip() and p.strip() != "+"}


def compute_pkd(affinity_m):
    try:
        affinity_m = float(affinity_m)
        if affinity_m <= 0:
            return None
        return -math.log10(affinity_m)
    except Exception:
        return None


def make_record(inst, best_row, confidence):
    """根据 splits 实例和 SAbDab 行生成一条记录。"""
    pkd = compute_pkd(best_row["affinity"])
    try:
        delta_g = float(best_row["delta_g"]) if pd.notna(best_row["delta_g"]) else None
    except Exception:
        delta_g = None

    return {
        "instance": inst["INSTANCE"],
        "pdb_id": inst["std_pdb"],
        "sabdab_id": inst["SABDAB_ID"],
        "heavy_id": inst["HEAVY_ID"],
        "light_id": inst["LIGHT_ID"],
        "h_chain": inst["Hchain"],
        "l_chain": inst["Lchain"],
        "antigen_chains": inst["agchains"],
        "antigen_types": inst["agtypes"],
        "h_seq": inst["Hseq_expected"],
        "l_seq": inst["Lseq_expected"],
        "vh_numerable_seq": inst["VH_numerable_seq"],
        "vl_numerable_seq": inst["VL_numerable_seq"],
        "antigen_seq": inst["agexpectedseqs"],
        "cdrh1": inst["CDRH1"],
        "cdrh2": inst["CDRH2"],
        "cdrh3": inst["CDRH3"],
        "cdrl1": inst["CDRL1"],
        "cdrl2": inst["CDRL2"],
        "cdrl3": inst["CDRL3"],
        "affinity_M": float(best_row["affinity"]),
        "pkd": pkd,
        "delta_g": delta_g,
        "split": inst["ab_ag_split"],
        "method": inst["method"],
        "resolution": inst["resolution"],
        "antibody_type": inst["type"],
        "construct": inst["construct"],
        "holo": inst["holo"],
        "is_paired": inst["is_paired"],
        "label_confidence": confidence,
        "sabdab_h_chain": best_row["Hchain"],
        "sabdab_l_chain": best_row["Lchain"],
        "sabdab_antigen_chains": best_row["antigen_chain"],
        "cif_path": f"data/splits_final/{inst['INSTANCE']}.cif",
    }


def build_dataset():
    print(f"Reading splits from {SPLITS_CSV}")
    splits = pd.read_csv(SPLITS_CSV, low_memory=False)
    splits["std_pdb"] = splits["PDB_ID"].apply(parse_pdb_id)
    splits["h_set"] = splits["Hchain"].apply(parse_chain_set)
    splits["l_set"] = splits["Lchain"].apply(parse_chain_set)
    splits["ag_set"] = splits["agchains"].apply(parse_chain_set)
    splits["is_paired"] = splits["type"].isin(["FV", "FAB", "FAB+FC"])

    print(f"Reading SAbDab summary from {SABDAB_TSV}")
    sab = pd.read_csv(SABDAB_TSV, sep="\t", low_memory=False)
    sab = sab[sab["affinity"].notna()].copy()
    sab["pdb_upper"] = sab["pdb"].astype(str).str.upper().str.strip()
    sab["h_set"] = sab["Hchain"].apply(parse_chain_set)
    sab["l_set"] = sab["Lchain"].apply(parse_chain_set)
    sab["ag_set"] = sab["antigen_chain"].apply(parse_chain_set)

    print(f"SAbDab rows with affinity: {len(sab)}")
    print(f"SAbDab unique PDBs with affinity: {sab['pdb_upper'].nunique()}")

    records = []
    matched_instances = set()

    # ---- Pass 1: 严格的链级匹配 ----
    for pdb_id, group in splits.groupby("std_pdb"):
        sab_rows = sab[sab["pdb_upper"] == pdb_id]
        if sab_rows.empty:
            continue

        for _, inst in group.iterrows():
            best_row = None
            best_score = -1

            for _, srow in sab_rows.iterrows():
                ag_overlap = len(inst["ag_set"] & srow["ag_set"])
                if inst["ag_set"] and ag_overlap == 0:
                    continue

                h_overlap = int(bool(inst["h_set"] & srow["h_set"]))
                l_overlap = int(bool(inst["l_set"] & srow["l_set"]))

                if inst["is_paired"]:
                    if not (h_overlap and l_overlap):
                        continue
                else:
                    if not (h_overlap or l_overlap):
                        continue

                score = ag_overlap + h_overlap + l_overlap
                if score > best_score:
                    best_score = score
                    best_row = srow

            if best_row is not None:
                records.append(make_record(inst, best_row, "chain_matched"))
                matched_instances.add(inst["INSTANCE"])

    # ---- Pass 2: 宽松的 PDB 级匹配 ----
    for pdb_id, group in splits.groupby("std_pdb"):
        sab_rows = sab[sab["pdb_upper"] == pdb_id]
        if sab_rows.empty:
            continue

        for _, inst in group.iterrows():
            if inst["INSTANCE"] in matched_instances:
                continue

            best_row = None
            best_score = -1

            for _, srow in sab_rows.iterrows():
                score = len(inst["ag_set"] & srow["ag_set"])
                score += int(bool(inst["h_set"] & srow["h_set"]))
                score += int(bool(inst["l_set"] & srow["l_set"]))
                if score > best_score:
                    best_score = score
                    best_row = srow

            if best_row is not None:
                records.append(make_record(inst, best_row, "pdb_only"))

    df = pd.DataFrame(records)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nTotal labeled instances: {len(df)}")
    print("Label confidence:")
    print(df["label_confidence"].value_counts())
    print(f"Valid pKD labels: {df['pkd'].notna().sum()}")
    print(f"Unique PDBs with labels: {df['pdb_id'].nunique()}")
    print("\nSplit distribution:")
    print(df["split"].value_counts())
    print("\nAntibody type distribution:")
    print(df["antibody_type"].value_counts())
    print("\npKD distribution:")
    print(df["pkd"].describe())
    print(f"\nOutput saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    build_dataset()
