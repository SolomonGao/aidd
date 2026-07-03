#!/usr/bin/env python3
"""
Step 2: 从 CIF 结构中提取抗体-抗原界面和几何特征

输入：
    - processed/sabdab2_labeled_dataset.csv
    - data/splits_final/*.cif

输出：
    - processed/structural_features.csv

计算的特征：
    - H/L/抗原链的残基数、原子数
    - 界面残基数（CA 距离 < 8 Å）
    - 界面原子接触对数（重原子距离 < 5 Å）
    - VH-VL 质心距离
    - 结构分辨率、方法等元数据
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB.MMCIFParser import MMCIFParser

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELED_CSV = PROJECT_ROOT / "processed" / "sabdab2_labeled_dataset.csv"
OUTPUT_CSV = PROJECT_ROOT / "processed" / "structural_features.csv"
CIF_DIR = PROJECT_ROOT / "data" / "splits_final"

PARSER = MMCIFParser(QUIET=True)


def parse_chain_set(value):
    if pd.isna(value) or str(value).strip() in ("", "NA", "+"):
        return []
    parts = re.split(r"[|/]", str(value))
    return [p.strip() for p in parts if p.strip() and p.strip() != "+"]


def extract_chain_coords(structure, chain_ids):
    """从结构中抽取指定链的 CA 坐标和全部重原子坐标。"""
    ca_coords = []
    ca_residue_ids = []
    heavy_coords = []
    heavy_residue_ids = []

    model = structure[0]
    for cid in chain_ids:
        if cid not in model:
            continue
        chain = model[cid]
        for res in chain:
            if res.id[0] != " ":  # 跳过 HETATM / 水
                continue
            resid = (res.id[1], res.id[2].strip())  # (resseq, icode)
            if "CA" in res:
                ca_coords.append(res["CA"].coord)
                ca_residue_ids.append(resid)
            for atom in res:
                if atom.element == "H":
                    continue
                heavy_coords.append(atom.coord)
                heavy_residue_ids.append(resid)

    return (
        np.array(ca_coords) if ca_coords else np.empty((0, 3)),
        ca_residue_ids,
        np.array(heavy_coords) if heavy_coords else np.empty((0, 3)),
        heavy_residue_ids,
    )


def count_contacts_chunked(coords_a, coords_b, cutoff, chunk_size=256):
    """分块计算 coords_a 与 coords_b 之间距离 < cutoff 的对数。"""
    if coords_a.shape[0] == 0 or coords_b.shape[0] == 0:
        return 0
    count = 0
    n = coords_a.shape[0]
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = coords_a[start:end]
        # (chunk, 1, 3) - (1, n_b, 3) -> (chunk, n_b)
        d2 = np.sum((chunk[:, None, :] - coords_b[None, :, :]) ** 2, axis=2)
        count += np.sum(d2 < cutoff ** 2)
    return count


def count_interface_residues(ca_a, ids_a, ca_b, cutoff=8.0):
    """返回 coords_a 中距离 coords_b 任意点 < cutoff 的唯一残基数。"""
    if ca_a.shape[0] == 0 or ca_b.shape[0] == 0:
        return 0
    # 分块避免大矩阵
    close = set()
    chunk_size = 256
    n = ca_a.shape[0]
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        d2 = np.sum((ca_a[start:end, None, :] - ca_b[None, :, :]) ** 2, axis=2)
        mask = d2 < cutoff ** 2
        for i, row in enumerate(mask):
            if np.any(row):
                close.add(ids_a[start + i])
    return len(close)


def compute_com(coords):
    if coords.shape[0] == 0:
        return None
    return coords.mean(axis=0)


def extract_features_for_row(row):
    instance = row["instance"]
    cif_path = CIF_DIR / f"{instance}.cif"

    features = {
        "instance": instance,
        "pdb_id": row["pdb_id"],
        "h_chain": row["h_chain"],
        "l_chain": row["l_chain"],
        "antigen_chains": row["antigen_chains"],
        "cif_path": str(cif_path.relative_to(PROJECT_ROOT)),
    }

    if not cif_path.exists():
        features["parse_status"] = "missing_cif"
        return features

    try:
        structure = PARSER.get_structure(instance, str(cif_path))
    except Exception as e:
        features["parse_status"] = f"parse_error: {e}"
        return features

    # 链 ID
    h_chains = parse_chain_set(row["h_chain"])
    l_chains = parse_chain_set(row["l_chain"])
    ag_chains = parse_chain_set(row["antigen_chains"])

    ab_chains = h_chains + l_chains

    # 提取坐标
    ca_h, ids_h, heavy_h, heavy_ids_h = extract_chain_coords(structure, h_chains)
    ca_l, ids_l, heavy_l, heavy_ids_l = extract_chain_coords(structure, l_chains)
    ca_ag, ids_ag, heavy_ag, heavy_ids_ag = extract_chain_coords(structure, ag_chains)

    ca_ab = np.vstack([ca_h, ca_l]) if ca_h.size and ca_l.size else (ca_h if ca_h.size else ca_l)
    ids_ab = ids_h + ids_l
    heavy_ab = np.vstack([heavy_h, heavy_l]) if heavy_h.size and heavy_l.size else (heavy_h if heavy_h.size else heavy_l)

    # 基础统计
    features["h_residues"] = len(ids_h)
    features["l_residues"] = len(ids_l)
    features["ag_residues"] = len(ids_ag)
    features["ab_residues"] = len(ids_ab)
    features["h_atoms"] = heavy_h.shape[0]
    features["l_atoms"] = heavy_l.shape[0]
    features["ag_atoms"] = heavy_ag.shape[0]

    # 界面特征
    features["interface_residues_ab"] = count_interface_residues(ca_ab, ids_ab, ca_ag, cutoff=8.0)
    features["interface_residues_ag"] = count_interface_residues(ca_ag, ids_ag, ca_ab, cutoff=8.0)
    features["interface_contacts_ca_8A"] = count_contacts_chunked(ca_ab, ca_ag, 8.0)
    features["interface_contacts_heavy_5A"] = count_contacts_chunked(heavy_ab, heavy_ag, 5.0)

    # VH-VL 距离（仅对 paired 抗体）
    com_h = compute_com(ca_h)
    com_l = compute_com(ca_l)
    if com_h is not None and com_l is not None:
        features["vh_vl_distance"] = float(np.linalg.norm(com_h - com_l))
    else:
        features["vh_vl_distance"] = None

    features["parse_status"] = "ok"
    return features


def main():
    print(f"Reading labeled dataset from {LABELED_CSV}")
    df = pd.read_csv(LABELED_CSV, low_memory=False)
    print(f"Total rows: {len(df)}")

    records = []
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"  Processing {idx}/{len(df)} ...")
        records.append(extract_features_for_row(row))

    features_df = pd.DataFrame(records)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nOutput saved to {OUTPUT_CSV}")
    print("\nParse status:")
    print(features_df["parse_status"].value_counts())
    print("\nFeature summary:")
    numeric_cols = [
        "h_residues",
        "l_residues",
        "ag_residues",
        "interface_residues_ab",
        "interface_residues_ag",
        "interface_contacts_ca_8A",
        "interface_contacts_heavy_5A",
        "vh_vl_distance",
    ]
    print(features_df[numeric_cols].describe())


if __name__ == "__main__":
    main()
