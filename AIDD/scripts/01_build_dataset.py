#!/usr/bin/env python3
"""
Step 1: 数据层 —— 生成干净的抗体-抗原亲和力样本表

输入：
    - sabdab_data/sabdab_summary.tsv
    - data/raw/skempi_v2.csv（不存在时自动下载）

输出：
    - processed/antibody_antigen_dataset.csv

对齐逻辑：
    1. 从 SKEMPI 的 #Pdb 字段解析出 PDB ID 和链分组（如 1JRH_LH_I -> 1JRH, [LH, I]）。
    2. 用 PDB ID 在 SAbDab summary 中查找对应条目。
    3. 选择满足以下条件的 SAbDab 条目：
       - SKEMPI 某一分组包含 SAbDab 的所有抗原链；
       - 其余分组包含 SAbDab 的 Hchain 和 Lchain。
    4. 取 Affinity_mut_parsed 作为亲和力（M），计算 pKD = -log10(KD/M)。
"""

import re
import urllib.request
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "processed"
SKEMPI_URL = "https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv"


def download_skempi(csv_path: Path) -> None:
    """下载 SKEMPI v2 CSV（约 1.6 MB）。"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading SKEMPI v2 from {SKEMPI_URL} ...")
    urllib.request.urlretrieve(SKEMPI_URL, csv_path)
    print(f"Saved to {csv_path}")


def parse_skempi_pdb(skempi_id: str):
    """
    把 '#Pdb' 字段拆成 PDB ID 和链分组。
    例：
        '1AHW_AB_C'     -> ('1AHW', ['AB', 'C'])
        '1JRH_LH_I'     -> ('1JRH', ['LH', 'I'])
        '1DVF_AB_CD'    -> ('1DVF', ['AB', 'CD'])
    """
    parts = skempi_id.split("_")
    pdb_id = parts[0].upper()
    chain_groups = parts[1:]
    return pdb_id, chain_groups


def find_matching_sabdab_row(sab_rows: pd.DataFrame, chain_groups):
    """
    在给定 PDB 的 SAbDab 条目中，找到与 SKEMPI 链分组匹配的那一行。

    匹配规则：
        - 存在某个分组 G，其包含 SAbDab 的全部抗原链；
        - G 不包含 H/L 链；
        - 其余分组合起来包含 H/L 链。
    """
    best_row = None
    best_score = None

    for _, row in sab_rows.iterrows():
        antigen_chains = [
            c.strip()
            for c in str(row["antigen_chain"]).split("|")
            if c.strip() and c.strip() != "NA"
        ]
        h_chain = str(row["Hchain"]).strip()
        l_chain = str(row["Lchain"]).strip()
        hl_set = {h_chain, l_chain}
        ag_set = set(antigen_chains)

        if not ag_set or not h_chain or not l_chain:
            continue

        for i, group in enumerate(chain_groups):
            gset = set(group)
            if not ag_set <= gset:
                continue
            if gset & hl_set:
                continue

            other_chars = set(
                "".join(chain_groups[j] for j in range(len(chain_groups)) if j != i)
            )
            if not hl_set <= other_chars:
                continue

            # 评分： prefer 抗原分组尽量紧凑
            score = len(group) - 0.1 * len(ag_set)
            if best_score is None or score < best_score:
                best_score = score
                best_row = row
                best_row = best_row.copy()
                best_row["_skempi_antigen_group"] = group
                best_row["_skempi_antibody_group"] = "".join(
                    chain_groups[j] for j in range(len(chain_groups)) if j != i
                )

    return best_row


def compute_pkd(value):
    """把 KD（单位 M）转成 pKD = -log10(M)；非法值返回 None。"""
    import math

    if pd.isna(value):
        return None
    try:
        value = float(value)
        if value <= 0:
            return None
        return -math.log10(value)
    except Exception:
        return None


def build_dataset():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. 读取 SAbDab summary ----
    sab_path = PROJECT_ROOT / "data" / "sabdab_data" / "sabdab_summary.tsv"
    sab = pd.read_csv(sab_path, sep="\t", low_memory=False)
    sab["pdb_upper"] = sab["pdb"].astype(str).str.upper().str.strip()
    sab = sab[
        sab["Hchain"].notna()
        & sab["Lchain"].notna()
        & sab["antigen_chain"].notna()
    ].copy()
    sab["Hchain"] = sab["Hchain"].astype(str).str.strip()
    sab["Lchain"] = sab["Lchain"].astype(str).str.strip()

    # ---- 2. 读取 / 下载 SKEMPI v2 ----
    skempi_path = RAW_DIR / "skempi_v2.csv"
    if not skempi_path.exists():
        download_skempi(skempi_path)

    sk = pd.read_csv(skempi_path, sep=";")
    sk["pdb_id"] = sk["#Pdb"].apply(lambda x: parse_skempi_pdb(x)[0])
    sk["chain_groups"] = sk["#Pdb"].apply(lambda x: parse_skempi_pdb(x)[1])

    # ---- 3. 对齐 ----
    overlap_pdbs = set(sk["pdb_id"]) & set(sab["pdb_upper"])
    print(f"SAbDab entries with H/L/antigen: {len(sab)}")
    print(f"SKEMPI rows total: {len(sk)}")
    print(f"Overlap PDB IDs: {len(overlap_pdbs)}")

    records = []
    matched = 0
    unmatched = 0

    for idx, sk_row in sk[sk["pdb_id"].isin(overlap_pdbs)].iterrows():
        pdb_id = sk_row["pdb_id"]
        chain_groups = sk_row["chain_groups"]
        sab_rows = sab[sab["pdb_upper"] == pdb_id]
        row = find_matching_sabdab_row(sab_rows, chain_groups)

        if row is None:
            unmatched += 1
            continue

        matched += 1

        # 亲和力解析：优先用 parsed 数值列
        aff_mut = sk_row.get("Affinity_mut_parsed", sk_row.get("Affinity_mut (M)"))
        aff_wt = sk_row.get("Affinity_wt_parsed", sk_row.get("Affinity_wt (M)"))

        records.append(
            {
                "skempi_id": sk_row["#Pdb"],
                "pdb": pdb_id,
                "h_chain": row["Hchain"],
                "l_chain": row["Lchain"],
                "antigen_chains": " | ".join(
                    c.strip()
                    for c in str(row["antigen_chain"]).split("|")
                    if c.strip() and c.strip() != "NA"
                ),
                "skempi_antigen_group": row["_skempi_antigen_group"],
                "skempi_antibody_group": row["_skempi_antibody_group"],
                "protein_1": sk_row.get("Protein 1"),
                "protein_2": sk_row.get("Protein 2"),
                "mutation_pdb": sk_row.get("Mutation(s)_PDB"),
                "mutation_cleaned": sk_row.get("Mutation(s)_cleaned"),
                "affinity_mut_M": aff_mut,
                "pkd_mut": compute_pkd(aff_mut),
                "affinity_wt_M": aff_wt,
                "pkd_wt": compute_pkd(aff_wt),
                "temperature": sk_row.get("Temperature"),
                "method": row.get("method"),
                "resolution": row.get("resolution"),
                "sabdab_affinity": row.get("affinity"),
                "sabdab_delta_g": row.get("delta_g"),
                "reference_pmid": sk_row.get("Reference"),
                "structure_path": f"data/raw/SKEMPI2_PDBs/PDBs/{pdb_id}.pdb",
            }
        )

    df = pd.DataFrame(records)
    out_path = PROCESSED_DIR / "antibody_antigen_dataset.csv"
    df.to_csv(out_path, index=False)

    print(f"Matched rows: {matched}")
    print(f"Unmatched rows: {unmatched}")
    print(f"Output: {out_path} ({len(df)} rows)")
    print("\nLabel distribution (pKD_mut):")
    print(df["pkd_mut"].describe())
    print("\nFirst 3 rows:")
    print(df.head(3).T)


if __name__ == "__main__":
    build_dataset()
