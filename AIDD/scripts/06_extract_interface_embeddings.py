#!/usr/bin/env python3
"""
Step 6: 界面残基 ESM-2 嵌入（paratope / epitope pooling）

动机：
    旧版对整条 VH/VL/抗原做 mean-pooling，把决定亲和力的少数界面残基
    （paratope × epitope）淹没在整条链的平均里，尤其抗原可长达上千残基。
    这里只对 *界面残基* 做池化：
      - paratope = 抗体(H+L)中靠近抗原的残基的 ESM 表示均值
      - epitope  = 抗原中靠近抗体的残基的 ESM 表示均值
    序列直接从结构里按残基顺序取出，界面 mask 用同一套残基索引计算，
    因此 ESM token 与结构残基天然对齐，无需额外的编号映射。

运行：
    conda activate aidd
    python scripts/06_extract_interface_embeddings.py            # ESM-2 650M
    python scripts/06_extract_interface_embeddings.py --model esm2_t30_150M_UR50D  # 更快的原型

输出：
    processed/esm2_interface_embeddings.npz
      instances, paratope_embeddings (N,D), epitope_embeddings (N,D),
      n_paratope, n_epitope, status
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import esm
from Bio.PDB.MMCIFParser import MMCIFParser

try:  # BioPython >=1.80 移除了 three_to_one
    from Bio.Data.PDBData import protein_letters_3to1 as _3to1
except Exception:  # 老版本回退
    from Bio.Data.SCOPData import protein_letters_3to1 as _3to1


def three_to_one(resname):
    return _3to1.get(resname.strip().upper(), "X")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELED_CSV = PROJECT_ROOT / "processed" / "sabdab2_labeled_dataset.csv"
CIF_DIR = PROJECT_ROOT / "data" / "splits_final"
OUTPUT_NPZ = PROJECT_ROOT / "processed" / "esm2_interface_embeddings.npz"

PARSER = MMCIFParser(QUIET=True)
MAX_TOKENS = 1022
INTERFACE_CUTOFF = 10.0  # CA-CA 距离阈值，定义界面邻域


def parse_chain_set(value):
    if pd.isna(value) or str(value).strip() in ("", "NA", "+"):
        return []
    parts = re.split(r"[|/]", str(value))
    return [p.strip() for p in parts if p.strip() and p.strip() != "+"]


def chain_residues(structure, chain_ids):
    """返回 [(one_letter, ca_coord)] 按结构残基顺序（仅标准氨基酸残基）。"""
    out = []
    model = structure[0]
    for cid in chain_ids:
        if cid not in model:
            continue
        for res in model[cid]:
            if res.id[0] != " " or "CA" not in res:
                continue
            out.append((three_to_one(res.get_resname()), res["CA"].coord))
    return out


def interface_mask(ca_side, ca_partner, cutoff=INTERFACE_CUTOFF):
    """side 中每个残基是否在 partner 任意残基 cutoff 内。"""
    if len(ca_side) == 0 or len(ca_partner) == 0:
        return np.zeros(len(ca_side), dtype=bool)
    A = np.array(ca_side); B = np.array(ca_partner)
    mask = np.zeros(len(A), dtype=bool)
    chunk = 512
    for s in range(0, len(A), chunk):
        d2 = np.sum((A[s:s+chunk, None, :] - B[None, :, :]) ** 2, axis=2)
        mask[s:s+chunk] = (d2 < cutoff ** 2).any(axis=1)
    return mask


@torch.no_grad()
def per_residue_repr(model, batch_converter, seq, device, layer):
    """单条序列的 per-residue ESM 表示，(L, D)。"""
    seq = seq[:MAX_TOKENS]
    _, _, toks = batch_converter([("x", seq)])
    toks = toks.to(device)
    out = model(toks, repr_layers=[layer], return_contacts=False)
    rep = out["representations"][layer][0, 1:len(seq) + 1].cpu().numpy()
    return rep  # (L, D)


def pooled_interface_emb(model, bc, device, layer, residues, mask):
    """residues=[(aa,ca)], mask=bool(len). 返回界面残基的均值表示，无界面则整链均值。"""
    if len(residues) == 0:
        return None
    seq = "".join(aa for aa, _ in residues)
    rep = per_residue_repr(model, bc, seq, device, layer)  # (L, D)
    m = mask[:rep.shape[0]]  # 截断安全
    if m.sum() == 0:
        return rep.mean(axis=0)
    return rep[m].mean(axis=0)


def interface_sum_count(model, bc, device, layer, residues, mask):
    """
    返回 (界面残基表示之和, 界面残基数, 整链均值)。
    拆成 sum/count 是为了能跨链正确合并——直接平均两条链的均值会
    给残基少的链过高权重。
    """
    if len(residues) == 0:
        return None, 0, None
    seq = "".join(aa for aa, _ in residues)
    rep = per_residue_repr(model, bc, seq, device, layer)
    m = mask[:rep.shape[0]]
    whole = rep.mean(axis=0)
    if m.sum() == 0:
        return None, 0, whole
    return rep[m].sum(axis=0), int(m.sum()), whole


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="esm2_t33_650M_UR50D")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 条（调试）")
    parser.add_argument("--split-hl", action="store_true",
                        help="H/L 链分别过 ESM（消除人为接缝）。额外输出 "
                             "paratope_h/l_embeddings，并把合并后的 paratope "
                             "按界面残基数加权，维度与默认模式一致")
    parser.add_argument("--out", type=str, default=None,
                        help="输出 npz 路径，默认 processed/esm2_interface_embeddings.npz")
    args = parser.parse_args()

    out_npz = Path(args.out) if args.out else OUTPUT_NPZ

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, alphabet = esm.pretrained.load_model_and_alphabet(args.model)
    layer = model.num_layers
    model = model.to(device).eval()
    bc = alphabet.get_batch_converter()
    print(f"Loaded {args.model} (repr layer {layer})")

    df = pd.read_csv(LABELED_CSV, low_memory=False)
    if args.limit:
        df = df.head(args.limit)
    print(f"{len(df)} instances")

    instances, para, epi, npara, nepi, status = [], [], [], [], [], []
    para_h, para_l = [], []
    D = model.embed_dim
    zero = np.zeros(D, dtype=np.float32)

    for i, row in df.reset_index(drop=True).iterrows():
        if i % 50 == 0:
            print(f"  {i}/{len(df)}")
        inst = row["instance"]
        cif = CIF_DIR / f"{inst}.cif"
        instances.append(inst)
        if not cif.exists():
            para.append(zero); epi.append(zero); npara.append(0); nepi.append(0)
            para_h.append(zero); para_l.append(zero)
            status.append("missing_cif"); continue
        try:
            st = PARSER.get_structure(inst, str(cif))
            h_ids = parse_chain_set(row["h_chain"])
            l_ids = parse_chain_set(row["l_chain"])
            ag_ids = parse_chain_set(row["antigen_chains"])
            ag_res = chain_residues(st, ag_ids)
            ag_ca = [c for _, c in ag_res]

            if args.split_hl:
                # --- H / L 各自独立过 ESM，不制造人为接缝 ---
                h_res = chain_residues(st, h_ids)
                l_res = chain_residues(st, l_ids)
                h_mask = interface_mask([c for _, c in h_res], ag_ca)
                l_mask = interface_mask([c for _, c in l_res], ag_ca)
                # 抗原侧的 partner 是 H+L 的全部坐标
                ab_ca_all = [c for _, c in h_res] + [c for _, c in l_res]
                ag_mask = interface_mask(ag_ca, ab_ca_all)

                h_sum, h_n, h_whole = interface_sum_count(
                    model, bc, device, layer, h_res, h_mask)
                l_sum, l_n, l_whole = interface_sum_count(
                    model, bc, device, layer, l_res, l_mask)

                # 合并 paratope：按界面残基数加权（等价于对两条链所有界面
                # 残基取一次平均），维度仍是 D，可与默认模式直接对比
                if h_n + l_n > 0:
                    tot = np.zeros(D, dtype=np.float64)
                    if h_n:
                        tot += h_sum
                    if l_n:
                        tot += l_sum
                    p = (tot / (h_n + l_n)).astype(np.float32)
                else:
                    wholes = [w for w in (h_whole, l_whole) if w is not None]
                    p = np.mean(wholes, axis=0).astype(np.float32) if wholes else zero

                ph = (h_sum / h_n).astype(np.float32) if h_n else (
                    h_whole.astype(np.float32) if h_whole is not None else zero)
                pl = (l_sum / l_n).astype(np.float32) if l_n else (
                    l_whole.astype(np.float32) if l_whole is not None else zero)
                n_para = h_n + l_n
            else:
                # --- 默认：H+L 串成一条序列（原始实现）---
                ab_res = chain_residues(st, h_ids + l_ids)
                ab_ca = [c for _, c in ab_res]
                ab_mask = interface_mask(ab_ca, ag_ca)
                ag_mask = interface_mask(ag_ca, ab_ca)
                p = pooled_interface_emb(model, bc, device, layer, ab_res, ab_mask)
                p = p if p is not None else zero
                ph = pl = zero
                n_para = int(ab_mask.sum())

            e = pooled_interface_emb(model, bc, device, layer, ag_res, ag_mask)
            para.append(p)
            para_h.append(ph); para_l.append(pl)
            epi.append(e if e is not None else zero)
            npara.append(n_para); nepi.append(int(ag_mask.sum()))
            status.append("ok")
        except Exception as ex:
            para.append(zero); epi.append(zero); npara.append(0); nepi.append(0)
            para_h.append(zero); para_l.append(zero)
            status.append(f"error: {type(ex).__name__}")

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(
        instances=np.array(instances, dtype=str),
        paratope_embeddings=np.vstack(para).astype(np.float32),
        epitope_embeddings=np.vstack(epi).astype(np.float32),
        n_paratope=np.array(npara), n_epitope=np.array(nepi),
        status=np.array(status, dtype=str),
    )
    if args.split_hl:
        arrays["paratope_h_embeddings"] = np.vstack(para_h).astype(np.float32)
        arrays["paratope_l_embeddings"] = np.vstack(para_l).astype(np.float32)
    np.savez(out_npz, **arrays)

    ok = sum(s == "ok" for s in status)
    mode = "split-HL" if args.split_hl else "concatenated-HL"
    print(f"\nSaved {out_npz}  mode={mode}  ok={ok}/{len(status)}  dim={D}")
    print(f"paratope residues: mean={np.mean(npara):.1f}  epitope: mean={np.mean(nepi):.1f}")
    print(f"arrays: {sorted(arrays)}")


if __name__ == "__main__":
    main()
