#!/usr/bin/env python3
"""
Step 5 (part 1): 用 ESM-2 650M 提取 VH / VL / 抗原序列嵌入

运行环境：
    conda activate aidd
    python scripts/04_extract_esm2_embeddings.py

输出：
    - processed/esm2_650m_embeddings.npz
      包含：
        - instances: list[str]
        - vh_embeddings: np.ndarray, shape (N, 1280)
        - vl_embeddings: np.ndarray, shape (N, 1280)
        - ag_embeddings: np.ndarray, shape (N, 1280)
        - combined_embeddings: np.ndarray, shape (N, 3840)

说明：
    - 首次运行会自动下载 ESM-2 650M 模型（约 2.4 GB）
    - 默认 batch_size=16，RTX 5070 Ti 16GB 显存可以跑
    - 抗原序列如果超过 1022 个残基会被截断
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import esm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELED_CSV = PROJECT_ROOT / "processed" / "sabdab2_labeled_dataset.csv"
OUTPUT_NPZ = PROJECT_ROOT / "processed" / "esm2_650m_embeddings.npz"

MAX_TOKENS = 1022  # ESM-2 最大长度，减去 <cls> 和 <eos>


def clean_seq(seq):
    if pd.isna(seq):
        return ""
    seq = str(seq).strip()
    # 去掉非标准字符，保留 20 种氨基酸和 X
    seq = "".join(c for c in seq if c.isalpha())
    return seq.upper()


def extract_embeddings(model, batch_converter, alphabet, sequences, device, batch_size=16):
    """对一组序列提取 mean-pooled 嵌入。"""
    embeddings = []
    model.eval()

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i : i + batch_size]
        # ESM batch_converter 需要 (name, seq) 列表
        data = [(f"seq_{j}", seq[:MAX_TOKENS]) for j, seq in enumerate(batch_seqs)]
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]  # (B, L, 1280)

        for j, seq in enumerate(batch_seqs):
            L = min(len(seq), MAX_TOKENS)
            # 位置 0 是 <cls>，1..L 是序列，L+1 是 <eos>
            emb = token_representations[j, 1 : L + 1].mean(dim=0).cpu().numpy()
            embeddings.append(emb)

    return np.vstack(embeddings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model", type=str, default="esm2_t33_650M_UR50D",
                        help="ESM-2 model name, e.g. esm2_t33_650M_UR50D, esm2_t12_85M_UR50D")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Loading ESM-2 model: {args.model} ...")
    model, alphabet = esm.pretrained.load_model_and_alphabet(args.model)
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()
    print("Model loaded.")

    df = pd.read_csv(LABELED_CSV, low_memory=False)
    print(f"Loaded {len(df)} labeled instances.")

    # 清洗序列
    vh_seqs = [clean_seq(s) for s in df["vh_numerable_seq"].fillna("")]
    vl_seqs = [clean_seq(s) for s in df["vl_numerable_seq"].fillna("")]
    # 抗原链用 '/' 分隔，拼成一条序列；空抗原用空串
    ag_seqs = []
    for seqs in df["antigen_seq"].fillna(""):
        seq = clean_seq(str(seqs).replace("/", ""))
        ag_seqs.append(seq)

    print("Extracting VH embeddings ...")
    vh_emb = extract_embeddings(model, batch_converter, alphabet, vh_seqs, device, args.batch_size)
    print("Extracting VL embeddings ...")
    vl_emb = extract_embeddings(model, batch_converter, alphabet, vl_seqs, device, args.batch_size)
    print("Extracting antigen embeddings ...")
    ag_emb = extract_embeddings(model, batch_converter, alphabet, ag_seqs, device, args.batch_size)

    combined = np.hstack([vh_emb, vl_emb, ag_emb])

    OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_NPZ,
        instances=df["instance"].values.astype(str),
        vh_embeddings=vh_emb,
        vl_embeddings=vl_emb,
        ag_embeddings=ag_emb,
        combined_embeddings=combined,
    )
    print(f"\nSaved embeddings to {OUTPUT_NPZ}")
    print(f"Shapes: VH={vh_emb.shape}, VL={vl_emb.shape}, AG={ag_emb.shape}, combined={combined.shape}")


if __name__ == "__main__":
    main()
