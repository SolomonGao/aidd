#!/usr/bin/env python3
"""
Step 7: 一键复现全部消融实验，输出对比表

背景：
    早期这些消融是零散跑的（有的靠脚本 05 换参数，有的是一次性代码），
    结果无法完整复现。本脚本把所有配置固化下来，一条命令跑完并生成表格。

运行：
    conda activate aidd
    python scripts/07_run_ablations.py                 # 全部配置
    python scripts/07_run_ablations.py --model ridge   # 换模型
    python scripts/07_run_ablations.py --only struct_only mean_pooled

输出：
    - reports/ablations/<config>.json   每个配置的完整指标
    - reports/ablation_table.md         Markdown 对比表（可直接贴进 README）
    - 控制台表格
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINER = PROJECT_ROOT / "scripts" / "05_train_esm_xgb.py"
MEAN_NPZ = PROJECT_ROOT / "processed" / "esm2_650m_embeddings.npz"
IFACE_NPZ = PROJECT_ROOT / "processed" / "esm2_interface_embeddings.npz"
OUT_DIR = PROJECT_ROOT / "reports" / "ablations"
TABLE_MD = PROJECT_ROOT / "reports" / "ablation_table.md"

# 每个消融配置：name -> (中文描述, 传给脚本 05 的额外参数)
CONFIGS = {
    "struct_only": (
        "仅结构特征（无嵌入）",
        ["--no-embeddings", "--add-struct-features"],
    ),
    "interface_only": (
        "仅界面池化 (paratope+epitope)",
        ["--emb-npz", str(IFACE_NPZ),
         "--feature-keys", "paratope_embeddings", "epitope_embeddings"],
    ),
    "mean_pooled": (
        "ESM-2 整链池化 (VH+VL+抗原)",
        ["--emb-npz", str(MEAN_NPZ), "--feature-keys", "combined_embeddings"],
    ),
    "mean_plus_struct": (
        "整链池化 + 结构特征",
        ["--emb-npz", str(MEAN_NPZ), "--feature-keys", "combined_embeddings",
         "--add-struct-features"],
    ),
    "interface_plus_struct": (
        "界面池化 + 结构特征",
        ["--emb-npz", str(IFACE_NPZ),
         "--feature-keys", "paratope_embeddings", "epitope_embeddings",
         "--add-struct-features"],
    ),
    "fused": (
        "整链池化 + 界面池化（融合）",
        ["--emb-npz", str(MEAN_NPZ), str(IFACE_NPZ),
         "--feature-keys", "combined_embeddings",
         "paratope_embeddings", "epitope_embeddings"],
    ),
}


def run_one(name, desc, extra, model, pca):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / f"{name}_{model}.json"
    cmd = [sys.executable, str(TRAINER), "--model", model, "--pca", str(pca),
           "--tag", name, "--metrics-json", str(json_path)] + extra
    print(f"\n{'='*72}\n▶ {name}  —  {desc}\n{'='*72}")
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if res.returncode != 0:
        print(f"  !! {name} FAILED (exit {res.returncode})")
        return None
    with open(json_path) as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["xgb", "ridge"], default="xgb")
    ap.add_argument("--pca", type=int, default=50)
    ap.add_argument("--only", nargs="+", default=None,
                    help="只跑指定配置名")
    args = ap.parse_args()

    names = args.only or list(CONFIGS)
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        ap.error(f"未知配置: {unknown}\n可选: {list(CONFIGS)}")

    results = []
    for name in names:
        desc, extra = CONFIGS[name]
        r = run_one(name, desc, extra, args.model, args.pca)
        if r:
            r["desc"] = desc
            r["name"] = name
            results.append(r)

    if not results:
        print("\n没有成功的配置。")
        return

    # 按 Test Spearman 升序，最优在最后一行（和 README 表格一致）
    results.sort(key=lambda r: r["test"]["Spearman"])
    best = max(results, key=lambda r: r["test"]["Spearman"])

    header = (f"| 特征表示 | 特征数 | Test Spearman | Test Pearson | Test R² | "
              f"GroupCV Spearman |")
    sep = "| --- | ---: | ---: | ---: | ---: | ---: |"
    lines = [
        f"消融实验（{args.model.upper()}, PCA={args.pca}, 抗原聚类划分, "
        f"train {results[0]['n_train']} / test {results[0]['n_test']}）", "",
        header, sep,
    ]
    for r in results:
        mark = "**" if r is best else ""
        lines.append(
            f"| {mark}{r['desc']}{mark} | {r['n_features']} | "
            f"{mark}{r['test']['Spearman']:.3f}{mark} | "
            f"{r['test']['Pearson']:.3f} | {r['test']['R2']:.3f} | "
            f"{r['groupcv']['Spearman']:.3f} |")

    table = "\n".join(lines)
    TABLE_MD.parent.mkdir(parents=True, exist_ok=True)
    TABLE_MD.write_text(table + "\n", encoding="utf-8")

    print("\n\n" + "="*72)
    print(table)
    print("="*72)
    print(f"\nMarkdown table -> {TABLE_MD}")
    print(f"Per-config JSON -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
