#!/usr/bin/env python3
"""
Step 5 (part 2): 用 ESM-2 嵌入（可选结构特征）训练亲和力回归模型

修复 & 改进（相对旧版）：
    - 修复了预测 CSV 的行错位 bug（pkd_pred 与 pkd_true / instance 现在对齐）
    - 去重：同一 (VH, VL, 抗原) 三元组只保留一行，避免同一复合物多份拷贝
      同时进入 train 造成的虚高与 train/test 泄漏
    - 全程报告 Spearman（排序相关）——绝对 pKD 校准在本任务上不现实，排序才是重点
    - 抗原分组交叉验证（GroupKFold，group = 抗原序列），给出诚实的泛化估计
    - 支持任意嵌入 npz（mean-pooled 或 interface-pooled）与线性/树模型对比

运行：
    conda activate aidd
    python scripts/05_train_esm_xgb.py                       # mean-pooled combined + Ridge
    python scripts/05_train_esm_xgb.py --model xgb
    python scripts/05_train_esm_xgb.py \
        --emb-npz processed/esm2_interface_embeddings.npz \
        --feature-keys paratope_embeddings epitope_embeddings
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EMB_NPZ = PROJECT_ROOT / "processed" / "esm2_650m_embeddings.npz"
LABELED_CSV = PROJECT_ROOT / "processed" / "sabdab2_labeled_dataset.csv"
STRUCT_CSV = PROJECT_ROOT / "processed" / "structural_features.csv"

STRUCT_COLS = [
    "h_residues", "l_residues", "ag_residues", "ab_residues",
    "h_atoms", "l_atoms", "ag_atoms",
    "interface_residues_ab", "interface_residues_ag",
    "interface_contacts_ca_8A", "interface_contacts_heavy_5A",
    "vh_vl_distance", "resolution",
]

# 用于去重和分组 CV 的键
SEQ_KEYS = ["vh_numerable_seq", "vl_numerable_seq", "antigen_seq"]
GROUP_KEY = "antigen_seq"  # 抗原感知分组：同一抗原不同时出现在 CV 的 train/val


def metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "Pearson": stats.pearsonr(y_true, y_pred)[0],
        "Spearman": stats.spearmanr(y_true, y_pred)[0],
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def fmt(name, m, n):
    return (f"{name:12s} n={n:4d}  R2={m['R2']:6.3f}  "
            f"Pearson={m['Pearson']:.3f}  Spearman={m['Spearman']:.3f}  "
            f"MAE={m['MAE']:.3f}")


def build_model(kind, n_pca):
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if n_pca:
        steps.append(("pca", PCA(n_components=n_pca, random_state=0)))
    if kind == "ridge":
        steps.append(("reg", Ridge(alpha=10.0)))
    else:
        from xgboost import XGBRegressor
        steps.append(("reg", XGBRegressor(
            n_estimators=400, max_depth=3, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.7, reg_alpha=0.5,
            reg_lambda=2.0, random_state=42, n_jobs=4)))
    return Pipeline(steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb-npz", type=str, nargs="+", default=[str(DEFAULT_EMB_NPZ)],
                        help="一个或多个嵌入 npz；按 instance 对齐后拼接")
    parser.add_argument("--feature-keys", type=str, nargs="+",
                        default=["combined_embeddings"],
                        help="要使用的特征数组名（会在所有 npz 里查找）")
    parser.add_argument("--model", choices=["ridge", "xgb"], default="ridge")
    parser.add_argument("--pca", type=int, default=50,
                        help="PCA 维数（0 表示不降维）")
    parser.add_argument("--add-struct-features", action="store_true")
    parser.add_argument("--only-chain-matched", action="store_true")
    parser.add_argument("--no-dedup", action="store_true",
                        help="不去重（用于复现旧行为，不推荐）")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--tag", type=str, default=None,
                        help="输出文件名后缀，默认从 emb-npz 推断")
    parser.add_argument("--no-embeddings", action="store_true",
                        help="不使用任何 ESM 嵌入（配合 --add-struct-features 做纯结构特征消融）")
    parser.add_argument("--metrics-json", type=str, default=None,
                        help="把本次运行的指标写成 JSON，供消融汇总脚本读取")
    args = parser.parse_args()

    if args.no_embeddings and not args.add_struct_features:
        parser.error("--no-embeddings 必须配合 --add-struct-features，否则没有任何特征")

    if args.no_embeddings:
        tag = args.tag or "structonly"
    else:
        tag = args.tag or "+".join(
            Path(p).stem.replace("esm2_", "").replace("_embeddings", "") for p in args.emb_npz)
    pred_csv = PROJECT_ROOT / "processed" / f"pred_{tag}_{args.model}.csv"
    plot_path = PROJECT_ROOT / "reports" / f"perf_{tag}_{args.model}.png"

    # --- 载入标签 ---
    labels = pd.read_csv(LABELED_CSV, low_memory=False)
    if args.only_chain_matched:
        labels = labels[labels["label_confidence"] == "chain_matched"].copy()
        print(f"Using only chain_matched labels: {len(labels)} rows")

    # --- 载入嵌入（可多个 npz，按 instance 对齐拼接）---
    if args.no_embeddings:
        print("No embeddings (structural-features-only ablation)")
        emb_cols = []
        df = labels.reset_index(drop=True)
    else:
        print(f"Loading embeddings from {args.emb_npz}  keys={args.feature_keys}")
        npzs = [np.load(p, allow_pickle=True) for p in args.emb_npz]
        insts = [z["instances"].astype(str) for z in npzs]
        parts = []
        for key in args.feature_keys:
            for z, inst in zip(npzs, insts):
                if key in z.files:
                    d = pd.DataFrame(z[key], index=inst)
                    d.columns = [f"{key}_{i}" for i in range(d.shape[1])]
                    d.index.name = "instance"
                    parts.append(d)
                    break
            else:
                raise KeyError(f"feature key '{key}' not found in any of {args.emb_npz}")
        emb_df = parts[0]
        for p in parts[1:]:
            emb_df = emb_df.join(p, how="inner")
        emb_cols = list(emb_df.columns)
        print(f"  embedding matrix: {emb_df.shape}")
        df = labels.merge(emb_df, on="instance", how="inner").reset_index(drop=True)
    print(f"Rows after merging embeddings: {len(df)}")

    # --- 去重：同一 (VH,VL,AG) 只保留一行 ---
    if not args.no_dedup:
        before = len(df)
        df = df.drop_duplicates(subset=SEQ_KEYS).reset_index(drop=True)
        print(f"Dedup by {SEQ_KEYS}: {before} -> {len(df)} rows")

    feature_cols = list(emb_cols)
    if args.add_struct_features:
        struct = pd.read_csv(STRUCT_CSV, low_memory=False)
        # resolution 在标签表里，不在结构特征表里；只合并结构表实际有的列
        struct_cols = [c for c in STRUCT_COLS if c in struct.columns]
        df = df.merge(struct[["instance"] + struct_cols], on="instance", how="left")
        extra = [c for c in STRUCT_COLS if c in df.columns and c not in feature_cols]
        feature_cols += extra
        print(f"Added {len(extra)} structural features")

    X = df[feature_cols].values
    y = df["pkd"].values
    groups = df[GROUP_KEY].fillna("NA").values

    train_mask = (df["split"] == "train").values
    test_mask = (df["split"] == "test").values

    # PCA 维数不能超过特征数（结构特征只有十几维）
    n_pca = args.pca
    if n_pca and n_pca >= X.shape[1]:
        print(f"PCA disabled: n_components={n_pca} >= n_features={X.shape[1]}")
        n_pca = 0

    # --- 抗原分组交叉验证（只在 train 内做，诚实估计泛化）---
    X_train, y_train = X[train_mask], y[train_mask]
    g_train = groups[train_mask]
    n_folds = min(args.cv_folds, len(np.unique(g_train)))
    gkf = GroupKFold(n_splits=n_folds)
    cv_true, cv_pred = [], []
    for tr_idx, va_idx in gkf.split(X_train, y_train, g_train):
        m = build_model(args.model, n_pca)
        m.fit(X_train[tr_idx], y_train[tr_idx])
        cv_pred.append(m.predict(X_train[va_idx]))
        cv_true.append(y_train[va_idx])
    cv_true = np.concatenate(cv_true)
    cv_pred = np.concatenate(cv_pred)

    # --- 用全部 train 拟合最终模型，评估 held-out test ---
    model = build_model(args.model, n_pca)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X[test_mask])

    m_train = metrics(y_train, y_train_pred)
    m_cv = metrics(cv_true, cv_pred)
    m_test = metrics(y[test_mask], y_test_pred)

    print("\n=== Metrics (antigen-cluster test split) ===")
    print(fmt("Train", m_train, train_mask.sum()))
    print(fmt("GroupCV", m_cv, len(cv_true)))
    print(fmt("Test", m_test, test_mask.sum()))

    if args.metrics_json:
        import json
        Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_json, "w") as fh:
            json.dump({
                "tag": tag, "model": args.model, "pca": n_pca,
                "n_features": int(X.shape[1]),
                "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
                "train": m_train, "groupcv": m_cv, "test": m_test,
            }, fh, indent=2)
        print(f"Metrics JSON saved to {args.metrics_json}")

    # --- 正确对齐地保存预测（按 df 原顺序回填）---
    pred_full = np.empty(len(df))
    pred_full[train_mask] = y_train_pred
    pred_full[test_mask] = y_test_pred
    pred_df = pd.DataFrame({
        "instance": df["instance"].values,
        "pdb_id": df["pdb_id"].values,
        "split": df["split"].values,
        "label_confidence": df["label_confidence"].values,
        "pkd_true": y,
        "pkd_pred": pred_full,
    })
    pred_csv.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(pred_csv, index=False)
    print(f"\nPredictions saved to {pred_csv}")

    # --- 图 ---
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    test_df = pred_df[pred_df["split"] == "test"]
    plt.figure(figsize=(6, 6))
    plt.scatter(test_df["pkd_true"], test_df["pkd_pred"], alpha=0.6, s=18)
    lims = [y.min() - 0.5, y.max() + 0.5]
    plt.plot(lims, lims, "k--", alpha=0.5)
    plt.xlabel("True pKD"); plt.ylabel("Predicted pKD")
    m = metrics(test_df["pkd_true"].values, test_df["pkd_pred"].values)
    plt.title(f"{tag} + {args.model}  Test Spearman={m['Spearman']:.2f}")
    plt.tight_layout(); plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
