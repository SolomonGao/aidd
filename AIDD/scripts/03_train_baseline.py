#!/usr/bin/env python3
"""
Step 3/4: 特征工程 + XGBoost 基线模型

输入：
    - processed/sabdab2_labeled_dataset.csv
    - processed/structural_features.csv

输出：
    - models/baseline_xgb.json
    - processed/baseline_predictions.csv
    - reports/baseline_performance.png
    - 控制台输出 R² / Pearson / Spearman / MAE
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELED_CSV = PROJECT_ROOT / "processed" / "sabdab2_labeled_dataset.csv"
STRUCT_CSV = PROJECT_ROOT / "processed" / "structural_features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_xgb.json"
PRED_CSV = PROJECT_ROOT / "processed" / "baseline_predictions.csv"
PLOT_PATH = PROJECT_ROOT / "reports" / "baseline_performance.png"

# 氨基酸物理化学分组
POS_CHARGED = {"K", "R"}
NEG_CHARGED = {"D", "E"}
HYDROPHOBIC = {"A", "I", "L", "M", "F", "V", "W", "Y"}
AROMATIC = {"F", "W", "Y"}
POLAR = {"S", "T", "N", "Q"}


def seq_fraction(seq, charset):
    if pd.isna(seq) or len(seq) == 0:
        return 0.0
    return sum(1 for aa in seq if aa in charset) / len(seq)


def net_charge(seq):
    if pd.isna(seq) or len(seq) == 0:
        return 0.0
    return sum(1 for aa in seq if aa in POS_CHARGED) - sum(1 for aa in seq if aa in NEG_CHARGED)


def parse_chain_set(value):
    if pd.isna(value) or str(value).strip() in ("", "NA", "+"):
        return 0
    return len([p for p in re.split(r"[|/]", str(value)) if p.strip() and p.strip() != "+"])


def build_feature_matrix(df):
    """从合并后的数据框构建特征矩阵。"""
    feats = pd.DataFrame({"instance": df["instance"]})

    # ---- 序列长度特征 ----
    feats["h_len"] = df["h_seq"].fillna("").apply(len)
    feats["l_len"] = df["l_seq"].fillna("").apply(len)
    feats["ag_len"] = df["antigen_seq"].fillna("").apply(len)
    feats["n_antigen_chains"] = df["antigen_chains"].apply(parse_chain_set)

    # ---- CDR 长度特征 ----
    for region in ["cdrh1", "cdrh2", "cdrh3", "cdrl1", "cdrl2", "cdrl3"]:
        feats[f"{region}_len"] = df[region].fillna("").apply(len)
    feats["cdr_total_len"] = (
        feats["cdrh1_len"] + feats["cdrh2_len"] + feats["cdrh3_len"] +
        feats["cdrl1_len"] + feats["cdrl2_len"] + feats["cdrl3_len"]
    )

    # ---- CDR-H3 理化特征 ----
    feats["cdrh3_charge"] = df["cdrh3"].apply(net_charge)
    feats["cdrh3_hydrophobic"] = df["cdrh3"].apply(lambda s: seq_fraction(s, HYDROPHOBIC))
    feats["cdrh3_aromatic"] = df["cdrh3"].apply(lambda s: seq_fraction(s, AROMATIC))
    feats["cdrh3_polar"] = df["cdrh3"].apply(lambda s: seq_fraction(s, POLAR))

    # ---- 结构特征 ----
    struct_cols = [
        "h_residues", "l_residues", "ag_residues", "ab_residues",
        "h_atoms", "l_atoms", "ag_atoms",
        "interface_residues_ab", "interface_residues_ag",
        "interface_contacts_ca_8A", "interface_contacts_heavy_5A",
        "vh_vl_distance",
    ]
    for col in struct_cols:
        feats[col] = df[col]

    # ---- 元数据特征 ----
    # 注意：holo 只表示结构里是否有抗原，和亲和力标签来源相关，
    # 作为特征容易让模型学到数据伪影，先不放进去。
    feats["resolution"] = df["resolution"]
    feats["label_confidence"] = df["label_confidence"]
    feats["antibody_type"] = df["antibody_type"]
    feats["method"] = df["method"].fillna("UNKNOWN")

    return feats


def main():
    print("Loading datasets...")
    labels = pd.read_csv(LABELED_CSV, low_memory=False)
    struct = pd.read_csv(STRUCT_CSV, low_memory=False)
    df = labels.merge(struct, on="instance", how="inner", suffixes=("", "_struct"))
    print(f"Merged rows: {len(df)}")

    # 去掉没有标签或没有结构特征的样本
    df = df[df["pkd"].notna()].copy()
    print(f"Rows with valid pKD: {len(df)}")

    feats = build_feature_matrix(df)
    y = df["pkd"].values

    # 训练/测试按 SAbDab2 提供的 split
    train_mask = df["split"] == "train"
    test_mask = df["split"] == "test"

    X_train = feats[train_mask]
    X_test = feats[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    # 定义特征组
    numeric_cols = [c for c in feats.columns if c not in ["instance", "antibody_type", "method", "label_confidence"]]
    categorical_cols = ["antibody_type", "method", "label_confidence"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("xgb", XGBRegressor(
                n_estimators=1000,
                max_depth=3,
                learning_rate=0.02,
                subsample=0.6,
                colsample_bytree=0.6,
                random_state=42,
                n_jobs=4,
                reg_alpha=1.0,
                reg_lambda=2.0,
                early_stopping_rounds=30,
            )),
        ]
    )

    print("\nTraining XGBoost baseline with early stopping...")
    # 从训练集里再分出 validation 用于 early stopping
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
    X_tr_proc = model.named_steps["preprocess"].fit_transform(X_tr)
    X_val_proc = model.named_steps["preprocess"].transform(X_val)
    model.named_steps["xgb"].fit(
        X_tr_proc, y_tr,
        eval_set=[(X_val_proc, y_val)],
        verbose=False,
    )

    # 保存模型
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.named_steps["xgb"].save_model(str(MODEL_PATH))
    print(f"Model saved to {MODEL_PATH}")

    # 预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    def report(y_true, y_pred, split_name):
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        pearson = stats.pearsonr(y_true, y_pred)[0]
        spearman = stats.spearmanr(y_true, y_pred)[0]
        print(f"\n{split_name} set:")
        print(f"  R²      = {r2:.3f}")
        print(f"  MAE     = {mae:.3f}")
        print(f"  Pearson = {pearson:.3f}")
        print(f"  Spearman= {spearman:.3f}")
        return {"r2": r2, "mae": mae, "pearson": pearson, "spearman": spearman}

    report(y_train, y_train_pred, "Train")
    report(y_test, y_test_pred, "Test")

    # 保存预测结果
    pred_df = pd.DataFrame({
        "instance": df["instance"],
        "pdb_id": df["pdb_id"],
        "split": df["split"],
        "label_confidence": df["label_confidence"],
        "pkd_true": y,
        "pkd_pred": np.concatenate([y_train_pred, y_test_pred]),
    })
    pred_df.to_csv(PRED_CSV, index=False)
    print(f"\nPredictions saved to {PRED_CSV}")

    # 画图
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=pred_df[pred_df["split"] == "test"], x="pkd_true", y="pkd_pred", alpha=0.6, hue="label_confidence")
    lims = [pred_df["pkd_true"].min() - 0.5, pred_df["pkd_true"].max() + 0.5]
    plt.plot(lims, lims, "k--", alpha=0.5)
    plt.xlabel("True pKD")
    plt.ylabel("Predicted pKD")
    plt.title("XGBoost Baseline: Test Set")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"Plot saved to {PLOT_PATH}")

    # 特征重要性
    feature_names = (
        numeric_cols +
        list(model.named_steps["preprocess"].named_transformers_["cat"].get_feature_names_out(categorical_cols))
    )
    importances = model.named_steps["xgb"].feature_importances_
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False)
    print("\nTop 15 important features:")
    print(imp_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
