#!/usr/bin/env python3
"""
Step 8: 对消融结果做不确定性量化（bootstrap 置信区间 + 配对显著性检验）

动机：
    Test 集只有 150 个样本，Spearman 的抽样波动约 ±0.14。
    在这个量级下，消融表里大部分「排序」其实落在噪声里。
    本脚本把这件事量化出来，避免把噪声当成结论。

方法：
    - 对每个配置的 test 预测做 bootstrap 重采样，给出 Spearman 的 95% CI
    - 配对检验：同一批测试样本上重采样，计算 Δ = Spearman(B) - Spearman(A)，
      看 95% CI 是否跨 0（配对比独立比较更有功效，因为消除了样本难度差异）

运行：
    conda activate aidd
    python scripts/08_significance_test.py
    python scripts/08_significance_test.py --ref mean_pooled --n-boot 8000

输出：
    - reports/significance_table.md
    - 控制台表格
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = PROJECT_ROOT / "processed"
OUT_MD = PROJECT_ROOT / "reports" / "significance_table.md"

# 配置名 -> (显示名, 预测文件名)
CONFIGS = {
    "struct_only":      ("仅结构特征",              "pred_struct_only_xgb.csv"),
    "interface_only":   ("界面池化 (H/L 串接)",      "pred_interface_only_xgb.csv"),
    "iface_nojunction": ("界面池化 (无接缝)",        "pred_iface_nojunction_xgb.csv"),
    "iface_hl_separate":("界面池化 (H/L 分开)",      "pred_iface_hl_separate_xgb.csv"),
    "interface_plus_struct": ("界面池化 + 结构特征",  "pred_interface_plus_struct_xgb.csv"),
    "mean_pooled":      ("ESM-2 整链池化",           "pred_mean_pooled_xgb.csv"),
    "mean_plus_struct": ("整链池化 + 结构特征",       "pred_mean_plus_struct_xgb.csv"),
    "fused":            ("整链 + 界面 融合",          "pred_fused_xgb.csv"),
    "fused_split":      ("整链 + 界面(无接缝) 融合",  "pred_fused_split_xgb.csv"),
}


def load_test(fname):
    p = PRED_DIR / fname
    if not p.exists():
        return None
    d = pd.read_csv(p)
    return d[d["split"] == "test"][["instance", "pkd_true", "pkd_pred"]].copy()


def spearman(y, p):
    return stats.spearmanr(y, p)[0]


def boot_ci(d, n_boot, rng):
    obs = spearman(d.pkd_true, d.pkd_pred)
    yt, yp = d.pkd_true.values, d.pkd_pred.values
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(d), len(d))
        s = spearman(yt[i], yp[i])
        if not np.isnan(s):
            vals.append(s)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return obs, lo, hi


def paired_diff(a, b, n_boot, rng):
    """Δ = Spearman(b) - Spearman(a)，在同一批 instance 上配对重采样。"""
    m = a.merge(b, on="instance", suffixes=("_a", "_b"))
    ya, pa = m.pkd_true_a.values, m.pkd_pred_a.values
    yb, pb = m.pkd_true_b.values, m.pkd_pred_b.values
    ds = []
    for _ in range(n_boot):
        i = rng.integers(0, len(m), len(m))
        sa, sb = spearman(ya[i], pa[i]), spearman(yb[i], pb[i])
        if not (np.isnan(sa) or np.isnan(sb)):
            ds.append(sb - sa)
    ds = np.array(ds)
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return ds.mean(), lo, hi, float((ds > 0).mean()), len(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="mean_pooled",
                    help="配对检验的参照配置")
    ap.add_argument("--n-boot", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    loaded = {k: load_test(f) for k, (_, f) in CONFIGS.items()}
    missing = [k for k, v in loaded.items() if v is None]
    avail = {k: v for k, v in loaded.items() if v is not None}
    if missing:
        print(f"跳过（预测文件不存在）: {missing}\n"
              f"  提示：先跑 scripts/07_run_ablations.py 生成基础配置\n")
    if args.ref not in avail:
        ap.error(f"参照配置 {args.ref} 的预测文件不存在")

    n_test = len(next(iter(avail.values())))
    # Spearman 标准误的经验近似（Fisher z）
    approx_se = 1.06 / np.sqrt(max(n_test - 3, 1))
    resolution = 1.96 * approx_se

    rows = []
    for k, d in avail.items():
        obs, lo, hi = boot_ci(d, args.n_boot, rng)
        rows.append((k, CONFIGS[k][0], obs, lo, hi))
    rows.sort(key=lambda r: r[2])

    print(f"\n{'配置':30s} {'Spearman':>9s}   {'95% CI':>18s}")
    print("-" * 64)
    for _, name, obs, lo, hi in rows:
        print(f"{name:30s} {obs:9.3f}   [{lo:+.3f}, {hi:+.3f}]")
    print(f"\nn_test = {n_test}   →  Spearman 分辨极限 ≈ ±{resolution:.2f}")

    ref_d = avail[args.ref]
    ref_name = CONFIGS[args.ref][0]
    print(f"\n配对检验（参照 = {ref_name}）")
    print("-" * 64)
    pairs = []
    for k, d in avail.items():
        if k == args.ref:
            continue
        dm, lo, hi, pgt, n = paired_diff(ref_d, d, args.n_boot, rng)
        sig = "显著" if (lo > 0 or hi < 0) else "不显著"
        pairs.append((k, CONFIGS[k][0], dm, lo, hi, pgt, sig))
        print(f"{CONFIGS[k][0]:30s} Δ={dm:+.3f}  CI=[{lo:+.3f},{hi:+.3f}]  {sig}")

    n_sig = sum(1 for p in pairs if p[6] == "显著")
    print(f"\n结论：{len(pairs)} 组对比中 {n_sig} 组显著。")

    # ---- Markdown 输出 ----
    md = [
        f"## 不确定性量化（bootstrap, n_boot={args.n_boot}, n_test={n_test}）", "",
        f"Test 集仅 {n_test} 个样本，Spearman 的分辨极限约 **±{resolution:.2f}**。",
        "低于这个量级的差异不能当作结论。", "",
        "| 配置 | Test Spearman | 95% CI |", "| --- | ---: | :---: |",
    ]
    for _, name, obs, lo, hi in rows:
        md.append(f"| {name} | {obs:.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    md += ["", f"### 配对显著性检验（参照 = {ref_name}）", "",
           "| 对比配置 | Δ Spearman | 95% CI | 结论 |", "| --- | ---: | :---: | :---: |"]
    for _, name, dm, lo, hi, pgt, sig in pairs:
        md.append(f"| {name} | {dm:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {sig} |")
    md += ["", f"**{len(pairs)} 组对比中 {n_sig} 组显著。**"]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nMarkdown -> {OUT_MD}")


if __name__ == "__main__":
    main()
