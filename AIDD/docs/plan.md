# AbDock-AI 实施计划（分步版）

## 当前已具备的条件
- 目录：`/home/xinggao/aidd/AIDD`
- 已有代码：
  - `data/SAbDab.py`：下载 SAbDab summary / 结构压缩包
  - `parser/AntibodyExtractor.py`：从 PDB 提取 VH/VL 链的 CA 坐标与序列
- 已有数据：
  - `sabdab_data/sabdab_summary.tsv`：约 21k 条 SAbDab 记录
  - `data/SKEMPI2_PDBs/PDBs/`：大量 PDB 与 `.mapping` 文件
- 环境：`conda env: aidd`（Python 3.9 + BioPython + pandas + numpy + requests）
- 网络可用：SKEMPI v2 CSV 可下载

## 建议分步路线

### Step 1：数据层 —— 生成干净的训练样本表 ✅ 已完成
**已升级为基于 SAbDab2 splits_final + SAbDab affinity**

实际结果：
- 脚本：`scripts/01b_build_sabdab2_dataset.py`
- 输出：`processed/sabdab2_labeled_dataset.csv`（1145 行）
- 有效亲和力标签（pKD）：1145 行，覆盖 676 个 PDB
- 训练/测试 split：947 / 198（使用 `abag_split`，抗原感知的序列相似度划分）
- 标签可信度：833 条 `chain_matched`（链级对齐） + 312 条 `pdb_only`（PDB 级对齐，略有噪声）
- CIF 结构文件覆盖率：1145 / 1145
- pKD 分布：均值 7.99，标准差 1.55，范围 3.70 ~ 12.40

旧的 SKEMPI 流程保留在 `scripts/01_build_dataset.py`，可作为突变/优化模块的补充数据。
目标：把 `sabdab_summary.tsv` 与 SKEMPI v2 亲和性数据对齐，得到 `processed/antibody_antigen_dataset.csv`
- 下载/读取 `skempi_v2.csv`
- 解析 `affinity` 列（KD / Kd / Ki…），统一为 pKD = -log10(KD/M)
- 用 PDB ID 把 SAbDab 的链信息（Hchain/Lchain/antigen_chain）与 SKEMPI 的突变/亲和力对应起来
- 过滤出同时有：PDB 结构、VH/VL 链、抗原链、亲和力数值 的样本
- 输出 CSV：pdb、vh_chain、vl_chain、antigen_chains、vh_seq、vl_seq、antigen_seq、pkd、affinity_raw、mutation_info…

### Step 2：结构层 —— 从 CIF 提取界面/几何特征 ✅ 已完成
- 脚本：`scripts/02_extract_structural_features.py`
- 输出：`processed/structural_features.csv`（1145 行，全部解析成功）
- 计算的特征：
  - H/L/抗原链残基数、原子数
  - 界面残基数（CA < 8 Å）：`interface_residues_ab` / `interface_residues_ag`
  - 界面接触对数：`interface_contacts_ca_8A` / `interface_contacts_heavy_5A`
  - VH-VL 质心距离：`vh_vl_distance`
- 与 pKD 的相关性：界面特征 ~0.13–0.15（弱正相关，符合预期）

### Step 3/4：特征工程 + XGBoost 基线模型 ✅ 已完成
- 脚本：`scripts/03_train_baseline.py`
- 特征：CDR 长度、电荷/疏水/芳香族比例、界面残基数、界面接触对数、VH-VL 距离、分辨率、抗体类型等
- 模型：`models/baseline_xgb.json`
- 结果：
  - **按 SAbDab2 序列聚类 split**：Test R² = **-0.15**（基本无法泛化到全新抗体/抗原簇）
  - **随机 split**：Test R² ≈ **0.50**，Pearson ≈ **0.71**
- 结论：当前手工特征有信息量，但跨序列簇泛化能力差，需要加入预训练语言模型嵌入（如 ESM-2）

### Step 5（修正）：ESM-2 嵌入模型 —— 诚实评估结果

**旧版结论有误。** 旧 `esm_xgb_predictions.csv` 因预测行错位 bug（`pkd_pred` 与
`pkd_true` 不对齐），显示 train R²=-0.6，看起来彻底失败；实际对齐后 train R²=0.89、
test R²≈0（严重过拟合、无法泛化）。

已做的修复与改进（`scripts/05_train_esm_xgb.py` 重写）：
- 修复预测保存的行错位 bug
- 去重：同一 (VH,VL,抗原) 三元组只保留一行（1145 → 758），消除同一复合物多份拷贝
- 全程报告 **Spearman**（排序相关，本任务的真正指标）
- **抗原分组交叉验证**（GroupKFold，group=抗原序列），给出诚实泛化估计
- 更强正则（max_depth=3, reg_lambda=2）+ PCA 降维

诚实结果（抗原聚类 test split，去重后 608 train / 150 test，XGBoost + PCA50）。
95% CI 来自 8000 次 bootstrap；Δ 为相对「整链池化」的配对差值：

| 特征 | Test Spearman | 95% CI | Δ vs 基准 |
|---|---|---|---|
| 仅结构特征 | 0.167 | [0.02, 0.31] | −0.224 **显著** |
| 界面池化（H/L 分开） | 0.195 | [0.05, 0.34] | −0.198 **显著** |
| 界面池化（H/L 串接） | 0.251 | [0.10, 0.39] | −0.141 不显著 |
| 界面池化（无接缝） | 0.293 | [0.14, 0.44] | −0.098 不显著 |
| mean-pooled + 结构特征 | 0.375 | [0.23, 0.51] | −0.018 不显著 |
| **mean-pooled ESM（基准）** | **0.393** | [0.25, 0.53] | — |
| mean-pooled + 界面池化 融合 | 0.413 | [0.27, 0.54] | +0.020 不显著 |

结论（已按显著性检验修正）：
- 关键收益来自「修 bug + 去重 + 正则」，把诚实 test 从 ≈0 提到 ≈0.39。
- **8 组对比只有 2 组显著，且都是「更差」方向**（仅结构特征、H/L 拆分）。
  Test 集 n=150，Spearman 分辨极限 ≈±0.17，因此融合的 +0.020 是噪声而非增益。
  更可靠的 GroupCV（n=608）上，整链池化单独用反而排第一。
  → **不能宣称「融合最优」**，各嵌入变体统计上无法区分。
- 一个被证伪的假设：H/L 串成一条序列会产生人为接缝。去掉接缝（维度不变）
  Δ=+0.040，CI [−0.088, +0.169]，不显著。
- 结构特征在嵌入之上无增量，符合「PLM 已隐含结构接触信息」的认知。
- 天花板就在 Spearman ~0.4 左右：跨抗原簇预测绝对 pKD 本就是领域内最难的设定，
  标签又混了不同实验方法与 `pdb_only` 噪声。若要更高，应换任务（ΔΔG 突变排序）
  或换抗体专用 PLM（AntiBERTy/IgBERT）。

脚本：
- `scripts/06_extract_interface_embeddings.py`：界面残基 ESM 池化，`--split-hl` 可分链
- `scripts/05_train_esm_xgb.py --emb-npz ... --feature-keys ...`：通用训练/评估
- `scripts/07_run_ablations.py`：一键跑全部消融 → `reports/ablation_table.md`
- `scripts/08_significance_test.py`：bootstrap CI + 配对检验 → `reports/significance_table.md`

### Step 5b：优化层（遗传算法）
- 对 CDR-H3 序列做 one-hot 编码
- 适应度 = 模型预测 pKD - λ × 可开发性惩罚（疏水 patch 比例）
- 输出 Pareto 前 20 候选

### Step 6：展示层
- Streamlit：单抗体分析 / 批量预测 / 亲和力优化
- 可视化：亲和力分布、CDR 热力图、Pareto 前沿

## 第一步交付物
- `scripts/01_build_dataset.py`：生成 `processed/antibody_antigen_dataset.csv`
- `data/.gitignore`：避免把大文件/中间结果提交
- 运行后返回样本数量、标签分布、示例行

## 说明
- 每一步只做当前模块，不一次性做完。
- 每步结束后会汇报结果，再决定下一步细节。
- 先不安装重型依赖（AlphaFold/ESM-2/Transformers），等做到对应模块再按需安装。
