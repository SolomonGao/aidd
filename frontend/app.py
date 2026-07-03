#!/usr/bin/env python3
"""
AbDock-AI Interactive Dashboard
==============================
A polished Streamlit frontend for visualizing antibody-antigen binding affinity
prediction results from the AbDock-AI pipeline.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "AIDD" / "processed"

PREDICTION_FILES = {
    "XGBoost — Mean + Interface (best)": PROCESSED / "pred_650m+interface_xgb.csv",
    "XGBoost — Whole-chain mean": PROCESSED / "pred_650m_xgb.csv",
    "XGBoost — Interface only": PROCESSED / "pred_interface_xgb.csv",
    "Ridge — Whole-chain mean": PROCESSED / "pred_650m_ridge.csv",
    "Ridge — Interface only": PROCESSED / "pred_interface_ridge.csv",
}
STRUCT_FEATURES = PROCESSED / "structural_features.csv"
DATASET = PROCESSED / "sabdab2_labeled_dataset.csv"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AbDock-AI | Antibody-Antigen Affinity Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark, glassy, scientific aesthetic
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Force entire app dark */
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
            color: #f8fafc !important;
        }

        .main .block-container {
            background: transparent;
            color: #f8fafc !important;
        }

        /* Headings */
        h1, h2, h3, h4, h5, h6 {
            color: #f8fafc !important;
            font-weight: 700;
        }

        /* All text elements */
        p, li, span, label, div, .stMarkdown, .stText {
            color: #f8fafc !important;
        }

        /* Streamlit native widgets */
        .stSelectbox label, .stMultiSelect label, .stTextInput label,
        .stNumberInput label, .stSlider label, .stRadio label, .stCheckbox label {
            color: #f8fafc !important;
            font-weight: 500 !important;
        }

        /* Dataframes and tables */
        .stDataFrame, .dataframe {
            color: #f8fafc !important;
        }
        .stDataFrame th, .dataframe th {
            background-color: #1e293b !important;
            color: #f8fafc !important;
        }
        .stDataFrame td, .dataframe td {
            background-color: #334155 !important;
            color: #f8fafc !important;
        }

        /* Expander */
        .streamlit-expanderHeader {
            color: #f8fafc !important;
            background-color: rgba(51, 65, 85, 0.6) !important;
            border-radius: 10px;
        }
        .streamlit-expanderContent {
            color: #f8fafc !important;
        }

        /* Warning/info boxes */
        .stAlert {
            color: #f8fafc !important;
        }
        .stAlert p {
            color: #f8fafc !important;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background: rgba(51, 65, 85, 0.7);
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 16px;
            padding: 1rem;
            backdrop-filter: blur(8px);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.15);
        }
        div[data-testid="stMetric"] label {
            color: #e2e8f0 !important;
            font-size: 0.85rem !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #38bdf8 !important;
            font-size: 2rem !important;
            font-weight: 700 !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
            font-size: 0.9rem !important;
        }

        /* Glass cards for markdown */
        .glass-card {
            background: rgba(51, 65, 85, 0.65);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 18px;
            padding: 1.5rem;
            margin-bottom: 1.25rem;
            backdrop-filter: blur(10px);
        }

        .hero-gradient {
            background: linear-gradient(135deg, #0ea5e9 0%, #8b5cf6 50%, #ec4899 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .pipeline-step {
            background: rgba(71, 85, 105, 0.7);
            border: 1px solid rgba(56, 189, 248, 0.35);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
            color: #f8fafc;
            font-size: 0.9rem;
            height: 100%;
        }

        .pipeline-arrow {
            display: flex;
            align-items: center;
            justify-content: center;
            color: #38bdf8;
            font-size: 1.5rem;
            height: 100%;
        }

        .footer {
            color: #cbd5e1 !important;
            font-size: 0.8rem;
            text-align: center;
            margin-top: 3rem;
        }
        .footer a {
            color: #38bdf8 !important;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: rgba(15, 23, 42, 0.98);
            border-right: 1px solid rgba(148, 163, 184, 0.12);
        }
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label {
            color: #e2e8f0 !important;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 2rem;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(51, 65, 85, 0.6);
            border-radius: 10px 10px 0 0;
            padding: 10px 20px;
            color: #f1f5f9;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(56, 189, 248, 0.15) !important;
            color: #38bdf8 !important;
            border-bottom: 2px solid #38bdf8;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["abs_error"] = (df["pkd_true"] - df["pkd_pred"]).abs()
    df["residual"] = df["pkd_pred"] - df["pkd_true"]
    return df


@st.cache_data(show_spinner=False)
def load_struct_features() -> pd.DataFrame:
    return pd.read_csv(STRUCT_FEATURES)


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATASET)


@st.cache_data(show_spinner=False)
def compute_metrics(df: pd.DataFrame) -> dict:
    from scipy.stats import pearsonr, spearmanr

    y_true = df["pkd_true"].values
    y_pred = df["pkd_pred"].values
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "spearman": spearmanr(y_true, y_pred)[0],
        "pearson": pearsonr(y_true, y_pred)[0],
        "r2": r2,
        "mae": np.mean(np.abs(y_true - y_pred)),
        "rmse": np.sqrt(np.mean((y_true - y_pred) ** 2)),
        "n": len(df),
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="text-align:center; margin-bottom:2rem;">
        <h1 style="font-size:1.6rem; margin-bottom:0.2rem;">🧬 AbDock-AI</h1>
        <p style="color:#94a3b8; font-size:0.85rem;">Antibody–Antigen Binding Affinity</p>
    </div>
    """,
    unsafe_allow_html=True,
)

model_choice = st.sidebar.selectbox(
    "Select prediction run",
    list(PREDICTION_FILES.keys()),
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Navigation**
    - 📊 Overview
    - 🎯 Results
    - 🔬 Predictions
    - 🏗️ Structure
    - 📖 About
    """
)

st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ About this dashboard"):
    st.markdown(
        """
        This dashboard visualizes the output of the **AbDock-AI** pipeline,
        which predicts antibody–antigen binding affinity (pKD) using ESM-2
        embeddings and structural features with honest, leakage-free evaluation.
        """
    )

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
pred_df = load_predictions(PREDICTION_FILES[model_choice])
struct_df = load_struct_features()
dataset_df = load_dataset()

metrics_all = compute_metrics(pred_df)
test_df = pred_df[pred_df["split"] == "test"]
train_df = pred_df[pred_df["split"] == "train"]
metrics_test = compute_metrics(test_df)
metrics_train = compute_metrics(train_df)

# ---------------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="glass-card">
        <h1 style="font-size:3rem; margin-bottom:0.4rem;">
            <span class="hero-gradient">AbDock-AI</span>
        </h1>
        <h3 style="color:#cbd5e1; font-weight:300; margin-top:0;">
            Antibody–Antigen Binding Affinity Prediction
        </h3>
        <p style="color:#94a3b8; font-size:1.05rem; max-width:900px;">
            A rigorous, leakage-free machine-learning pipeline that predicts <strong>pKD</strong>
            from sequence and 3D structure. Designed to reduce costly wet-lab iterations during
            antibody affinity maturation.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

hero_cols = st.columns(5)
hero_metrics = [
    ("Spearman ρ", f"{metrics_test['spearman']:.3f}", "test"),
    ("Pearson r", f"{metrics_test['pearson']:.3f}", "test"),
    ("R²", f"{metrics_test['r2']:.3f}", "test"),
    ("Test complexes", f"{metrics_test['n']}", "test"),
    ("Total complexes", f"{metrics_all['n']}", "all"),
]
for col, (label, value, split) in zip(hero_cols, hero_metrics):
    with col:
        st.metric(label=label, value=value)

st.markdown("---")

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab_overview, tab_results, tab_predictions, tab_structure, tab_about = st.tabs(
    ["📊 Overview", "🎯 Results", "🔬 Predictions", "🏗️ Structure", "📖 About"]
)

# ===========================================================================
# TAB 1 — Overview
# ===========================================================================
with tab_overview:
    st.markdown("## Pipeline Architecture")
    st.markdown(
        """
        <p style="color:#94a3b8; margin-top:-0.5rem;">
            End-to-end flow from raw SAbDab complexes to predicted pKD with honest evaluation.
        </p>
        """,
        unsafe_allow_html=True,
    )

    pipeline_steps = [
        ("🧪", "SAbDab Complexes", "~1,100 curated antibody–antigen structures"),
        ("🧹", "Curation", "pKD labels, deduplication, antigen-cluster split"),
        ("🧬", "ESM-2 650M", "Whole-chain mean pooling (VH · VL · Ag)"),
        ("🤝", "Interface", "Paratope + epitope residue pooling"),
        ("📐", "Geometry", "CA contacts, heavy contacts, VH–VL geometry"),
        ("🧩", "Fusion + PCA", "Concatenated features reduced to 50 dims"),
        ("⚡", "XGBoost / Ridge", "GroupKFold CV grouped by antigen"),
        ("📈", "pKD + Metrics", "Spearman · Pearson · R²"),
    ]

    rows = [pipeline_steps[i : i + 4] for i in range(0, len(pipeline_steps), 4)]
    for row in rows:
        cols = st.columns(4)
        for col, (emoji, title, desc) in zip(cols, row):
            col.markdown(
                f"""
                <div class="pipeline-step">
                    <div style="font-size:2rem; margin-bottom:0.5rem;">{emoji}</div>
                    <div style="font-weight:700; color:#38bdf8; margin-bottom:0.3rem;">{title}</div>
                    <div style="color:#94a3b8; font-size:0.8rem;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("## Model Comparison")
    comparison_data = {
        "Feature set": [
            "Structural features only",
            "Interface-residue pooling (paratope + epitope)",
            "Whole-chain mean pooling (VH + VL + antigen)",
            "Mean pooling + structural features",
            "Mean pooling + interface pooling (fused)",
        ],
        "Spearman": [0.189, 0.251, 0.393, 0.375, 0.413],
        "Pearson": [0.173, 0.243, 0.385, 0.371, 0.405],
        "R2": [-0.04, 0.01, 0.15, 0.13, 0.16],
    }
    comp_df = pd.DataFrame(comparison_data)

    fig_comp = go.Figure()
    colors = {"Spearman": "#38bdf8", "Pearson": "#a78bfa", "R2": "#f472b6"}
    for metric in ["Spearman", "Pearson", "R2"]:
        fig_comp.add_trace(
            go.Bar(
                name=metric,
                x=comp_df["Feature set"],
                y=comp_df[metric],
                marker_color=colors[metric],
            )
        )
    fig_comp.update_layout(
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        xaxis_tickangle=-30,
        yaxis_title="Score",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=120),
    )
    st.plotly_chart(fig_comp, width="stretch")

    st.markdown(
        """
        <div class="glass-card">
            <h4 style="margin-top:0;">🔑 Key Takeaways</h4>
            <ul style="color:#cbd5e1; line-height:1.8;">
                <li><strong>Whole-chain mean pooling</strong> carries the strongest signal (Spearman 0.393).</li>
                <li><strong>Interface pooling</strong> alone is noisier, but adds orthogonal signal when fused (0.393 → 0.413).</li>
                <li><strong>Hand-crafted structural features</strong> add little on top of language-model embeddings.</li>
                <li>The biggest gain came from fixing evaluation hygiene — moving the honest held-out correlation from ≈0 to ~0.41.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ===========================================================================
# TAB 2 — Results
# ===========================================================================
with tab_results:
    st.markdown("## Test-Set Performance")

    res_cols = st.columns(4)
    res_metrics = [
        ("Spearman ρ", metrics_test["spearman"]),
        ("Pearson r", metrics_test["pearson"]),
        ("R²", metrics_test["r2"]),
        ("MAE", metrics_test["mae"]),
    ]
    for col, (label, value) in zip(res_cols, res_metrics):
        col.metric(label=label, value=f"{value:.3f}")

    st.markdown("### True vs Predicted pKD")
    fig_scatter = px.scatter(
        test_df,
        x="pkd_true",
        y="pkd_pred",
        color="abs_error",
        color_continuous_scale="plasma",
        hover_data=["pdb_id", "instance"],
        labels={"pkd_true": "True pKD", "pkd_pred": "Predicted pKD"},
        template="plotly_dark",
        title="Held-out antigen-cluster test split",
    )
    fig_scatter.update_traces(marker=dict(size=10, opacity=0.85, line=dict(width=1, color="white")))
    # Diagonal
    x_range = [test_df["pkd_true"].min() - 0.5, test_df["pkd_true"].max() + 0.5]
    fig_scatter.add_trace(
        go.Scatter(
            x=x_range,
            y=x_range,
            mode="lines",
            name="Perfect prediction",
            line=dict(color="#38bdf8", dash="dash", width=2),
        )
    )
    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        xaxis=dict(gridcolor="rgba(148,163,184,0.15)", zeroline=False),
        yaxis=dict(gridcolor="rgba(148,163,184,0.15)", zeroline=False),
        coloraxis_colorbar=dict(title="|Error|"),
    )
    st.plotly_chart(fig_scatter, width="stretch")

    st.markdown("### Residual Distribution")
    fig_hist = px.histogram(
        test_df,
        x="residual",
        nbins=30,
        color_discrete_sequence=["#8b5cf6"],
        template="plotly_dark",
        labels={"residual": "Predicted − True pKD"},
    )
    fig_hist.add_vline(x=0, line_dash="dash", line_color="#38bdf8", line_width=2)
    fig_hist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        xaxis=dict(gridcolor="rgba(148,163,184,0.15)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.15)"),
    )
    st.plotly_chart(fig_hist, width="stretch")

    st.markdown("### Metric Breakdown by Split")
    breakdown = pd.DataFrame(
        {
            "Split": ["Train", "Test"],
            "Spearman": [metrics_train["spearman"], metrics_test["spearman"]],
            "Pearson": [metrics_train["pearson"], metrics_test["pearson"]],
            "R²": [metrics_train["r2"], metrics_test["r2"]],
            "MAE": [metrics_train["mae"], metrics_test["mae"]],
            "RMSE": [metrics_train["rmse"], metrics_test["rmse"]],
            "N": [metrics_train["n"], metrics_test["n"]],
        }
    )
    st.dataframe(
        breakdown.style.format(
            {
                "Spearman": "{:.3f}",
                "Pearson": "{:.3f}",
                "R²": "{:.3f}",
                "MAE": "{:.3f}",
                "RMSE": "{:.3f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

# ===========================================================================
# TAB 3 — Predictions
# ===========================================================================
with tab_predictions:
    st.markdown("## Interactive Prediction Explorer")

    split_filter = st.multiselect(
        "Filter by split",
        options=pred_df["split"].unique().tolist(),
        default=["test"],
    )

    filtered_df = pred_df[pred_df["split"].isin(split_filter)]

    col_a, col_b = st.columns([2, 1])
    with col_a:
        fig_pred = px.scatter(
            filtered_df,
            x="pkd_true",
            y="pkd_pred",
            color="split",
            hover_data=["pdb_id", "instance", "abs_error"],
            labels={"pkd_true": "True pKD", "pkd_pred": "Predicted pKD"},
            template="plotly_dark",
            color_discrete_sequence=["#38bdf8", "#f472b6"],
            title=f"{model_choice}",
        )
        fig_pred.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color="white")))
        lo, hi = filtered_df["pkd_true"].min() - 0.5, filtered_df["pkd_true"].max() + 0.5
        fig_pred.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                name="Perfect prediction",
                line=dict(color="#fbbf24", dash="dash", width=2),
            )
        )
        fig_pred.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            xaxis=dict(gridcolor="rgba(148,163,184,0.15)", zeroline=False),
            yaxis=dict(gridcolor="rgba(148,163,184,0.15)", zeroline=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_pred, width="stretch")

    with col_b:
        st.markdown("### Worst predictions")
        worst = filtered_df.nlargest(10, "abs_error")[
            ["pdb_id", "split", "pkd_true", "pkd_pred", "abs_error"]
        ].reset_index(drop=True)
        st.dataframe(
            worst.style.format({"pkd_true": "{:.2f}", "pkd_pred": "{:.2f}", "abs_error": "{:.2f}"}),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### Search by PDB ID")
    pdb_query = st.text_input("Enter PDB ID (e.g., 1BJ1)", value="").upper().strip()
    if pdb_query:
        matches = filtered_df[filtered_df["pdb_id"].str.upper().str.contains(pdb_query)]
        if matches.empty:
            st.warning("No matching PDB IDs found.")
        else:
            st.dataframe(
                matches[["instance", "pdb_id", "split", "pkd_true", "pkd_pred", "abs_error"]]
                .sort_values("abs_error", ascending=False)
                .style.format({"pkd_true": "{:.3f}", "pkd_pred": "{:.3f}", "abs_error": "{:.3f}"}),
                use_container_width=True,
                hide_index=True,
            )

# ===========================================================================
# TAB 4 — Structure
# ===========================================================================
with tab_structure:
    st.markdown("## Structural Feature Landscape")

    feat_cols = ["h_residues", "l_residues", "ag_residues", "ab_residues",
                 "interface_residues_ab", "interface_residues_ag",
                 "interface_contacts_ca_8A", "interface_contacts_heavy_5A", "vh_vl_distance"]
    available_cols = [c for c in feat_cols if c in struct_df.columns]

    x_var = st.selectbox("X-axis feature", available_cols, index=available_cols.index("interface_residues_ab"))
    y_var = st.selectbox("Y-axis feature", available_cols, index=available_cols.index("interface_contacts_ca_8A"))
    color_var = st.selectbox("Color by", ["None"] + available_cols, index=0)

    merged = struct_df.merge(pred_df[["instance", "pkd_true", "pkd_pred"]], on="instance", how="left")

    plot_kwargs = dict(
        x=x_var,
        y=y_var,
        color=(None if color_var == "None" else color_var),
        hover_data=["pdb_id", "instance", "pkd_true", "pkd_pred"],
        template="plotly_dark",
        color_continuous_scale="plasma",
    )
    fig_struct = px.scatter(merged, **{k: v for k, v in plot_kwargs.items() if v is not None})
    fig_struct.update_traces(marker=dict(size=9, opacity=0.75, line=dict(width=0.5, color="white")))
    fig_struct.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        xaxis=dict(gridcolor="rgba(148,163,184,0.15)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.15)"),
    )
    st.plotly_chart(fig_struct, width="stretch")

    st.markdown("### Feature Summary")
    summary = struct_df[available_cols].describe().T
    st.dataframe(summary.style.format("{:.2f}"), use_container_width=True)

    st.markdown("### Distribution of Interface Residues")
    fig_box = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Paratope residues", "Epitope residues"),
    )
    fig_box.add_trace(
        go.Box(y=struct_df["interface_residues_ab"], name="Paratope", marker_color="#38bdf8"),
        row=1, col=1,
    )
    fig_box.add_trace(
        go.Box(y=struct_df["interface_residues_ag"], name="Epitope", marker_color="#f472b6"),
        row=1, col=2,
    )
    fig_box.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        showlegend=False,
    )
    fig_box.update_yaxes(gridcolor="rgba(148,163,184,0.15)")
    st.plotly_chart(fig_box, width="stretch")

# ===========================================================================
# TAB 5 — About
# ===========================================================================
with tab_about:
    st.markdown("## Methodology")
    st.markdown(
        """
        <div class="glass-card">
            <p style="color:#cbd5e1; line-height:1.7;">
                The model fuses two complementary sequence representations:
            </p>
            <ul style="color:#cbd5e1; line-height:1.8;">
                <li><strong>Whole-chain mean pooling</strong> — mean of ESM-2 per-residue embeddings over the full VH, VL, and antigen chains.</li>
                <li><strong>Interface-residue pooling</strong> — ESM-2 embeddings pooled over only the paratope and epitope residues, identified from 3D structure via CA &lt; 8 Å and heavy &lt; 5 Å contacts.</li>
            </ul>
            <p style="color:#94a3b8; line-height:1.7;">
                Sequence and structure indices are kept naturally aligned, so no external numbering map is required.
                Features are fused, projected with PCA(50), and fed to XGBoost / Ridge with GroupKFold CV grouped by antigen.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## Rigor Highlights")
    rigor = [
        ("🛡️", "Leakage-aware evaluation", "Antigen-clustered train/test split plus GroupKFold grouped by antigen."),
        ("🧹", "Data hygiene", "Near-identical complex copies are de-duplicated before splitting."),
        ("📊", "Ranking-first metrics", "Spearman/Pearson alongside R², because absolute calibration across antigens is hard."),
        ("🐛", "Rigorous debugging", "A prediction-alignment bug was identified and fixed, revealing the model's true behavior."),
    ]
    for emoji, title, desc in rigor:
        st.markdown(
            f"""
            <div class="glass-card" style="display:flex; gap:1rem; align-items:flex-start;">
                <div style="font-size:2rem;">{emoji}</div>
                <div>
                    <h4 style="margin:0 0 0.3rem 0; color:#38bdf8;">{title}</h4>
                    <p style="margin:0; color:#cbd5e1;">{desc}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("## Limitations & Next Steps")
    st.markdown(
        """
        <div class="glass-card">
            <ul style="color:#cbd5e1; line-height:1.8;">
                <li>The ceiling for cross-antigen absolute pKD is ~Spearman 0.4; labels mix assay types and confidence levels.</li>
                <li>A more directly useful reframing is <strong>ΔΔG mutation ranking</strong> within a single antibody lineage.</li>
                <li>Swapping generic ESM-2 for an antibody-specific LM (AntiBERTy / IgBERT) is a promising next experiment.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        Data: <a href="https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/" target="_blank">SAbDab</a> ·
        LM: <a href="https://github.com/facebookresearch/esm" target="_blank">ESM-2</a> ·
        Built with Streamlit + Plotly
    </div>
    """,
    unsafe_allow_html=True,
)
