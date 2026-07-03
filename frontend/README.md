# AbDock-AI Streamlit Dashboard

A polished, interactive visualization frontend for the AbDock-AI antibody–antigen binding affinity pipeline.

## Features

- **Overview** — pipeline architecture diagram and feature-set benchmark from the paper.
- **Results** — test-set scatter plots, residual distributions, and metric tables.
- **Predictions** — interactive explorer with split filtering, worst-prediction tables, and PDB search.
- **Structure** — structural feature landscape (interface residues, contacts, VH–VL geometry).
- **About** — methodology, rigor highlights, limitations, and next steps.

## Quick start

```bash
cd /home/xinggao/aidd/frontend

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

## Data sources

The dashboard reads processed artifacts from `../AIDD/processed/`:

- `pred_650m+interface_xgb.csv`
- `pred_650m_xgb.csv`
- `pred_interface_xgb.csv`
- `pred_650m_ridge.csv`
- `pred_interface_ridge.csv`
- `structural_features.csv`
- `sabdab2_labeled_dataset.csv`

Make sure these files exist before launching the app.
