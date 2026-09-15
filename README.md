# Early Type 2 Diabetes Prediction: Evaluating Domain Adaptation and Fairness Aware Machine Learning Across Diverse Populations

## Project Structure
```
dissertation_project/
├── data/
│   ├── raw/              # Place your raw data files here
│   ├── processed/        # Auto-generated cleaned datasets
│   └── external/         # Downloaded reference data
├── outputs/
│   ├── figures/          # All plots for Chapters 4-6 & Appendices
│   ├── tables/           # CSV result tables
│   └── models/           # Saved model weights
├── src/
│   ├── data/             # Data loaders & processors
│   ├── models/           # Baselines, DANN, CORAL, MMD, Fairness
│   ├── evaluation/       # Metrics, statistical tests, visualizations
│   └── pipeline.py       # Main orchestrator
├── notebooks/            # Step-by-step analysis scripts
├── main.py               # Run full pipeline
├── config.py             # Paths & hyperparameters
└── requirements.txt      # Python dependencies
```

## Setup Instructions (VS Code)

1. **Open folder in VS Code**: `File > Open Folder > dissertation_project/`

2. **Create virtual environment** (in VS Code terminal):
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

3. **Place raw data files** in `data/raw/`:
   - `diabetes.csv` (PIMA Indians)
   - `P_DEMO.xpt`, `P_BMX.xpt`, `P_GLU.xpt`, `P_GHB.xpt`, `P_DIQ.xpt` (NHANES)
   - `LLCP2015.XPT` or `LLCP2015.ASC` (BRFSS)

4. **Run pipeline**:
```bash
python main.py
```

## Raw Data Notes

### PIMA Indians
- Source: UCI ML Repository
- 768 records, all Pima Indian women, age 21+, 1988
- Zero values in Glucose, BloodPressure, SkinThickness, Insulin, BMI are missing data

### NHANES 2017-March 2020
- Source: CDC NCHS
- Complex survey design: use `SDMVPSU`, `SDMVSTRA`, `WTINTPRP`, `WTMECPRP`
- Diabetes label: `DIQ010=1` OR `LBXGLU>=126` OR `LBXGH>=6.5`
- Fasting glucose uses subsample weights `WTSAFPRP`

### BRFSS 2015
- Source: CDC BRFSS
- Telephone survey, no biomarker measurements (no glucose!)
- Hard adaptation scenario: lifestyle-only features
- Raw variables mapped to cleaned schema in `src/data/brfss_processor.py`

## Analysis Pipeline (5 Phases)

| Phase | Description | Output |
|-------|-------------|--------|
| 1 | Data loading, cleaning, feature alignment | `outputs/tables/*_processed.csv` |
| 2 | Baseline models trained on PIMA, evaluated on all subgroups | Bias amplification map |
| 3 | DANN, CORAL, MMD domain adaptation | Adapted models saved |
| 4 | Fairness constraints (Demographic Parity, Equalized Odds) | Fair models |
| 5 | Statistical testing, Pareto analysis, visualizations | Figures & tables for thesis |

## Chapter Outputs

- **Chapter 4**: Data pipeline architecture, model implementation details → `outputs/figures/`
- **Chapter 5**: Baseline bias map, adaptation results, fairness trade-offs → `outputs/tables/`
- **Chapter 6**: Calibration curves, feature space plots, Pareto frontiers → `outputs/figures/`
- **Appendices**: Full subgroup tables, statistical test results, code → `outputs/tables/`, `src/`

## Key Files for Each Chapter

### Chapter 4 (Implementation)
- `src/data/*.py` — data pipeline
- `src/models/*.py` — model architectures
- `outputs/figures/feature_space_*.png` — alignment visualizations

### Chapter 5 (Results)
- `outputs/tables/baseline_bias_map.csv`
- `outputs/tables/adaptation_results.csv`
- `outputs/tables/fairness_metrics.csv`
- `outputs/figures/pareto_frontier_*.png`

### Chapter 6 (Discussion)
- `outputs/figures/calibration_*.png`
- `outputs/figures/accuracy_fairness_tradeoff.png`
- `outputs/tables/statistical_tests.csv`

## Reproducibility
- Random seed fixed at 42 (`config.py`)
- All preprocessing decisions documented in source code
- Model hyperparameters in `config.py`
- Git-track all code; do NOT commit raw data files (too large)

## Citation & Ethics
- NHANES & BRFSS: public domain de-identified survey data
- PIMA: UCI ML Repository
- Ethics: UREC1 (No Human Participants)
- AI Transparency: Level 2 (AI for Shaping)
