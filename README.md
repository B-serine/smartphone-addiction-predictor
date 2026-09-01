
# 📱 Smartphone Addiction Predictor — ROC-AUC Tabular ML Pipeline

Predicting the probability of smartphone addiction from behavioral and lifestyle data, using a full machine learning pipeline: cleaning → feature engineering → target encoding → gradient boosting (XGBoost, LightGBM, CatBoost) → deep learning (PyTorch) → stacked ensemble.

Evaluated with **ROC-AUC** (Area Under the ROC Curve).

---

## 📝 Description

This repository contains an end-to-end machine learning pipeline for a binary classification task: predicting whether a person is addicted to their smartphone (`addicted_label`, probability between 0 and 1) based on 11 behavioral and demographic features — screen time, social media/gaming usage, sleep, notifications, stress, academic/work impact, and more.

The project is built as a set of progressively more advanced Jupyter notebooks, going from a clean baseline pipeline to a competition-grade stacked ensemble combining gradient boosting models and a deep learning model with entity embeddings. Every notebook is fully commented, reproducible, and designed so you can plug in your own `train.csv` / `test.csv` and get submission files ready to go.

**Task type:** Binary classification (tabular data)
**Metric:** ROC-AUC (Area Under the ROC Curve)
**Train set:** 691,369 rows
**Test set:** 296,302 rows

---

## 📂 Dataset

| Column | Description |
|---|---|
| `id` | Unique row identifier |
| `age` | Age of the individual |
| `daily_screen_time_hours` | Average daily screen time (hours) |
| `social_media_hours` | Daily time spent on social media (hours) |
| `gaming_hours` | Daily time spent gaming (hours) |
| `work_study_hours` | Daily time spent on work/study (hours) |
| `sleep_hours` | Average daily sleep (hours) |
| `notifications_per_day` | Number of phone notifications received per day |
| `app_opens_per_day` | Number of times apps are opened per day |
| `weekend_screen_time` | Screen time on weekends (hours) |
| `gender` | Gender of the individual |
| `stress_level` | Self-reported stress level |
| `academic_work_impact` | Self-reported impact of phone use on academic/work performance |
| `addicted_label` | **Target** — probability/label of smartphone addiction (train only) |

### Submission format

```
id,addicted_label
691369,0.2
691370,0.3
691371,0.1
```

---

## 🧠 Approach

The pipeline is organized into three notebooks of increasing sophistication:

### 1. `smartphone_addiction_prediction.ipynb` — Baseline pipeline
- Full data cleaning: impossible-value detection, missing-value imputation (median/mode), outlier clipping, categorical encoding
- Feature engineering: screen-time ratios, notification intensity, weekend vs. weekday behavior
- 6 baseline models trained and compared on a held-out validation split: Logistic Regression, KNN, Random Forest, XGBoost, LightGBM, Gradient Boosting
- ROC-AUC evaluation and ROC curve comparison plots
- A model-improvement checklist (which parameter to raise/lower depending on over/underfitting)
- Bonus simple ensemble (probability averaging)

### 2. `smartphone_addiction_advanced.ipynb` — Tuned boosting pipeline
- Trims down to the three strongest tabular models: **XGBoost, LightGBM, CatBoost**
- Fixes ordinal encoding (severity-ordered categories instead of arbitrary alphabetical label encoding)
- Adds interaction features
- **Optuna** hyperparameter search per model
- **5-fold Stratified Cross-Validation** with out-of-fold (OOF) evaluation — the most honest ROC-AUC estimate before submitting
- Weighted blend of the three models based on each model's own OOF AUC

### 3. `smartphone_addiction_deep.ipynb` — Max-score stacked pipeline
- Everything from the advanced pipeline, plus:
- **Out-of-fold smoothed target encoding** for categorical/binned features (leak-free)
- A **PyTorch deep learning model** (MLP with entity embeddings for categorical features, batch normalization, dropout, early stopping), trained with 5-fold CV
- **Stacking**: a logistic regression meta-model trained on the out-of-fold predictions of all four base models (XGBoost, LightGBM, CatBoost, Neural Net), learning the optimal way to combine them
- Correlation diagnostics between models to check whether they're capturing genuinely different signal
- A troubleshooting checklist for pushing the score further (interaction target encoding, distribution shift checks, deeper diagnostics)

---

## 📁 Repository Structure

```
.
├── smartphone_addiction_prediction.ipynb   # Baseline: cleaning + 6 models
├── smartphone_addiction_advanced.ipynb     # XGBoost + LightGBM + CatBoost, tuned, 5-fold CV
├── smartphone_addiction_deep.ipynb         # + target encoding + PyTorch NN + stacking
├── data/
│   ├── train.csv                           # (not included — add your own)
│   └── test.csv                            # (not included — add your own)
├── submissions/                            # generated submission CSVs land here
└── README.md
```

> **Note:** the dataset files are not included in this repository. Place your `train.csv` and `test.csv` in the `data/` folder (or update the `TRAIN_PATH` / `TEST_PATH` variables at the top of each notebook) before running.

---

## ⚙️ Setup

### Option A — Conda (recommended)

```bash
# Create a clean environment
conda create -n addiction python=3.11 -y
conda activate addiction

# Install core dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter notebook

# Install the gradient boosting models
pip install xgboost lightgbm catboost

# Install hyperparameter tuning
pip install optuna

# Install PyTorch (CPU-only build; see pytorch.org for GPU builds)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Launch Jupyter
jupyter notebook
```

### Option B — venv + pip

```bash
python -m venv addiction_env
source addiction_env/bin/activate      # Windows: addiction_env\Scripts\activate

pip install pandas numpy scikit-learn matplotlib seaborn jupyter notebook \
            xgboost lightgbm catboost optuna
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## ▶️ Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/smartphone-addiction-predictor.git
   cd smartphone-addiction-predictor
   ```
2. Add your `train.csv` and `test.csv` to the `data/` folder.
3. Activate your environment (see Setup above).
4. Open and run whichever notebook fits your needs, top to bottom:
   - Want a clear, simple baseline to learn from? → `smartphone_addiction_prediction.ipynb`
   - Want a strong, properly tuned boosting pipeline? → `smartphone_addiction_advanced.ipynb`
   - Want the maximum score with deep learning + stacking? → `smartphone_addiction_deep.ipynb`
5. Submission files (`submission_<model>.csv`) are generated automatically in the `id,addicted_label` format required by the competition.

---

## 📊 Results

Each notebook prints and plots a comparison table of ROC-AUC scores per model. Fill in your own results here after running:

| Model | Validation / OOF ROC-AUC |
|---|---|
| Logistic Regression | — |
| KNN | — |
| Random Forest | — |
| Gradient Boosting | — |
| XGBoost | — |
| LightGBM | — |
| CatBoost | — |
| Neural Network (PyTorch) | — |
| **Stacked Ensemble** | — |

---

## 🔧 Tech Stack

- **Python 3.11**
- **pandas / numpy** — data manipulation
- **scikit-learn** — preprocessing, cross-validation, metrics, meta-model
- **XGBoost / LightGBM / CatBoost** — gradient boosting models
- **PyTorch** — deep learning model with entity embeddings
- **Optuna** — hyperparameter optimization
- **matplotlib / seaborn** — visualization

---

## 💡 Key Techniques Used

- Leak-free data cleaning (imputation and outlier clipping fit on train only, applied to test)
- Correct ordinal vs. nominal categorical encoding
- Feature engineering (ratios, interactions, binning)
- Out-of-fold smoothed target encoding
- Stratified K-Fold cross-validation with out-of-fold evaluation
- Hyperparameter optimization with Optuna
- Deep learning with entity embeddings for categorical features
- Model stacking with a learned meta-model

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

## 🙋 Acknowledgements

Built as part of a personal ROC-AUC optimization exercise on a smartphone addiction prediction dataset.
