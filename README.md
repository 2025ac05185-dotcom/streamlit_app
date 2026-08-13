# Telco Customer Churn — Classification Model Comparison

**BITS Pilani WILP · M.Tech (AIML) · Machine Learning · Assignment 2**

Five classifiers trained on one shared preprocessing pipeline, scored on a
common held-out split, and served through an interactive Streamlit dashboard.

---

## a. Problem Statement

A telecommunications provider loses roughly a quarter of its subscriber base
each billing cycle. Winning a replacement customer costs several times more
than keeping an existing one, so the commercial value sits in **flagging
at-risk customers early enough to intervene** — a retention call, a contract
upgrade offer, a discount.

Framed as supervised learning: given 19 attributes describing a customer's
demographics, subscribed services, contract type and billing history, predict
the binary label `Churn ∈ {Yes, No}`.

The business asymmetry matters and shapes every modelling choice below. A
**false negative** (a churner the model calls safe) costs a lost customer. A
**false positive** (a loyal customer flagged) costs one unnecessary retention
call. Those are not equally expensive, which is why accuracy is the *least*
interesting number in the results table and MCC the most.

---

## b. Dataset Description

| Property | Value |
| --- | --- |
| Source | IBM Sample Data Sets — Telco Customer Churn (public; mirrored on Kaggle) |
| Instances | 7,043 customers (requirement: ≥ 500) |
| Columns | 21 raw → 19 predictive features after dropping `customerID` and the label (requirement: ≥ 12) |
| Encoded width | 45 columns after one-hot encoding |
| Task | Binary classification |
| Class balance | 5,174 retained (73.46%) / 1,869 churned (26.54%) |
| Split | 75 / 25 stratified — 5,282 train, 1,761 test, `random_state=42` |

### Feature groups

- **Demographic (4)** — `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Services (9)** — `PhoneService`, `MultipleLines`, `InternetService`,
  `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`,
  `StreamingTV`, `StreamingMovies`
- **Account (6)** — `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`,
  `MonthlyCharges`, `TotalCharges`

### Data quality issues found and handled

1. **`TotalCharges` is not numeric.** Eleven rows hold a single space character
   instead of a number. All eleven belong to customers with `tenure = 0` — they
   had been billed nothing because they had not completed a cycle. Read
   naively, pandas types the whole column as text and every downstream scaler
   fails silently or loudly. Handled with `pd.to_numeric(errors="coerce")`
   followed by median imputation inside the pipeline.
2. **`customerID`** is a unique key with zero predictive value and is dropped
   before modelling. Left in, tree-based models would happily memorise
   individual rows.
3. **Whitespace** is stripped from every string column before encoding, so
   `"Yes"` and `"Yes "` cannot become two distinct one-hot categories.

### Preprocessing pipeline

All five models share **one identical `ColumnTransformer`**, so any difference
in the results table is attributable to the algorithm and not to inconsistent
feature engineering:

- Numeric → median imputation → `StandardScaler`
- Categorical → most-frequent imputation → `OneHotEncoder(handle_unknown="ignore")`

Scaling is mandatory rather than decorative here: kNN measures Euclidean
distance, so an unscaled `TotalCharges` (range ≈ 0–8,700) would completely
drown out `tenure` (0–72) and all 40-odd binary indicators.

Because preprocessing is bundled *inside* each saved `Pipeline`, the Streamlit
app accepts a **raw** CSV upload with no client-side encoding whatsoever.

---

## c. GitHub Repository Link

> **https://github.com/<your-username>/telco-churn-ml**
>
> _Replace with the actual repository URL before submitting._

### Live Streamlit App

> **https://<your-app-name>.streamlit.app**
>
> _Replace with the actual deployed URL before submitting._

### Repository structure

```
telco-churn-ml/
├── app.py                  # Streamlit application
├── requirements.txt        # Pinned dependencies
├── README.md               # This file
├── test_data.csv           # Held-out test split (1,761 rows, raw format)
├── .gitignore
├── .streamlit/
│   └── config.toml         # Dark theme + chart colour tokens
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── model/
    ├── preprocessing.py    # Shared feature pipeline
    ├── train_models.py     # Trains and evaluates all five models
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── random_forest.joblib
    ├── metrics.json        # Machine-readable results
    └── metrics.md          # Results as a Markdown table
```

### Reproducing the results

```bash
git clone https://github.com/<your-username>/telco-churn-ml.git
cd telco-churn-ml
pip install -r requirements.txt

python model/train_models.py   # retrains, rewrites test_data.csv + metrics
streamlit run app.py           # dashboard at localhost:8501
```

---

## d. Models Used

Five classifiers, each wrapped in the shared preprocessing pipeline and fitted
on the same 5,282-row training split.

| # | Model | Configuration |
| --- | --- | --- |
| 1 | Logistic Regression | `max_iter=2000`, `class_weight="balanced"` |
| 2 | Decision Tree | `max_depth=6`, `min_samples_leaf=25`, `class_weight="balanced"` |
| 3 | k-Nearest Neighbours | `n_neighbors=25`, `weights="distance"` |
| 4 | Naive Bayes | `GaussianNB` on the dense encoded matrix |
| 5 | Random Forest (Ensemble) | `n_estimators=400`, `min_samples_leaf=5`, `class_weight="balanced_subsample"` |

Every algorithm that exposes a class weight is given one. Without it the models
drift toward predicting "No" for everybody, which scores a comfortable-looking
73% accuracy while catching nobody worth retaining.

### Evaluation metrics — held-out test set (1,761 rows, threshold 0.50)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.7496 | **0.8460** | 0.5182 | 0.7944 | 0.6272 | 0.4743 |
| Decision Tree | 0.7348 | 0.8320 | 0.5000 | 0.8201 | 0.6212 | 0.4667 |
| kNN | **0.7802** | 0.8154 | **0.5897** | 0.5632 | 0.5761 | 0.4281 |
| Naive Bayes | 0.6979 | 0.8112 | 0.4616 | **0.8373** | 0.5951 | 0.4284 |
| Random Forest (Ensemble) | 0.7740 | 0.8437 | 0.5561 | 0.7323 | **0.6322** | **0.4828** |

**Reference point:** predicting "No churn" for every single customer scores
**0.7348 accuracy** on this test set — and catches nobody, for an MCC of 0.000.
Naive Bayes scores *worse* than that baseline on accuracy (0.6979) and the
Decision Tree ties it exactly (0.7348), yet both carry an MCC above 0.46. Two of
the five models are therefore indistinguishable from — or beaten by — a model
that does nothing, if you only read the accuracy column. That is exactly why the
table has six columns.

---

### Observations on model performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Best ranking model in the set (AUC 0.8460) despite being the simplest. Churn in this dataset is close to linearly separable in log-odds space: the fitted coefficients say tenure (−1.14) and a two-year contract (−0.74) protect against churn, while fibre-optic internet (+0.68) and month-to-month contracts (+0.64) drive it. Class weighting pushes the decision boundary well below the natural rate, so it flags 716 of 1,761 customers — recall 0.79 bought at precision 0.52. Its accuracy (0.7496) sits barely above the do-nothing baseline, which is a feature of that trade-off, not a failure. |
| **Decision Tree** | Splits first on `Contract_Month-to-month`, confirming the single strongest signal in the data. Highest recall of any non-Bayes model (0.8201) and completely interpretable — 55 leaves at depth 6, a rule set a retention team could read. But it lands at exactly 0.7348 accuracy, identical to the majority-class baseline, while its MCC of 0.4667 is far from the baseline's 0.0. A crisp demonstration that accuracy alone would rate this model as worthless. Depth had to be capped: unpruned, it memorises the training split and generalises noticeably worse. |
| **kNN** | Highest accuracy (0.7802) and highest precision (0.5897), and the weakest model where it counts. It flags only 446 customers and misses 44% of actual churners (recall 0.5632) — the most conservative model in the set. Its lowest-in-class AUC (0.8154) shows the ranking itself is weaker, not just the threshold. With 45 encoded dimensions the distance metric is diluted by dozens of sparse one-hot columns, the classic curse-of-dimensionality failure. Also the slowest at inference: 200 KB of stored training data that must be scanned per query. |
| **Naive Bayes** | Best recall in the study (0.8373) and worst accuracy (0.6979) — it flags 847 of 1,761 customers, nearly half the book. The feature-independence assumption is badly violated here: `InternetService`, `OnlineSecurity`, `TechSupport` and `StreamingTV` are structurally correlated (no internet ⇒ no add-ons), so correlated evidence gets counted repeatedly and probabilities are pushed to extremes. Poorly calibrated, but the ranking survives (AUC 0.8112) and it trains in milliseconds. Usable as a cheap high-recall pre-filter, not as a final decision-maker. |
| **Random Forest (Ensemble)** | Best F1 (0.6322) and best MCC (0.4828) — the most balanced model on the two metrics that account for both error types. Bagging 400 trees repairs the single tree's variance problem while keeping most of its recall: 0.7323 recall at 0.5561 precision, catching 342 of 467 churners. AUC 0.8437 is within 0.002 of Logistic Regression, so on pure ranking the two are effectively tied. Importances corroborate the linear model — `tenure` (0.127), `Contract_Month-to-month` (0.121) and `TotalCharges` (0.107) dominate. Costs 7.3 MB on disk against Logistic Regression's 3 KB. |
| **Overall winner for this dataset** | **Random Forest**, on MCC (0.4828) and F1 (0.6322) — the metrics that stay honest under a 73/27 class imbalance. The caveat is worth stating plainly: Logistic Regression *out-ranks* it on AUC (0.8460 vs 0.8437) at roughly 1/2000th the model size and with directly readable coefficients. Random Forest wins the assignment's scoring criteria; for a production retention system where a stakeholder has to be told *why* a customer was flagged, Logistic Regression would be the defensible choice. |

### A note on threshold choice

Every metric except AUC above is computed at the default 0.50 cut-off. AUC is
threshold-independent, which is why it is the fairest single-number comparison
of the five algorithms — and why the deployed app exposes a **threshold slider**:
dragging it re-scores precision, recall, F1 and MCC live while the ROC curve
stays put, making the retention-cost trade-off visible rather than assumed.

---

## Streamlit Application

Deployed on Streamlit Community Cloud. Features:

| Requirement | Implementation |
| --- | --- |
| Dataset upload (CSV) | Sidebar uploader; falls back to the bundled `test_data.csv` so the app is never empty on first load |
| Model selection dropdown | All five trained pipelines, switchable live |
| Display of evaluation metrics | All six metrics as headline stat tiles, recomputed on the uploaded data |
| Confusion matrix / classification report | Interactive heatmap **and** per-class precision/recall/F1 report, side by side |

Beyond the requirements:

- **Decision-threshold slider** that re-scores every metric live, with a
  "threshold economics" row (churners caught, false-alarm rate, flag rate)
  translating the cut-off into retention-team workload
- **Confusion matrix shaded by row share**, so the 1,021 true negatives cannot
  wash out the three cells that carry the actual decision
- **ROC curve** with hover read-out, plus an **all-models ROC overlay** and a
  comparison bar chart re-rankable by any of the six metrics
- **Per-row predictions** led by churn probability, prediction and correctness,
  with a misclassified-rows-only filter and CSV download

Charts are built with Altair rather than static Matplotlib images, so every
mark carries a tooltip. The colour palette is validated for colour-vision
deficiency (worst adjacent-pair separation ΔE 8.4 under protanopia, every
series ≥ 3:1 contrast against the card surface).

---

## Execution Environment

Models were trained and the app was run on **BITS Virtual Lab** — screenshot
included in the submitted PDF.
