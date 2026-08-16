# Telco Customer Churn — Comparing Five Classification Models

**BITS Pilani WILP · M.Tech (AIML) · Machine Learning · Assignment 2**

I trained five classifiers on the IBM Telco Customer Churn dataset, evaluated
them on the same held-out test split, and deployed a Streamlit dashboard where
you can upload a CSV and compare the models interactively.

---

## a. Problem Statement

A telecom company loses about a quarter of its customers. Getting a new customer
costs a lot more than keeping an existing one, so if the company can identify
which customers are likely to leave, it can try to keep them with a retention
call or a better offer.

As a machine learning problem this is binary classification: given 19 attributes
about a customer (demographics, which services they subscribe to, their contract
type and their billing history), predict whether `Churn` is Yes or No.

The two kinds of mistakes are not equally costly, and this affected most of my
choices later on. If the model says a customer is safe but they actually leave
(false negative), the company loses that customer. If the model flags a loyal
customer (false positive), the company wastes one retention call. The first
mistake is much more expensive than the second. This is why I did not treat
accuracy as the main metric and used MCC instead.

---

## b. Dataset Description

| Property | Value |
| --- | --- |
| Source | IBM Sample Data Sets — Telco Customer Churn (public, also on Kaggle) |
| Instances | 7,043 customers (requirement was ≥ 500) |
| Columns | 21 raw columns, 19 usable features after dropping `customerID` and the label (requirement was ≥ 12) |
| After encoding | 45 columns |
| Task | Binary classification |
| Class balance | 5,174 stayed (73.46%) and 1,869 churned (26.54%) |
| Split | 75/25 stratified, so 5,282 for training and 1,761 for testing, `random_state=42` |

### Features

- **Demographic (4):** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Services (9):** `PhoneService`, `MultipleLines`, `InternetService`,
  `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`,
  `StreamingTV`, `StreamingMovies`
- **Account (6):** `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`,
  `MonthlyCharges`, `TotalCharges`

### Data problems I had to fix

1. **`TotalCharges` was not numeric.** 11 rows contain a single space instead of
   a number. I checked these rows and all 11 are customers with `tenure = 0`,
   meaning they had just joined and had not been billed yet. If you load the CSV
   normally, pandas reads the whole column as text and scaling breaks. I fixed
   this with `pd.to_numeric(errors="coerce")` so the blanks become NaN, and then
   median imputation inside the pipeline fills them.
2. **`customerID` is useless for prediction.** It is a unique key, so I dropped
   it. If I left it in, the tree models could just memorise individual rows.
3. **Extra whitespace in text columns.** I strip it before encoding, otherwise
   `"Yes"` and `"Yes "` would be treated as two different categories.

### Preprocessing

All five models use the **same `ColumnTransformer`**, so any difference in the
results comes from the algorithm and not from different feature engineering:

- Numeric columns: median imputation, then `StandardScaler`
- Categorical columns: most-frequent imputation, then
  `OneHotEncoder(handle_unknown="ignore")`

Scaling was necessary here, not optional. kNN uses Euclidean distance, and
`TotalCharges` goes up to about 8,700 while `tenure` only goes up to 72. Without
scaling, `TotalCharges` would dominate the distance calculation and the other
40+ binary columns would barely matter.

I put the preprocessing **inside** each saved `Pipeline`. This turned out to be
very useful for the app, because it means the Streamlit app can accept a raw CSV
and does not have to redo any encoding itself.

---

## c. GitHub Repository Link

> **https://github.com/2025ac05185-dotcom/streamlit_app**

### Live Streamlit App

> **https://customerchurntelco.streamlit.app**

### Repository structure

```
streamlit_app/
├── app.py                  # Streamlit application
├── requirements.txt        # Pinned dependencies
├── README.md               # This file
├── test_data.csv           # Held-out test split (1,761 rows, raw format)
├── .gitignore
├── .devcontainer/
│   └── devcontainer.json
├── .streamlit/
│   └── config.toml         # Theme and chart colours
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── model/
    ├── preprocessing.py    # Shared preprocessing pipeline
    ├── train_models.py     # Trains and evaluates all five models
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── random_forest.joblib
    ├── metrics.json        # Results in JSON
    └── metrics.md          # Results as a Markdown table
```

### How to run it

```bash
git clone https://github.com/2025ac05185-dotcom/streamlit_app.git
cd streamlit_app
pip install -r requirements.txt

python model/train_models.py   # retrains and rewrites test_data.csv + metrics
streamlit run app.py           # opens the dashboard on localhost:8501
```

---

## d. Models Used

Five classifiers, all using the same preprocessing pipeline and trained on the
same 5,282 rows.

| # | Model | Settings |
| --- | --- | --- |
| 1 | Logistic Regression | `max_iter=2000`, `class_weight="balanced"` |
| 2 | Decision Tree | `max_depth=6`, `min_samples_leaf=25`, `class_weight="balanced"` |
| 3 | k-Nearest Neighbours | `n_neighbors=25`, `weights="distance"` |
| 4 | Naive Bayes | `GaussianNB` |
| 5 | Random Forest (Ensemble) | `n_estimators=400`, `min_samples_leaf=5`, `class_weight="balanced_subsample"` |

I used `class_weight="balanced"` wherever the algorithm supports it. When I first
tried without it, the models mostly predicted "No" for everyone. That gives about
73% accuracy, which looks fine until you notice it catches almost no churners at
all, which makes the model useless for the actual business problem.

### Evaluation metrics on the test set (1,761 rows, threshold 0.50)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.7496 | **0.8460** | 0.5182 | 0.7944 | 0.6272 | 0.4743 |
| Decision Tree | 0.7348 | 0.8320 | 0.5000 | 0.8201 | 0.6212 | 0.4667 |
| kNN | **0.7808** | 0.8154 | **0.5910** | 0.5632 | 0.5768 | 0.4292 |
| Naive Bayes | 0.6979 | 0.8112 | 0.4616 | **0.8373** | 0.5951 | 0.4284 |
| Random Forest (Ensemble) | 0.7740 | 0.8437 | 0.5561 | 0.7323 | **0.6322** | **0.4828** |

One thing I checked as a sanity test: if you just predict "No churn" for every
customer, you get **0.7348 accuracy** on this test set, and an MCC of 0. Compare
that to the table. The Decision Tree gets exactly the same accuracy (0.7348) and
Naive Bayes actually does worse (0.6979), but both have an MCC above 0.46. So if
I only looked at accuracy, I would have concluded that two of my five models were
no better than doing nothing, which is clearly wrong. This is the main reason the
assignment asks for six metrics instead of one.

---

### Observations on model performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | This was the best model at ranking customers (AUC 0.8460), which surprised me because it is the simplest one I tried. It seems churn in this dataset is close to linearly separable in log-odds space. Looking at the coefficients, tenure (−1.14) and a two-year contract (−0.74) reduce churn risk, while fibre-optic internet (+0.68) and month-to-month contracts (+0.64) increase it. Because of the class weighting it flags 716 of the 1,761 customers, giving recall 0.79 but precision only 0.52. Its accuracy (0.7496) is barely above the baseline, but that is the expected result of that trade-off rather than a problem with the model. |
| **Decision Tree** | The first split is on `Contract_Month-to-month`, which matches what the other models say is the strongest signal. Recall is 0.8201, the highest apart from Naive Bayes, and the tree is easy to explain since it has 55 leaves at depth 6. The interesting part is that its accuracy is exactly 0.7348, identical to the majority-class baseline, while its MCC is 0.4667 instead of 0. I had to limit the depth because when I left it unpruned it memorised the training data and did noticeably worse on the test set. |
| **kNN** | This model has the highest accuracy (0.7808) and the highest precision (0.5910), but it is the weakest model where it matters. It only flags 445 customers and misses about 44% of the actual churners (recall 0.5632), so it is the most conservative of the five. Its AUC is also the lowest (0.8154), which tells me the ranking itself is weaker and it is not just a threshold problem. I think this is the curse of dimensionality: with 45 encoded columns, most of them sparse one-hot columns, the distance measure gets diluted. It is also the slowest at prediction time because it stores 200 KB of training data that has to be searched for every query. |
| **Naive Bayes** | Best recall of all five models (0.8373) but the worst accuracy (0.6979). It flags 847 of 1,761 customers, which is almost half the customer base and probably too many for a real retention team to call. The independence assumption is badly violated in this dataset, because `InternetService`, `OnlineSecurity`, `TechSupport` and `StreamingTV` are obviously related (if you have no internet you cannot have the add-ons), so the same evidence gets counted several times and the probabilities get pushed towards 0 and 1. The probabilities are therefore not well calibrated, but the ranking is still reasonable (AUC 0.8112) and it trains almost instantly. It could work as a cheap first filter, but not as the final decision maker. |
| **Random Forest (Ensemble)** | Best F1 (0.6322) and best MCC (0.4828), so it is the most balanced model on the two metrics that consider both types of error. Bagging 400 trees fixes the variance problem of the single tree while keeping most of its recall: 0.7323 recall at 0.5561 precision, catching 342 of the 467 churners in the test set. Its AUC (0.8437) is only 0.0024 behind Logistic Regression, so for ranking the two are basically equal. The feature importances agree with the logistic regression coefficients, with `tenure` (0.127), `Contract_Month-to-month` (0.121) and `TotalCharges` (0.107) at the top. The downside is size: about 7.3 MB on disk compared to 2.6 KB for Logistic Regression. |
| **Overall winner for this dataset** | **Random Forest**, based on MCC (0.4828) and F1 (0.6322), which are the metrics that stay meaningful with a 73/27 class imbalance. I should add one caveat though. Logistic Regression actually beats it on AUC (0.8460 vs 0.8437) while being a tiny fraction of the size and having coefficients you can read directly. So Random Forest wins on the assignment's metrics, but if this were a real retention system where someone has to explain to a customer or a manager why a particular account was flagged, I would choose Logistic Regression. |

### A note on the threshold

Every metric above except AUC is calculated at the default 0.50 cut-off. AUC does
not depend on the threshold, which is why it is the fairest single number for
comparing the five algorithms. It is also why I added a **threshold slider** to
the app: moving it recalculates precision, recall, F1 and MCC live while the ROC
curve stays the same, so the trade-off is something you can see instead of
something you have to take my word for.

I also found that **0.50 is not the best threshold**. Sweeping the Random Forest
across the whole 0.05 to 0.95 range, F1 and MCC both peak at **0.47**:

| Threshold | F1 | MCC |
| :--- | ---: | ---: |
| 0.50 (default) | 0.6322 | 0.4828 |
| **0.47 (best)** | **0.6443** | **0.4997** |

That is +0.0121 F1 and +0.0169 MCC just from changing the cut-off, with no
retraining and no new features. The **Threshold analysis** tab in the app runs
this sweep live for whichever model is selected and has buttons to jump straight
to the best-F1 or best-MCC threshold.

---

## Streamlit Application

The app is deployed at **https://customerchurntelco.streamlit.app**.

| Requirement | How I implemented it |
| --- | --- |
| Dataset upload (CSV) | File uploader in the sidebar. If nothing is uploaded it falls back to the bundled `test_data.csv`, so the app is never blank when you first open it |
| Model selection dropdown | All five trained pipelines, switchable without reloading |
| Display of evaluation metrics | All six metrics shown as tiles at the top, recalculated on whatever data is loaded |
| Confusion matrix / classification report | Interactive heatmap and the per-class precision/recall/F1 report, side by side |

The layout is four tabs (**Performance**, **Threshold analysis**, **Model
comparison** and **Predictions**) underneath a fixed row of six metric tiles.
The tiles also show where the selected model ranks against the other four, for
example "Best of 5" or "2nd of 5 · −0.0062 vs best".

Things I added beyond the requirements:

- **Threshold analysis tab** with a live precision/recall/F1 sweep across the
  whole 0.05–0.95 range, the current cut-off marked on both the sweep and the
  ROC curve, and buttons to jump to the best-F1 or best-MCC threshold
- **Threshold economics**, showing churners caught, false alarm rate and flag
  rate, so the threshold can be read as actual workload for a retention team
- **Confusion matrix shaded by row percentage** rather than raw count, because
  the 1,021 true negatives were washing out the three cells that actually matter
- **ROC overlay of all five models** and a comparison bar chart that can be
  re-sorted by any of the six metrics
- **Row-level predictions** with churn probability, prediction and whether it was
  correct, a filter to show only the misclassified rows, and a CSV download

### Implementation notes

The predicted probabilities for all five models are calculated **once per
uploaded file** and cached using a hash of the CSV contents. Scoring all five
pipelines takes about 78 ms on the 1,761-row split, and before I added the cache
that was happening on every single slider movement. Now moving the threshold only
re-compares an already-computed array, so it responds in about 155 ms.

I used Altair for the charts instead of static Matplotlib images so that every
point has a tooltip. I also checked the colour palette for colour-vision
deficiency: the worst adjacent pair is ΔE 8.4 under protanopia and every colour
has at least 3:1 contrast against the card background. For the three lines in the
threshold sweep I used a stricter check (worst ΔE 9.4 under deuteranopia) because
those lines cross each other, so being next to each other in the legend is not
enough.

---

## Execution Environment

The models were trained and the app was tested on **BITS Virtual Lab**. A
screenshot is included in the submitted PDF.
