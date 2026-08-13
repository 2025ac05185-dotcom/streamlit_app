"""
Telco Churn Model Explorer - Streamlit front end.

Upload a raw test CSV, pick one of five trained classifiers, and inspect its
behaviour on that data: the six assignment metrics, a confusion matrix, a
classification report, threshold analysis, a side-by-side comparison of all
models, and row-level predictions you can download.

Every model's predicted probabilities are computed once per uploaded file and
cached, so moving the decision threshold re-scores instantly instead of
re-running five pipelines.

Launch locally with:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "model"))

from preprocessing import split_features_and_target  # noqa: E402

# ------------------------------------------------------------------- config --
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest.joblib",
}

MODEL_NOTES = {
    "Logistic Regression": "Linear log-odds model, class-weighted. Strongest ranking in the set, deliberately loose threshold.",
    "Decision Tree": "Depth capped at 6 with a 25-row leaf minimum, so the splits stay readable.",
    "kNN": "25 distance-weighted neighbours in the scaled 45-column encoded space.",
    "Naive Bayes": "Gaussian likelihoods, features assumed independent — a strong assumption here.",
    "Random Forest (Ensemble)": "400 bagged trees with balanced subsample weighting.",
}

METRIC_HELP = {
    "Accuracy": "Share of all customers classified correctly.",
    "AUC": "Ranking quality across every threshold. Unaffected by the slider.",
    "Precision": "Of those flagged as churning, how many actually churned.",
    "Recall": "Of those who actually churned, how many were caught.",
    "F1": "Harmonic mean of precision and recall.",
    "MCC": "Correlation between prediction and truth. Robust to class imbalance.",
}

METRICS = list(METRIC_HELP)

# ------------------------------------------------------------------ palette --
# Validated categorical ramp. Adjacent pairs (lines, bars): worst CVD dE 8.4,
# normal-vision 19.3. The three-slot subset below also clears the stricter
# all-pairs gate, which is what crossing lines need.
SURFACE = "#181817"
INK = "#f2f1ec"
INK_SECONDARY = "#c3c2b7"
INK_MUTED = "#898781"
GRID = "#2c2c2a"
AXIS = "#383835"
RECEDE = "#3a3a37"

SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]
TRIPLE = SERIES[:3]  # clears all-pairs CVD — safe for lines that cross
BLUE = SERIES[0]
BLUE_RAMP = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"]

st.set_page_config(
    page_title="Telco Churn Model Explorer",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------- data + cache --
@st.cache_resource(show_spinner=False)
def load_models() -> dict:
    return {
        name: joblib.load(ROOT / "model" / fname)
        for name, fname in MODEL_FILES.items()
        if (ROOT / "model" / fname).exists()
    }


@st.cache_data(show_spinner=False)
def load_training_metrics() -> dict:
    path = ROOT / "model" / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data(show_spinner=False)
def load_bundled_test() -> pd.DataFrame | None:
    path = ROOT / "test_data.csv"
    return pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() else None


def fingerprint(frame: pd.DataFrame) -> str:
    """Stable content hash, used as the cache key for everything downstream."""
    digest = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    return hashlib.sha256(digest).hexdigest()[:16]


@st.cache_data(show_spinner="Scoring every model…")
def score_all(data_key: str, _models: dict, _X: pd.DataFrame) -> dict:
    """Predicted churn probabilities per model — the only expensive step.

    Keyed on the data fingerprint alone, so changing the threshold or the
    selected model re-uses this instead of re-running five pipelines
    (measured at ~78 ms per rerun on the bundled 1,761-row split).
    """
    return {name: model.predict_proba(_X)[:, 1] for name, model in _models.items()}


def evaluate(y_true, y_pred, y_prob) -> dict:
    # AUC is undefined when an uploaded CSV holds a single class; report NaN
    # rather than letting sklearn warn and hand back a bare nan.
    single_class = len(np.unique(y_true)) < 2
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": float("nan") if single_class else roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


@st.cache_data(show_spinner=False)
def threshold_sweep(data_key: str, name: str, _prob, _y) -> pd.DataFrame:
    """Precision / Recall / F1 / MCC across the full threshold grid."""
    rows = []
    for t in np.round(np.arange(0.05, 0.9501, 0.01), 2):
        pred = (_prob >= t).astype(int)
        rows.append(
            {
                "Threshold": float(t),
                "Precision": precision_score(_y, pred, zero_division=0),
                "Recall": recall_score(_y, pred, zero_division=0),
                "F1": f1_score(_y, pred, zero_division=0),
                "MCC": matthews_corrcoef(_y, pred),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def all_model_metrics(data_key: str, threshold: float, _probs: dict, _y) -> pd.DataFrame:
    """Every model scored at one threshold — cheap, since probabilities are cached."""
    rows = {
        name: evaluate(_y, (p >= threshold).astype(int), p)
        for name, p in _probs.items()
    }
    frame = pd.DataFrame(rows).T
    frame.index.name = "ML Model Name"
    return frame


# ------------------------------------------------------------------- chrome --
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
          :root {{
            --sp-1:.25rem; --sp-2:.5rem; --sp-3:.75rem; --sp-4:1rem;
            --sp-5:1.5rem; --sp-6:2rem;
            --r-sm:8px; --r-md:12px; --r-lg:16px;
            --ink:{INK}; --ink-2:{INK_SECONDARY}; --ink-3:{INK_MUTED};
            --line:{GRID}; --card:{SURFACE}; --accent:{BLUE};
          }}
          .block-container {{
            padding-top: 2rem; padding-bottom: 3.5rem; max-width: 1560px;
          }}
          header[data-testid="stHeader"] {{ background: transparent; }}
          #MainMenu, footer {{ visibility: hidden; }}

          /* focus is never removed, only restyled */
          *:focus-visible {{
            outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 4px;
          }}
          @media (prefers-reduced-motion: reduce) {{
            * {{ transition: none !important; animation: none !important; }}
          }}

          /* ---- hero ---- */
          .hero {{
            border: 1px solid var(--line); border-radius: var(--r-lg);
            padding: 1.35rem 1.6rem; margin-bottom: var(--sp-5);
            background:
              radial-gradient(115% 170% at 0% 0%, rgba(57,135,229,.15) 0%, rgba(57,135,229,0) 58%),
              var(--card);
          }}
          .hero h1 {{
            margin: 0 0 .3rem; font-size: 1.85rem; font-weight: 600;
            letter-spacing: -.024em; color: var(--ink);
          }}
          .hero p {{
            margin: 0; color: var(--ink-2); font-size: .94rem;
            max-width: 74ch; line-height: 1.55;
          }}
          .chips {{ display: flex; flex-wrap: wrap; gap: var(--sp-2); margin-top: 1.05rem; }}
          .chip {{
            display: inline-flex; align-items: center; gap: .42rem;
            border: 1px solid var(--line); border-radius: 999px;
            padding: .3rem .72rem; font-size: .78rem; color: var(--ink-2);
            background: rgba(255,255,255,.02); white-space: nowrap;
          }}
          .chip b {{ color: var(--ink); font-weight: 600; }}
          .dot {{ width: 6px; height: 6px; border-radius: 50%; background: var(--accent); flex: none; }}

          /* ---- stat tiles ---- */
          .tile {{
            position: relative; height: 100%;
            border: 1px solid var(--line); border-radius: var(--r-md);
            padding: .8rem .9rem .95rem; background: var(--card);
            transition: border-color .15s ease;
          }}
          .tile:hover {{ border-color: #3d3d3a; }}
          .tile .top {{
            display: flex; align-items: baseline; justify-content: space-between;
            gap: var(--sp-2); margin-bottom: .28rem;
          }}
          .tile .label {{
            font-size: .72rem; text-transform: uppercase; letter-spacing: .07em;
            color: var(--ink-3); font-weight: 600;
          }}
          .rank {{
            font-size: .66rem; font-weight: 600; letter-spacing: .02em;
            padding: .12rem .4rem; border-radius: 999px; white-space: nowrap;
            color: var(--ink-3); border: 1px solid var(--line);
          }}
          .rank.lead {{
            color: #9ec5f4; border-color: rgba(57,135,229,.45);
            background: rgba(57,135,229,.12);
          }}
          .tile .value {{
            font-size: 1.8rem; font-weight: 600; color: var(--ink);
            line-height: 1.1; letter-spacing: -.022em;
          }}
          .tile .meter {{
            margin-top: .6rem; height: 4px; border-radius: 999px;
            background: rgba(255,255,255,.07); overflow: hidden;
          }}
          .tile .meter span {{
            display: block; height: 100%; border-radius: 999px; background: var(--accent);
          }}
          .tile .foot {{ margin-top: .42rem; font-size: .715rem; color: var(--ink-3); line-height: 1.45; }}

          /* ---- section label ---- */
          .section {{
            display: flex; align-items: center; gap: var(--sp-3);
            font-size: .72rem; text-transform: uppercase; letter-spacing: .085em;
            color: var(--ink-3); font-weight: 600; margin: .1rem 0 .65rem;
          }}
          .section::after {{
            content: ""; flex: 1; height: 1px; background: var(--line);
          }}

          /* ---- tabs (Streamlit 1.61 uses role=, not baseweb attrs) ---- */
          .stTabs [role="tablist"] {{ gap: .12rem; border-bottom: 1px solid var(--line); }}
          .stTabs [role="tab"] {{
            padding: .55rem .95rem; color: var(--ink-3);
            font-size: .9rem; font-weight: 500; border-radius: var(--r-sm) var(--r-sm) 0 0;
            transition: color .12s ease, background .12s ease;
          }}
          .stTabs [role="tab"]:hover {{ color: var(--ink-2); background: rgba(255,255,255,.03); }}
          .stTabs [role="tab"][aria-selected="true"] {{ color: var(--ink); font-weight: 600; }}

          /* ---- sidebar ---- */
          section[data-testid="stSidebar"] {{ border-right: 1px solid var(--line); }}
          section[data-testid="stSidebar"] .block-container {{ padding-top: 1.5rem; }}

          /* ---- charts ---- */
          div[data-testid="stVegaLiteChart"] {{
            border: 1px solid var(--line); border-radius: var(--r-md);
            padding: .75rem; background: var(--card);
          }}
          hr {{ border-color: var(--line); }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def tile(
    label: str,
    value: float,
    note: str,
    fill: float | None = None,
    text: str | None = None,
    rank: str | None = None,
    lead: bool = False,
) -> str:
    """Stat tile: label, optional rank chip, hero figure, 0-1 meter, caption."""
    raw_fill = value if fill is None else fill
    # An undefined metric (single-class upload) must not emit width:nan%.
    pct = 0.0 if np.isnan(raw_fill) else max(0.0, min(1.0, raw_fill)) * 100
    if text is not None:
        shown = text
    elif np.isnan(value):
        shown = "—"
    else:
        shown = f"{value:.4f}"
    chip = f'<span class="rank{" lead" if lead else ""}">{rank}</span>' if rank else ""
    return (
        f'<div class="tile"><div class="top"><span class="label">{label}</span>{chip}</div>'
        f'<div class="value">{shown}</div>'
        f'<div class="meter"><span style="width:{pct:.1f}%"></span></div>'
        f'<div class="foot">{note}</div></div>'
    )


def style(chart: alt.Chart, height: int = 300, legend: bool = True) -> alt.Chart:
    """Recessive hairline chrome, transparent surface, no view border."""
    styled = (
        chart.properties(height=height, background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=INK_MUTED,
            titleColor=INK_SECONDARY,
            labelFontSize=11,
            titleFontSize=11,
            titleFontWeight=500,
            titlePadding=10,
            gridColor=GRID,
            gridWidth=1,
            domainColor=AXIS,
            tickColor=AXIS,
            labelFont="system-ui",
            titleFont="system-ui",
        )
        .configure_title(
            color=INK, fontSize=13, fontWeight=600, anchor="start", font="system-ui"
        )
    )
    if not legend:
        return styled.configure_legend(disable=True)
    return styled.configure_legend(
        labelColor=INK_SECONDARY,
        titleColor=INK_MUTED,
        labelFontSize=11,
        titleFontSize=10,
        symbolStrokeWidth=3,
        symbolSize=90,
        labelFont="system-ui",
        titleFont="system-ui",
        orient="bottom",
        direction="vertical",
        columns=1,
        offset=12,
        labelLimit=280,
    )


def rule_at(value: float, label: str) -> alt.Chart:
    """A hairline marker for the live threshold, with a text tag."""
    frame = pd.DataFrame({"Threshold": [value], "tag": [label]})
    rule = alt.Chart(frame).mark_rule(
        color=INK_MUTED, strokeWidth=1, strokeDash=[3, 3]
    ).encode(x=alt.X("Threshold:Q"))
    tag = alt.Chart(frame).mark_text(
        align="left", dx=6, dy=-6, baseline="top", fontSize=10,
        font="system-ui", color=INK_SECONDARY,
    ).encode(x=alt.X("Threshold:Q"), y=alt.value(0), text="tag:N")
    return rule + tag


inject_css()

# ------------------------------------------------------------------ sidebar --
models = load_models()
if not models:
    st.error(
        "No trained models found in `model/`. Run `python model/train_models.py` "
        "from the repository root first."
    )
    st.stop()

st.sidebar.markdown('<div class="section">Data</div>', unsafe_allow_html=True)
uploaded = st.sidebar.file_uploader(
    "Test data (CSV)",
    type="csv",
    label_visibility="collapsed",
    help="Raw Telco-format CSV including the Churn column.",
)

if uploaded is not None:
    raw = pd.read_csv(uploaded, dtype=str, keep_default_na=False)
    source_label = f"Uploaded — {uploaded.name}"
else:
    raw = load_bundled_test()
    source_label = "Bundled held-out split"

if raw is None:
    st.warning("Upload a CSV to begin.")
    st.stop()

st.sidebar.caption(
    "Raw Telco-format CSV including the `Churn` column. Leave empty to score "
    "the bundled held-out split."
)

st.sidebar.markdown('<div class="section">Model</div>', unsafe_allow_html=True)
selected = st.sidebar.selectbox(
    "Model", list(models.keys()), index=4, label_visibility="collapsed"
)
st.sidebar.caption(MODEL_NOTES[selected])

st.sidebar.markdown('<div class="section">Decision threshold</div>', unsafe_allow_html=True)
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.50
st.sidebar.slider(
    "Decision threshold",
    min_value=0.05,
    max_value=0.95,
    step=0.01,
    key="threshold",
    label_visibility="collapsed",
    help="Probability above which a customer is flagged as churning.",
)
threshold = st.session_state.threshold
st.sidebar.caption(
    "Lower it to catch more churners at the cost of false alarms. The "
    "**Threshold analysis** tab suggests the optimum."
)

# --------------------------------------------------------------------- data --
try:
    X, y = split_features_and_target(raw)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

data_key = fingerprint(raw)
try:
    probs = score_all(data_key, models, X)
except Exception as exc:  # noqa: BLE001
    st.error(f"Scoring failed — does the CSV match the Telco schema? ({exc})")
    st.stop()

prob = probs[selected]
pred = (prob >= threshold).astype(int)
metrics = evaluate(y, pred, prob)
comparison = all_model_metrics(data_key, threshold, probs, y)
churn_rate = float(np.mean(y))

# --------------------------------------------------------------------- hero --
st.markdown(
    f"""
    <div class="hero">
      <h1>Telco Churn Model Explorer</h1>
      <p>Five classifiers over one shared preprocessing pipeline, scored live on
         whatever CSV you feed them. Move the decision threshold to trade recall
         against false alarms and watch every metric respond.</p>
      <div class="chips">
        <span class="chip"><span class="dot"></span>{selected}</span>
        <span class="chip">Threshold <b>{threshold:.2f}</b></span>
        <span class="chip">{source_label}</span>
        <span class="chip"><b>{len(raw):,}</b> rows &times; {raw.shape[1]} cols</span>
        <span class="chip">Base churn rate <b>{churn_rate:.1%}</b></span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------- kpi tiles --
ordinal = {1: "Best of 5", 2: "2nd of 5", 3: "3rd of 5", 4: "4th of 5", 5: "5th of 5"}
cols = st.columns(6, gap="small")
for col, name in zip(cols, METRICS):
    value = metrics[name]
    column = comparison[name]
    if column.notna().any():
        place = int(column.rank(ascending=False, method="min")[selected])
        best = float(column.max())
        gap = "" if place == 1 else f"<br>{value - best:+.4f} vs best"
        rank_text, lead = ordinal.get(place, f"{place} of 5"), place == 1
    else:
        rank_text, lead, gap = None, False, ""
    col.markdown(
        tile(
            name,
            value,
            f"{METRIC_HELP[name]}{gap}",
            fill=(value + 1) / 2 if name == "MCC" else value,
            rank=rank_text,
            lead=lead,
        ),
        unsafe_allow_html=True,
    )

st.write("")

tab_perf, tab_thresh, tab_compare, tab_rows = st.tabs(
    ["Performance", "Threshold analysis", "Model comparison", "Predictions"]
)

# --------------------------------------------------------- tab: performance --
with tab_perf:
    # labels=[0, 1] forces a 2x2 result even when an uploaded CSV holds a single
    # class and no positives are predicted — otherwise sklearn returns 1x1 and
    # the unpack below raises.
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.markdown('<div class="section">Confusion matrix</div>', unsafe_allow_html=True)
        # Colour by row share, not raw count: the 1,021 true negatives would
        # otherwise swamp the ramp and flatten the three cells that matter.
        # An absent class gives a zero row total; clamp so the shares stay 0.
        row_totals = np.maximum(cm.sum(axis=1, keepdims=True), 1)
        cm_df = pd.DataFrame(
            [
                {
                    "Actual": a,
                    "Predicted": p,
                    "Count": int(cm[i][j]),
                    "Rate": float(cm[i][j] / row_totals[i][0]),
                }
                for i, a in enumerate(["Stayed", "Churned"])
                for j, p in enumerate(["Stayed", "Churned"])
            ]
        )
        order = ["Stayed", "Churned"]
        base = alt.Chart(cm_df).encode(
            x=alt.X("Predicted:N", sort=order, title="Predicted",
                    axis=alt.Axis(labelAngle=0, domain=False, ticks=False, grid=False,
                                  labelFontSize=12, labelPadding=8)),
            y=alt.Y("Actual:N", sort=order, title="Actual",
                    axis=alt.Axis(domain=False, ticks=False, grid=False,
                                  labelFontSize=12, labelPadding=8)),
        )
        # A surface-coloured stroke is the 2px gap between cells, not a border.
        cells = base.mark_rect(stroke=SURFACE, strokeWidth=4, cornerRadius=6).encode(
            color=alt.Color("Rate:Q", scale=alt.Scale(range=BLUE_RAMP[::-1], domain=[0, 1]),
                            legend=None),
            tooltip=[
                alt.Tooltip("Actual:N", title="Actually"),
                alt.Tooltip("Predicted:N", title="Predicted"),
                alt.Tooltip("Count:Q", title="Customers", format=","),
                alt.Tooltip("Rate:Q", title="Share of actual class", format=".1%"),
            ],
        )
        counts = base.mark_text(fontSize=22, fontWeight=600, font="system-ui", dy=-9).encode(
            text=alt.Text("Count:Q", format=","),
            # Light cells (low rate) take dark ink; dark cells take white.
            color=alt.condition(alt.datum.Rate > 0.5, alt.value("#ffffff"), alt.value("#0b0b0b")),
        )
        rates = base.mark_text(fontSize=11, font="system-ui", dy=13).encode(
            text=alt.Text("Rate:Q", format=".1%"),
            color=alt.condition(alt.datum.Rate > 0.5,
                                alt.value("rgba(255,255,255,0.78)"),
                                alt.value("rgba(11,11,11,0.72)")),
        )
        st.altair_chart(style(cells + counts + rates, height=250, legend=False),
                        width="stretch")
        st.caption(
            f"**{tp}** churners caught · **{fn}** missed · **{fp}** loyal customers "
            "flagged unnecessarily. Cells are shaded by share of each **actual** "
            "class, so a dark diagonal is a good model."
        )

    with right:
        st.markdown('<div class="section">Classification report</div>', unsafe_allow_html=True)
        report = classification_report(
            y, pred, labels=[0, 1], target_names=["Stayed", "Churned"],
            output_dict=True, zero_division=0,
        )
        st.dataframe(pd.DataFrame(report).transpose().round(3), width="stretch")

        st.markdown('<div class="section">Threshold economics</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.markdown(
            tile("Caught", tp / (tp + fn) if tp + fn else 0.0,
                 f"{tp:,} of {tp + fn:,} churners"), unsafe_allow_html=True)
        c2.markdown(
            tile("False alarms", fp / (fp + tn) if fp + tn else 0.0,
                 f"{fp:,} of {fp + tn:,} loyal"), unsafe_allow_html=True)
        c3.markdown(
            tile("Flag rate", (tp + fp) / cm.sum(), f"{tp + fp:,} customers contacted"),
            unsafe_allow_html=True)

# ------------------------------------------------- tab: threshold analysis --
with tab_thresh:
    sweep = threshold_sweep(data_key, selected, prob, y)
    best_f1 = sweep.loc[sweep["F1"].idxmax()]
    best_mcc = sweep.loc[sweep["MCC"].idxmax()]

    def _apply(value: float) -> None:
        st.session_state.threshold = float(value)

    st.markdown('<div class="section">Suggested operating points</div>',
                unsafe_allow_html=True)
    o1, o2, o3, o4 = st.columns([1, 1, 1, 1.4], gap="medium")
    o1.markdown(tile("Best F1", float(best_f1["F1"]),
                     f"at threshold {best_f1['Threshold']:.2f}"), unsafe_allow_html=True)
    o2.markdown(tile("Best MCC", float(best_mcc["MCC"]),
                     f"at threshold {best_mcc['Threshold']:.2f}",
                     fill=(float(best_mcc["MCC"]) + 1) / 2), unsafe_allow_html=True)
    o3.markdown(tile("Current", float(metrics["MCC"]),
                     f"MCC at threshold {threshold:.2f}",
                     fill=(float(metrics["MCC"]) + 1) / 2), unsafe_allow_html=True)
    with o4:
        st.write("")
        st.button(f"Apply best-F1 threshold ({best_f1['Threshold']:.2f})",
                  on_click=_apply, args=(best_f1["Threshold"],), width="stretch")
        st.button(f"Apply best-MCC threshold ({best_mcc['Threshold']:.2f})",
                  on_click=_apply, args=(best_mcc["Threshold"],), width="stretch")

    st.write("")
    sweep_col, roc_col = st.columns([1.25, 1], gap="large")

    with sweep_col:
        long = sweep.melt("Threshold", value_vars=["Precision", "Recall", "F1"],
                          var_name="Metric", value_name="Value")
        lines = (
            alt.Chart(long)
            .mark_line(strokeWidth=2, strokeCap="round")
            .encode(
                # nice=False, or Vega pads the axis out to 0.00-1.00 and the
                # lines float inside a range the sweep never covers.
                x=alt.X("Threshold:Q", scale=alt.Scale(domain=[0.05, 0.95], nice=False),
                        title="Decision threshold"),
                y=alt.Y("Value:Q", scale=alt.Scale(domain=[0, 1]), title=None,
                        axis=alt.Axis(format=".0%")),
                color=alt.Color("Metric:N",
                                scale=alt.Scale(domain=["Precision", "Recall", "F1"],
                                                range=TRIPLE),
                                legend=alt.Legend(title=None)),
                tooltip=[alt.Tooltip("Metric:N"),
                         alt.Tooltip("Threshold:Q", format=".2f"),
                         alt.Tooltip("Value:Q", format=".4f")],
            )
        )
        st.altair_chart(
            style(lines + rule_at(threshold, f"now {threshold:.2f}"), height=330)
            .properties(title="Precision / Recall / F1 across the threshold"),
            width="stretch",
        )
        st.caption(
            "Precision and recall move in opposite directions by construction; "
            "F1 peaks where the trade-off balances. The dashed line is the "
            "threshold currently set in the sidebar."
        )

    with roc_col:
        fpr, tpr, _ = roc_curve(y, prob) if len(np.unique(y)) > 1 else (
            np.array([0, 1]), np.array([0, 1]), None)
        step = max(1, len(fpr) // 400)
        roc_df = pd.DataFrame({"FPR": fpr[::step], "TPR": tpr[::step], "Series": selected})
        diag = pd.DataFrame({"FPR": [0.0, 1.0], "TPR": [0.0, 1.0], "Series": "Random"})
        axis_kw = dict(scale=alt.Scale(domain=[0, 1]),
                       axis=alt.Axis(format=".0%", tickCount=5))
        curve = (
            alt.Chart(pd.concat([roc_df, diag], ignore_index=True))
            .mark_line(strokeWidth=2, strokeCap="round")
            .encode(
                x=alt.X("FPR:Q", title="False positive rate", **axis_kw),
                y=alt.Y("TPR:Q", title="True positive rate", **axis_kw),
                color=alt.Color("Series:N",
                                scale=alt.Scale(domain=[selected, "Random"],
                                                range=[BLUE, INK_MUTED]),
                                legend=alt.Legend(title=None)),
                strokeDash=alt.condition(alt.datum.Series == "Random",
                                         alt.value([4, 4]), alt.value([0])),
                detail="Series:N",
                tooltip=[alt.Tooltip("FPR:Q", format=".2%"),
                         alt.Tooltip("TPR:Q", format=".2%")],
            )
        )
        # Where the sidebar threshold actually puts you on that curve.
        op = pd.DataFrame({
            "FPR": [fp / (fp + tn) if fp + tn else 0.0],
            "TPR": [tp / (tp + fn) if tp + fn else 0.0],
        })
        point = alt.Chart(op).mark_point(
            size=150, filled=True, color=BLUE, stroke=SURFACE, strokeWidth=2
        ).encode(x=alt.X("FPR:Q", **axis_kw), y=alt.Y("TPR:Q", **axis_kw),
                 tooltip=[alt.Tooltip("FPR:Q", title="False positive rate", format=".2%"),
                          alt.Tooltip("TPR:Q", title="True positive rate", format=".2%")])
        op_label = alt.Chart(op).mark_text(
            align="left", dx=10, dy=4, fontSize=10, font="system-ui", color=INK_SECONDARY
        ).encode(x=alt.X("FPR:Q", **axis_kw), y=alt.Y("TPR:Q", **axis_kw),
                 text=alt.value(f"threshold {threshold:.2f}"))
        st.altair_chart(
            style(curve + point + op_label, height=330)
            .properties(title=f"ROC — AUC {metrics['AUC']:.4f}"),
            width="stretch",
        )
        st.caption(
            "The marked point is where the current threshold sits on the curve. "
            "Moving the slider slides the point; it never reshapes the curve, "
            "which is why AUC is threshold-independent."
        )

# --------------------------------------------------- tab: model comparison --
with tab_compare:
    best = comparison["MCC"].idxmax()

    st.markdown('<div class="section">Rank models by</div>', unsafe_allow_html=True)
    metric_choice = st.radio(
        "Rank by", METRICS, index=5, horizontal=True, label_visibility="collapsed"
    )

    left, right = st.columns([1.15, 1], gap="large")

    with left:
        rank_df = (
            comparison[metric_choice].reset_index()
            .rename(columns={metric_choice: "Value"})
            .sort_values("Value", ascending=False)
        )
        top = rank_df.iloc[0]["ML Model Name"]
        # Bars keep their zero baseline; only the headroom is trimmed, so the
        # five models are actually distinguishable instead of all reading ~half.
        hi, lo = float(rank_df["Value"].max()), float(rank_df["Value"].min())
        upper = min(1.0, hi * 1.25) if hi > 0 else 1.0
        domain = [min(0.0, lo * 1.15), upper]
        # Emphasis, not a value-ramp: the leader carries slot-1 blue, rest recede.
        bars = (
            alt.Chart(rank_df)
            .mark_bar(cornerRadiusEnd=4, height=22)
            .encode(
                x=alt.X("Value:Q", title=metric_choice, scale=alt.Scale(domain=domain)),
                y=alt.Y("ML Model Name:N", sort="-x", title=None,
                        axis=alt.Axis(labelLimit=200, labelFontSize=12)),
                color=alt.condition(alt.datum["ML Model Name"] == top,
                                    alt.value(BLUE), alt.value(RECEDE)),
                tooltip=[alt.Tooltip("ML Model Name:N", title="Model"),
                         alt.Tooltip("Value:Q", title=metric_choice, format=".4f")],
            )
        )
        labels = bars.mark_text(align="left", dx=8, fontSize=11, font="system-ui").encode(
            text=alt.Text("Value:Q", format=".4f"), color=alt.value(INK_SECONDARY)
        )
        st.altair_chart(
            style(bars + labels, height=260, legend=False)
            .properties(title=f"{metric_choice} at threshold {threshold:.2f}"),
            width="stretch",
        )

    with right:
        curves = []
        for name, p in probs.items():
            if len(np.unique(y)) < 2:
                continue
            f, t, _ = roc_curve(y, p)
            s = max(1, len(f) // 250)
            curves.append(pd.DataFrame({"FPR": f[::s], "TPR": t[::s], "Model": name}))
        if curves:
            curve_df = pd.concat(curves, ignore_index=True)
            auc_label = {n: f"{n} ({comparison.loc[n, 'AUC']:.3f})" for n in comparison.index}
            curve_df["Model (AUC)"] = curve_df["Model"].map(auc_label)
            overlay = (
                alt.Chart(curve_df)
                .mark_line(strokeWidth=2, strokeCap="round")
                .encode(
                    x=alt.X("FPR:Q", title="False positive rate",
                            scale=alt.Scale(domain=[0, 1]),
                            axis=alt.Axis(format=".0%", tickCount=5)),
                    y=alt.Y("TPR:Q", title="True positive rate",
                            scale=alt.Scale(domain=[0, 1]),
                            axis=alt.Axis(format=".0%", tickCount=5)),
                    color=alt.Color("Model (AUC):N",
                                    scale=alt.Scale(
                                        domain=[auc_label[n] for n in comparison.index],
                                        range=SERIES),
                                    legend=alt.Legend(title=None)),
                    tooltip=[alt.Tooltip("Model:N"), alt.Tooltip("FPR:Q", format=".2%"),
                             alt.Tooltip("TPR:Q", format=".2%")],
                )
            )
            # Streamlit fits plot + title + legend into `height`, so a five-row
            # legend needs real headroom or the plot collapses.
            st.altair_chart(
                style(overlay, height=380).properties(title="ROC — every model"),
                width="stretch",
            )
        else:
            st.info("ROC needs both classes present in the uploaded file.")

    st.success(f"Highest MCC on this data: **{best}** ({comparison.loc[best, 'MCC']:.4f})")

    st.markdown('<div class="section">Full metric table</div>', unsafe_allow_html=True)
    st.dataframe(
        comparison.style.format("{:.4f}").highlight_max(axis=0, color="#16345c"),
        width="stretch",
    )

    trained = load_training_metrics()
    if trained:
        with st.expander("Reference metrics recorded during training"):
            ref = pd.DataFrame(trained).T.set_index("label").astype(float)
            ref.index.name = "ML Model Name"
            st.dataframe(ref.round(4), width="stretch")

# ----------------------------------------------------------- tab: row-level --
with tab_rows:
    out = raw.copy()
    out["Churn_Probability"] = prob.round(4)
    out["Predicted"] = np.where(pred == 1, "Yes", "No")
    out["Correct"] = np.where(
        out["Predicted"] == raw["Churn"].astype(str).str.strip(), "✓", "✗"
    )

    n_wrong = int((out["Correct"] == "✗").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(tile("Rows scored", 0, f"from {source_label.lower()}", fill=1.0,
                     text=f"{len(out):,}"), unsafe_allow_html=True)
    c2.markdown(tile("Correct", 1 - n_wrong / len(out), f"{len(out) - n_wrong:,} rows"),
                unsafe_allow_html=True)
    c3.markdown(tile("Misclassified", n_wrong / len(out), f"{n_wrong:,} rows"),
                unsafe_allow_html=True)
    c4.markdown(tile("Mean churn probability", float(prob.mean()),
                     "across every scored row"), unsafe_allow_html=True)

    st.write("")
    only_wrong = st.checkbox("Misclassified rows only", value=False)
    view = out[out["Correct"] == "✗"] if only_wrong else out

    # Lead with the three columns this tab exists for; the 21 raw feature
    # columns follow, instead of burying the verdict off the right edge.
    lead_cols = ["Churn_Probability", "Predicted", "Churn", "Correct"]
    ordered = [c for c in lead_cols if c in view.columns] + [
        c for c in view.columns if c not in lead_cols
    ]
    st.dataframe(
        view.head(300)[ordered],
        width="stretch",
        height=420,
        column_config={
            "Churn_Probability": st.column_config.ProgressColumn(
                "Churn probability", min_value=0.0, max_value=1.0, format="%.4f"),
            "Predicted": st.column_config.TextColumn("Predicted", width="small"),
            "Churn": st.column_config.TextColumn("Actual", width="small"),
            "Correct": st.column_config.TextColumn("✓", width="small"),
        },
    )
    st.caption(f"Showing {min(len(view), 300):,} of {len(view):,} rows.")

    st.download_button(
        "Download predictions as CSV",
        out.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{selected.split()[0].lower()}.csv",
        mime="text/csv",
        type="primary",
    )

st.divider()
st.caption(
    "BITS Pilani WILP · M.Tech (AIML) · Machine Learning Assignment 2 · "
    "Dataset: IBM Telco Customer Churn (7,043 customers)."
)
