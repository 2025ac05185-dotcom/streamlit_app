"""
Telco Churn Model Explorer - Streamlit front end.

Upload a raw test CSV, pick one of five trained classifiers, and inspect its
behaviour on that data: the six assignment metrics, a confusion matrix, a
classification report, an ROC curve, a side-by-side comparison of all models,
and row-level predictions you can download.

Launch locally with:  streamlit run app.py
"""

from __future__ import annotations

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

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest.joblib",
}

MODEL_NOTES = {
    "Logistic Regression": "Linear log-odds model, class-weighted. Strong ranking, deliberately loose threshold.",
    "Decision Tree": "Depth capped at 6 with a 25-row leaf minimum, so the splits stay readable.",
    "kNN": "25 distance-weighted neighbours in the scaled 45-column encoded space.",
    "Naive Bayes": "Gaussian likelihoods, features assumed independent - a strong assumption here.",
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

# ------------------------------------------------------------------ palette --
# Validated categorical ramp: worst adjacent CVD dE 8.4, normal-vision dE 19.3,
# every slot >= 3:1 against the #181817 card surface.
SURFACE = "#181817"
PLANE = "#0d0d0d"
INK = "#f2f1ec"
INK_SECONDARY = "#c3c2b7"
INK_MUTED = "#898781"
GRID = "#2c2c2a"
AXIS = "#383835"

SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]
BLUE = SERIES[0]
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
WARNING = "#fab219"
BLUE_RAMP = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"]

st.set_page_config(
    page_title="Telco Churn Model Explorer",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------------------------------- chrome --
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
          .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px; }}
          header[data-testid="stHeader"] {{ background: transparent; }}
          #MainMenu, footer {{ visibility: hidden; }}

          /* ---- hero ---- */
          .hero {{
            border: 1px solid {GRID};
            border-radius: 16px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1.5rem;
            background:
              radial-gradient(120% 180% at 0% 0%, rgba(57,135,229,0.16) 0%, rgba(57,135,229,0) 55%),
              {SURFACE};
          }}
          .hero h1 {{
            margin: 0 0 .35rem 0; font-size: 1.9rem; font-weight: 660;
            letter-spacing: -0.022em; color: {INK};
          }}
          .hero p {{ margin: 0; color: {INK_SECONDARY}; font-size: .95rem; max-width: 68ch; }}
          .chips {{ display: flex; flex-wrap: wrap; gap: .45rem; margin-top: 1rem; }}
          .chip {{
            display: inline-flex; align-items: center; gap: .4rem;
            border: 1px solid {GRID}; border-radius: 999px;
            padding: .28rem .7rem; font-size: .78rem; color: {INK_SECONDARY};
            background: rgba(255,255,255,0.02); white-space: nowrap;
          }}
          .chip b {{ color: {INK}; font-weight: 600; }}
          .dot {{ width: 6px; height: 6px; border-radius: 50%; background: {BLUE}; flex: none; }}
          .dot.good {{ background: {GOOD}; }}

          /* ---- stat tiles ---- */
          .tile {{
            border: 1px solid {GRID}; border-radius: 14px; padding: .85rem .95rem 0.95rem;
            background: {SURFACE}; height: 100%;
          }}
          .tile .label {{
            font-size: .73rem; text-transform: uppercase; letter-spacing: .07em;
            color: {INK_MUTED}; font-weight: 600; margin-bottom: .3rem;
          }}
          .tile .value {{
            font-size: 1.85rem; font-weight: 640; color: {INK};
            line-height: 1.1; letter-spacing: -0.02em;
          }}
          .tile .meter {{
            margin-top: .65rem; height: 4px; border-radius: 999px;
            background: rgba(255,255,255,0.07); overflow: hidden;
          }}
          .tile .meter span {{ display: block; height: 100%; border-radius: 999px; background: {BLUE}; }}
          .tile .foot {{ margin-top: .45rem; font-size: .72rem; color: {INK_MUTED}; }}

          /* ---- section headings ---- */
          .section {{
            font-size: .76rem; text-transform: uppercase; letter-spacing: .08em;
            color: {INK_MUTED}; font-weight: 650; margin: .2rem 0 .7rem;
          }}

          /* ---- tabs (Streamlit 1.61 renders these with role=, not baseweb attrs) ---- */
          .stTabs [role="tablist"] {{ gap: .15rem; border-bottom: 1px solid {GRID}; }}
          .stTabs [role="tab"] {{
            padding: .55rem .95rem; color: {INK_MUTED};
            font-size: .9rem; font-weight: 550; border-radius: 8px 8px 0 0;
            transition: color .12s ease, background .12s ease;
          }}
          .stTabs [role="tab"]:hover {{ color: {INK_SECONDARY}; background: rgba(255,255,255,0.03); }}
          .stTabs [role="tab"][aria-selected="true"] {{ color: {INK}; font-weight: 600; }}

          /* ---- sidebar ---- */
          section[data-testid="stSidebar"] {{ border-right: 1px solid {GRID}; }}
          section[data-testid="stSidebar"] .block-container {{ padding-top: 1.6rem; }}

          /* ---- misc ---- */
          div[data-testid="stVegaLiteChart"] {{
            border: 1px solid {GRID}; border-radius: 14px;
            padding: .8rem; background: {SURFACE};
          }}
          hr {{ border-color: {GRID}; }}
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
) -> str:
    """A stat tile: label, hero figure, a thin 0-1 meter, and a caption."""
    raw_fill = value if fill is None else fill
    # An undefined metric (single-class upload) must not emit width:nan%.
    pct = 0.0 if np.isnan(raw_fill) else max(0.0, min(1.0, raw_fill)) * 100
    if text is not None:
        shown = text
    elif np.isnan(value):
        shown = "—"
    else:
        shown = f"{value:.4f}"
    return (
        f'<div class="tile"><div class="label">{label}</div>'
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
        .configure_title(color=INK, fontSize=13, fontWeight=600, anchor="start", font="system-ui")
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
        labelLimit=260,
    )


# --------------------------------------------------------------------- data --
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


def evaluate(y_true, y_pred, y_prob) -> dict:
    # AUC is undefined when an uploaded CSV holds a single class; report it as
    # NaN rather than letting sklearn warn and hand back a bare nan.
    single_class = len(np.unique(y_true)) < 2
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": float("nan") if single_class else roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


inject_css()

# ----------------------------------------------------------------- sidebar --
st.sidebar.markdown('<div class="section">Data</div>', unsafe_allow_html=True)

models = load_models()
if not models:
    st.error(
        "No trained models found in `model/`. Run `python model/train_models.py` "
        "from the repository root first."
    )
    st.stop()

uploaded = st.sidebar.file_uploader(
    "Test data (CSV)",
    type="csv",
    label_visibility="collapsed",
    help="Raw Telco-format CSV including the Churn column. Leave empty to use "
    "the bundled held-out split.",
)

if uploaded is not None:
    raw = pd.read_csv(uploaded, dtype=str, keep_default_na=False)
    source_label = f"Uploaded — {uploaded.name}"
else:
    raw = load_bundled_test()
    source_label = "Bundled held-out split"
st.sidebar.caption(
    "Raw Telco-format CSV including the `Churn` column. Leave empty to score "
    "the bundled held-out split."
)

if raw is None:
    st.warning("Upload a CSV to begin.")
    st.stop()

st.sidebar.markdown('<div class="section">Model</div>', unsafe_allow_html=True)
selected = st.sidebar.selectbox(
    "Model", list(models.keys()), index=4, label_visibility="collapsed"
)
st.sidebar.caption(MODEL_NOTES[selected])

st.sidebar.markdown('<div class="section">Decision threshold</div>', unsafe_allow_html=True)
threshold = st.sidebar.slider(
    "Decision threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.01,
    label_visibility="collapsed",
    help="Probability above which a customer is flagged as churning. Lower it "
    "to catch more churners at the cost of false alarms.",
)
st.sidebar.caption(
    "Lower the threshold to catch more churners at the cost of false alarms."
)

# -------------------------------------------------------------------- main --
try:
    X, y = split_features_and_target(raw)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

pipe = models[selected]
try:
    prob = pipe.predict_proba(X)[:, 1]
except Exception as exc:  # noqa: BLE001
    st.error(f"Scoring failed — does the CSV match the Telco schema? ({exc})")
    st.stop()

pred = (prob >= threshold).astype(int)
metrics = evaluate(y, pred, prob)
churn_rate = float(np.mean(y))

st.markdown(
    f"""
    <div class="hero">
      <h1>Telco Churn Model Explorer</h1>
      <p>Five classifiers over one shared preprocessing pipeline, scored live on
         whatever CSV you feed them. Move the threshold to trade recall against
         false alarms.</p>
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

cols = st.columns(6, gap="small")
for col, (name, value) in zip(cols, metrics.items()):
    fill = (value + 1) / 2 if name == "MCC" else value
    col.markdown(tile(name, value, METRIC_HELP[name], fill), unsafe_allow_html=True)

st.write("")

tab_matrix, tab_curve, tab_compare, tab_rows = st.tabs(
    ["Confusion matrix", "ROC curve", "All models", "Row predictions"]
)

# ------------------------------------------------------- tab: confusion cm --
with tab_matrix:
    # labels=[0, 1] forces a 2x2 result even when an uploaded CSV holds a single
    # class and no positives are predicted — otherwise sklearn returns 1x1 and
    # the unpack below raises.
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    left, right = st.columns([1, 1.15], gap="large")

    with left:
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
            color=alt.Color(
                "Rate:Q",
                scale=alt.Scale(range=BLUE_RAMP[::-1], domain=[0, 1]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Actual:N", title="Actually"),
                alt.Tooltip("Predicted:N", title="Predicted"),
                alt.Tooltip("Count:Q", title="Customers", format=","),
                alt.Tooltip("Rate:Q", title="Share of actual class", format=".1%"),
            ],
        )
        counts = base.mark_text(
            fontSize=22, fontWeight=600, font="system-ui", dy=-9
        ).encode(
            text=alt.Text("Count:Q", format=","),
            # Light cells (low rate) take dark ink; dark cells take white.
            color=alt.condition(
                alt.datum.Rate > 0.5, alt.value("#ffffff"), alt.value("#0b0b0b")
            ),
        )
        rates = base.mark_text(fontSize=11, font="system-ui", dy=13).encode(
            text=alt.Text("Rate:Q", format=".1%"),
            color=alt.condition(
                alt.datum.Rate > 0.5,
                alt.value("rgba(255,255,255,0.78)"),
                alt.value("rgba(11,11,11,0.72)"),
            ),
        )
        st.altair_chart(
            style(cells + counts + rates, height=250, legend=False), width="stretch"
        )
        st.caption(
            f"**{tp}** churners caught · **{fn}** missed · "
            f"**{fp}** loyal customers flagged unnecessarily. Cells are shaded by "
            "share of each **actual** class, so a dark diagonal is a good model."
        )

    with right:
        st.markdown('<div class="section">Classification report</div>', unsafe_allow_html=True)
        report = classification_report(
            y, pred, labels=[0, 1], target_names=["Stayed", "Churned"],
            output_dict=True, zero_division=0,
        )
        st.dataframe(
            pd.DataFrame(report).transpose().round(3),
            width="stretch",
        )
        st.markdown('<div class="section">Threshold economics</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.markdown(
            tile("Caught", tp / (tp + fn) if tp + fn else 0.0,
                 f"{tp:,} of {tp + fn:,} churners"),
            unsafe_allow_html=True,
        )
        c2.markdown(
            tile("False alarms", fp / (fp + tn) if fp + tn else 0.0,
                 f"{fp:,} of {fp + tn:,} loyal"),
            unsafe_allow_html=True,
        )
        c3.markdown(
            tile("Flag rate", (tp + fp) / cm.sum(), f"{tp + fp:,} customers contacted"),
            unsafe_allow_html=True,
        )

# ------------------------------------------------------------- tab: ROC ----
with tab_curve:
    fpr, tpr, thr = roc_curve(y, prob)
    step = max(1, len(fpr) // 400)
    roc_df = pd.DataFrame(
        {"FPR": fpr[::step], "TPR": tpr[::step], "Series": selected}
    )
    diag = pd.DataFrame({"FPR": [0.0, 1.0], "TPR": [0.0, 1.0], "Series": "Random"})
    both = pd.concat([roc_df, diag], ignore_index=True)

    hover = alt.selection_point(
        fields=["FPR"], nearest=True, on="pointerover", empty=False, clear="pointerout"
    )
    colour = alt.Color(
        "Series:N",
        scale=alt.Scale(domain=[selected, "Random"], range=[BLUE, INK_MUTED]),
        legend=alt.Legend(title=None),
    )
    axis_kw = dict(
        scale=alt.Scale(domain=[0, 1]),
        axis=alt.Axis(format=".0%", tickCount=5),
    )
    lines = (
        alt.Chart(both)
        .mark_line(strokeWidth=2, interpolate="monotone", strokeCap="round")
        .encode(
            x=alt.X("FPR:Q", title="False positive rate", **axis_kw),
            y=alt.Y("TPR:Q", title="True positive rate", **axis_kw),
            color=colour,
            strokeDash=alt.condition(
                alt.datum.Series == "Random", alt.value([4, 4]), alt.value([0])
            ),
            detail="Series:N",
        )
    )
    points = (
        alt.Chart(roc_df)
        .mark_point(size=110, filled=True, stroke=SURFACE, strokeWidth=2, color=BLUE)
        .encode(
            x=alt.X("FPR:Q", **axis_kw),
            y=alt.Y("TPR:Q", **axis_kw),
            opacity=alt.condition(hover, alt.value(1), alt.value(0)),
            tooltip=[
                alt.Tooltip("FPR:Q", title="False positive rate", format=".2%"),
                alt.Tooltip("TPR:Q", title="True positive rate", format=".2%"),
            ],
        )
        .add_params(hover)
    )
    # Both axes are 0-1, so keep the plot roughly square rather than letting it
    # stretch full-width and flatten the curve.
    plot_col, side_col = st.columns([1.6, 1], gap="large")
    with plot_col:
        st.altair_chart(style(lines + points, height=420), width="stretch")
    with side_col:
        st.markdown(
            tile("AUC", metrics["AUC"], f"{selected} · threshold-independent"),
            unsafe_allow_html=True,
        )
        st.write("")
        st.caption(
            "AUC measures ranking quality across every threshold at once — moving "
            "the slider changes every other number on this page, but never this "
            "curve. The dashed diagonal is a coin flip (AUC 0.50); the further the "
            "curve bows toward the top-left, the better the model separates "
            "churners from stayers. Hover the curve to read the exact trade-off "
            "at any point."
        )

# ------------------------------------------------------- tab: all models ---
with tab_compare:
    rows, curves = {}, []
    for i, (name, model) in enumerate(models.items()):
        p = model.predict_proba(X)[:, 1]
        rows[name] = evaluate(y, (p >= threshold).astype(int), p)
        f, t, _ = roc_curve(y, p)
        s = max(1, len(f) // 250)
        curves.append(pd.DataFrame({"FPR": f[::s], "TPR": t[::s], "Model": name}))

    comparison = pd.DataFrame(rows).T
    comparison.index.name = "ML Model Name"
    best = comparison["MCC"].idxmax()

    st.markdown('<div class="section">Ranked by</div>', unsafe_allow_html=True)
    metric_choice = st.radio(
        "Rank by",
        list(metrics.keys()),
        index=5,
        horizontal=True,
        label_visibility="collapsed",
    )

    left, right = st.columns([1.15, 1], gap="large")

    with left:
        rank_df = (
            comparison[metric_choice]
            .reset_index()
            .rename(columns={metric_choice: "Value"})
            .sort_values("Value", ascending=False)
        )
        top = rank_df.iloc[0]["ML Model Name"]
        # Bars keep their zero baseline; only the headroom is trimmed, so the
        # five models are actually distinguishable instead of all reading ~half.
        hi = float(rank_df["Value"].max())
        lo = float(rank_df["Value"].min())
        upper = min(1.0, hi * 1.25) if hi > 0 else 1.0
        domain = [min(0.0, lo * 1.15), upper]
        # Emphasis, not a value-ramp: the leader carries slot-1 blue, the rest recede.
        bars = (
            alt.Chart(rank_df)
            .mark_bar(cornerRadiusEnd=4, height=22)
            .encode(
                x=alt.X("Value:Q", title=metric_choice, scale=alt.Scale(domain=domain)),
                y=alt.Y("ML Model Name:N", sort="-x", title=None,
                        axis=alt.Axis(labelLimit=200, labelFontSize=12)),
                color=alt.condition(
                    alt.datum["ML Model Name"] == top,
                    alt.value(BLUE),
                    alt.value("#3a3a37"),
                ),
                tooltip=[
                    alt.Tooltip("ML Model Name:N", title="Model"),
                    alt.Tooltip("Value:Q", title=metric_choice, format=".4f"),
                ],
            )
        )
        value_labels = bars.mark_text(
            align="left", dx=8, fontSize=11, font="system-ui", color=INK_SECONDARY
        ).encode(text=alt.Text("Value:Q", format=".4f"), color=alt.value(INK_SECONDARY))
        st.altair_chart(
            style(bars + value_labels, height=260, legend=False).properties(
                title=f"{metric_choice} at threshold {threshold:.2f}"
            ),
            width="stretch",
        )

    with right:
        curve_df = pd.concat(curves, ignore_index=True)
        auc_label = {n: f"{n} ({comparison.loc[n, 'AUC']:.3f})" for n in comparison.index}
        curve_df["Model (AUC)"] = curve_df["Model"].map(auc_label)
        overlay = (
            alt.Chart(curve_df)
            .mark_line(strokeWidth=2, strokeCap="round")
            .encode(
                x=alt.X("FPR:Q", title="False positive rate",
                        scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%", tickCount=5)),
                y=alt.Y("TPR:Q", title="True positive rate",
                        scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%", tickCount=5)),
                color=alt.Color(
                    "Model (AUC):N",
                    scale=alt.Scale(
                        domain=[auc_label[n] for n in comparison.index], range=SERIES
                    ),
                    legend=alt.Legend(title=None),
                ),
                tooltip=[
                    alt.Tooltip("Model:N"),
                    alt.Tooltip("FPR:Q", format=".2%"),
                    alt.Tooltip("TPR:Q", format=".2%"),
                ],
            )
        )
        # Streamlit fits the whole chart (plot + title + legend) into `height`,
        # so a five-row legend needs real headroom or the plot collapses.
        st.altair_chart(
            style(overlay, height=380).properties(title="ROC — every model"),
            width="stretch",
        )

    st.success(
        f"Highest MCC on this data: **{best}** ({comparison.loc[best, 'MCC']:.4f})"
    )

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

# -------------------------------------------------------- tab: row output --
with tab_rows:
    out = raw.copy()
    out["Churn_Probability"] = prob.round(4)
    out["Predicted"] = np.where(pred == 1, "Yes", "No")
    out["Correct"] = np.where(
        out["Predicted"] == raw["Churn"].astype(str).str.strip(), "✓", "✗"
    )

    n_wrong = int((out["Correct"] == "✗").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(tile("Rows scored", 0, f"from {source_label.lower()}",
                     fill=1.0, text=f"{len(out):,}"),
                unsafe_allow_html=True)
    c2.markdown(tile("Correct", 1 - n_wrong / len(out), f"{len(out) - n_wrong:,} rows"),
                unsafe_allow_html=True)
    c3.markdown(tile("Misclassified", n_wrong / len(out), f"{n_wrong:,} rows"),
                unsafe_allow_html=True)
    c4.markdown(tile("Mean churn probability", float(prob.mean()),
                     "across every scored row"), unsafe_allow_html=True)

    st.write("")
    f1c, f2c = st.columns([1, 2])
    only_wrong = f1c.checkbox("Misclassified rows only", value=False)
    view = out[out["Correct"] == "✗"] if only_wrong else out

    # Lead with the three columns this tab exists for; the 21 raw feature
    # columns follow, instead of burying the verdict off the right edge.
    lead = ["Churn_Probability", "Predicted", "Churn", "Correct"]
    ordered = [c for c in lead if c in view.columns] + [
        c for c in view.columns if c not in lead
    ]
    st.dataframe(
        view.head(300)[ordered],
        width="stretch",
        height=420,
        column_config={
            "Churn_Probability": st.column_config.ProgressColumn(
                "Churn probability", min_value=0.0, max_value=1.0, format="%.4f"
            ),
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
