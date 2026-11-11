"""ReviewSense AI: a cloud-powered product-review sentiment dashboard."""

import hashlib
import html
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_processor import (
    add_prediction_columns,
    add_vader_columns,
    choose_output_names,
    detect_label_column,
    detect_rating_column,
    load_reviews_csv,
    prepare_texts,
    suggest_review_column,
)
from utils.aspect_miner import extract_ngrams
from utils.cloud_ai import (
    CloudAIError,
    analyze_reviews,
    cloudflare_credentials,
    get_usage_snapshot,
    summarize_reviews,
)
from utils.evaluator import evaluate_predictions
from utils.vader_analyzer import analyze_vader

APP_TITLE = "ReviewSense AI"
SAMPLE_FILE = Path(__file__).parent / "sample_reviews.csv"
PREANALYZED_SAMPLE_FILE = Path(__file__).parent / "sample_reviews_preanalyzed.csv"
LOGO_FILE = Path(__file__).parent / "assets" / "logo.png"
COLORS = {"Positive": "#10b981", "Negative": "#fb7185", "Neutral": "#fbbf24"}
PAGES = [
    "Dashboard Overview",
    "Upload Reviews",
    "AI Sentiment Analysis",
    "Review Analytics",
    "Aspect Analysis",
    "Customer Complaints",
    "Review Explorer",
    "Live Single-Review Sandbox",
    "Model Evaluation",
    "Apps & Downloads",
    "About Project",
]

st.set_page_config(page_title=APP_TITLE, page_icon="✦", layout="wide")


def styles():
    st.markdown(
        """
        <style>
        .stApp { background: radial-gradient(ellipse at 82% 0%, rgba(139,92,246,.13), transparent 34%), #0b0f19; }
        [data-testid="stSidebar"] { background:#111827; border-right:1px solid #263244; }
        .block-container { max-width:1440px; padding:2rem clamp(1rem,3vw,2.5rem) 3rem; }
        .eyebrow { color:#c4b5fd; text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:700; margin-bottom:.45rem; }
        .hero-title { font-size:clamp(2rem,4vw,3.2rem); line-height:1.05; letter-spacing:-.045em; font-weight:750; margin:0; }
        .hero-subtitle { color:#aeb8ca; font-size:1rem; margin-top:.65rem; }
        .metric-card { background:linear-gradient(145deg,#1f2937,#151d2b); border:1px solid #374151; border-radius:16px; padding:1.05rem 1.1rem; min-height:122px; box-shadow:0 10px 24px rgba(0,0,0,.12); }
        .metric-label { color:#b7c1d1; font-size:.82rem; font-weight:600; }
        .metric-value { color:#f5f3ff; font-size:clamp(1.05rem,2.1vw,1.8rem); font-weight:730; margin-top:.28rem; overflow-wrap:anywhere; }
        .metric-foot { color:#9aa7ba; font-size:.76rem; margin-top:.12rem; }
        .section-title { font-size:1.18rem; font-weight:680; margin:.25rem 0 .1rem; }
        div[data-testid="stDataFrame"] { border:1px solid #374151; border-radius:12px; overflow:hidden; }
        div[data-testid="stAlert"] { border-radius:12px; }
        .stButton > button, .stDownloadButton > button { border-radius:10px; transition:transform .15s ease,border-color .15s ease; }
        .stButton > button:hover, .stDownloadButton > button:hover { transform:translateY(-1px); border-color:#8b5cf6; }
        @media (max-width: 720px) {
            .block-container { padding-top:1.2rem; padding-left:.85rem; padding-right:.85rem; }
            .metric-card { min-height:104px; padding:.85rem; }
            .metric-value { font-size:1.45rem; }
        }
        footer { visibility:hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def header(title, subtitle):
    st.markdown('<div class="eyebrow">Product intelligence · ReviewSense AI</div>', unsafe_allow_html=True)
    st.markdown(f'<h1 class="hero-title">{html.escape(title)}</h1>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-subtitle">{html.escape(subtitle)}</div>', unsafe_allow_html=True)
    st.write("")


def metric(label, value, note, color="#a78bfa"):
    st.markdown(
        f"""<div class="metric-card" style="border-top:2px solid {color}">
        <div class="metric-label">{html.escape(str(label))}</div>
        <div class="metric-value">{html.escape(str(value))}</div>
        <div class="metric-foot">{html.escape(str(note))}</div></div>""",
        unsafe_allow_html=True,
    )


def usage_caption():
    usage = get_usage_snapshot()
    return (
        f"App safety usage: {usage['reviews_today']}/{usage['review_limit']} reviews today · "
        f"{usage['requests_last_minute']}/{usage['request_limit_per_minute']} requests/min."
    )


def safe_ai_error(error):
    message = str(error)
    for technical_name in ("Cloudflare Workers AI", "Cloudflare AI", "Cloudflare"):
        message = message.replace(technical_name, "the remote AI service")
    for setting_name in ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"):
        message = message.replace(setting_name, "server-side credentials")
    return message


def figure_style(fig, height=340):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=12, r=12, t=30, b=12),
        font=dict(family="Inter, sans-serif", color="#cbd2e1"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def sentiment_counts(data, sentiment_column):
    return data[sentiment_column].value_counts().reindex(
        ["Positive", "Negative", "Neutral"], fill_value=0
    )


def render_upload_page(dataframe, filename, review_column, analysis_ready):
    header("Upload reviews", "Load a CSV, confirm its text field, and explicitly start any cloud inference.")
    if filename in {SAMPLE_FILE.name, PREANALYZED_SAMPLE_FILE.name}:
        st.info(f"Active dataset: {filename} · {len(dataframe):,} reviews.")
    else:
        st.success(f"Upload received in app memory: {filename} · {len(dataframe):,} reviews.")
    st.write(f"Review text column: **{review_column}**")
    st.caption(
        "CSV limits: 2 MB and 100 reviews. Empty reviews are retained as Neutral; duplicate non-empty reviews are flagged."
    )
    st.warning(
        "When using AI-powered analysis, review text is sent from this app's server to an external AI service. "
        "The project has no database and does not intentionally save the uploaded CSV."
    )
    st.dataframe(dataframe.head(10), hide_index=True, width="stretch")
    if analysis_ready:
        st.success("Analysis is ready. Open AI Sentiment Analysis, Aspect Analysis, or Review Explorer.")
    else:
        st.info("To avoid using cloud quota by accident, start analysis with the button in the sidebar.")


def load_dataset(source, uploaded):
    if source == "Analyze sample CSV":
        return pd.read_csv(SAMPLE_FILE), SAMPLE_FILE.name
    if source == "Pre-analyzed demo (instant)":
        return pd.read_csv(PREANALYZED_SAMPLE_FILE), PREANALYZED_SAMPLE_FILE.name
    if uploaded is None:
        st.info("Upload a CSV from the sidebar, or choose a sample demo.")
        st.stop()
    try:
        return load_reviews_csv(uploaded), uploaded.name
    except ValueError as error:
        st.error(str(error))
        st.stop()


def prediction_signature(texts, engine_choice):
    digest = hashlib.sha256(engine_choice.encode("utf-8"))
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


def run_analysis(dataframe, review_column, requested_engine):
    texts = tuple(prepare_texts(dataframe, review_column))
    names = choose_output_names(dataframe.columns)
    result = add_vader_columns(dataframe, list(texts), names)

    if requested_engine == "Pre-analyzed demo (instant)":
        required = ["AI Sentiment", "AI Confidence", "VADER Sentiment", "Methods Agree"]
        missing = [column for column in required if column not in dataframe.columns]
        if missing:
            raise ValueError(f"Pre-analyzed demo is missing columns: {', '.join(missing)}")
        for key, column in (
            ("ai", "AI Sentiment"),
            ("confidence", "AI Confidence"),
            ("vader", "VADER Sentiment"),
            ("agreement", "Methods Agree"),
            ("vader_score", "VADER Compound"),
            ("duplicate", "Duplicate Review"),
        ):
            names[key] = column
        names["reason"] = "AI Explanation"
        names["aspects"] = "Detected Aspects"
        names["source"] = "Analysis Type"
        demo = dataframe.copy()
        demo[names["reason"]] = "Explanation not included in the pre-analyzed demo."
        demo[names["aspects"]] = "[]"
        demo[names["source"]] = "Pre-analyzed demo"
        return demo, names, "Pre-analyzed demo", None

    if requested_engine == "VADER fallback":
        names["ai"] = names["vader_fallback"]
        result[names["ai"]] = result[names["vader"]]
        result[names["confidence"]] = None
        result[names["agreement"]] = None
        result[names["reason"]] = "VADER lexicon and rules; no generated explanation."
        result[names["aspects"]] = ""
        result[names["source"]] = "VADER fallback"
        return result, names, "VADER fallback", None

    account_id, api_token, model_name = cloudflare_credentials()
    signature = prediction_signature(texts, model_name)
    prediction_cache = st.session_state.setdefault("prediction_cache", {})
    predictions = prediction_cache.get(signature)
    if predictions is None:
        progress = st.progress(0, text="Sending reviews for AI-powered analysis…")
        try:
            predictions = analyze_reviews(
                list(texts),
                account_id=account_id,
                api_token=api_token,
                model=model_name,
                progress_callback=lambda done, total: progress.progress(
                    min(done / total, 1.0), text=f"Analyzed batch {done} of {total}"
                ),
            )
        except CloudAIError as error:
            progress.empty()
            st.error(safe_ai_error(error))
            st.info("No silent fallback was applied. Select VADER fallback in the sidebar to analyze locally.")
            st.stop()
        finally:
            progress.empty()
        prediction_cache.clear()
        prediction_cache[signature] = predictions

    result = add_prediction_columns(result, predictions, names)
    result[names["source"]] = "AI-powered analysis"
    mode = "AI-powered analysis"
    return result, names, mode, None


def render_overview(data, names, engine_name, review_column):
    header("Understand every review.", "A clear view of customer sentiment, recurring feedback, and product experience.")
    counts = sentiment_counts(data, names["ai"])
    total = len(data)
    percentages = counts / total * 100 if total else counts.astype(float)
    net_score = ((counts["Positive"] - counts["Negative"]) / total * 100) if total else 0
    values = [
        ("Total reviews", f"{total:,}", "Rows analyzed", "#a78bfa"),
        ("Positive", f"{counts['Positive']:,}", f"{percentages['Positive']:.1f}% of reviews", "#a78bfa"),
        ("Negative", f"{counts['Negative']:,}", f"{percentages['Negative']:.1f}% of reviews", "#fb7185"),
        ("Neutral", f"{counts['Neutral']:,}", f"{percentages['Neutral']:.1f}% of reviews", "#94a3b8"),
        ("Net sentiment", f"{net_score:+.1f}", "(Positive − Negative) / total × 100", "#10b981"),
    ]
    rating_column = st.session_state.get("selected_rating_column")
    if rating_column in data.columns:
        ratings = pd.to_numeric(data[rating_column], errors="coerce").dropna()
        average_rating = f"{ratings.mean():.2f}" if not ratings.empty else "—"
        rating_note = "Mean of available ratings"
    else:
        average_rating, rating_note = "—", "No rating column selected"

    negative_reviews = data.loc[data[names["ai"]] == "Negative", review_column].fillna("").astype(str).tolist()
    positive_reviews = data.loc[data[names["ai"]] == "Positive", review_column].fillna("").astype(str).tolist()
    negative_phrases = extract_ngrams(negative_reviews, limit=1)
    positive_phrases = extract_ngrams(positive_reviews, limit=1)
    values.extend(
        [
            ("Average rating", average_rating, rating_note, "#fbbf24"),
            (
                "Common complaint phrase",
                negative_phrases[0][0] if negative_phrases else "Not enough data",
                f"Frequency: {negative_phrases[0][1]}" if negative_phrases else "No repeated phrase found",
                "#fb7185",
            ),
            (
                "Common praise phrase",
                positive_phrases[0][0] if positive_phrases else "Not enough data",
                f"Frequency: {positive_phrases[0][1]}" if positive_phrases else "No repeated phrase found",
                "#10b981",
            ),
        ]
    )
    for row_start in range(0, len(values), 3):
        for column, args in zip(st.columns(3, gap="small"), values[row_start : row_start + 3]):
            with column:
                metric(*args)

    st.write("")
    left, right = st.columns([1, 1.35], gap="large")
    with left:
        st.markdown('<div class="section-title">Sentiment distribution</div>', unsafe_allow_html=True)
        donut = px.pie(names=counts.index, values=counts.values, hole=.68, color=counts.index, color_discrete_map=COLORS)
        donut.update_traces(textinfo="percent", hovertemplate="%{label}: %{value} (%{percent})<extra></extra>")
        donut.add_annotation(text=f"<b>{total:,}</b><br>reviews", showarrow=False, font=dict(size=19, color="#f5f3ff"))
        st.plotly_chart(figure_style(donut), width="stretch")
    with right:
        st.markdown('<div class="section-title">Reviews by sentiment</div>', unsafe_allow_html=True)
        bar = px.bar(x=counts.index, y=counts.values, color=counts.index, color_discrete_map=COLORS,
                     text=counts.values, labels={"x": "Sentiment", "y": "Reviews"})
        bar.update_traces(textposition="outside", cliponaxis=False)
        bar.update_yaxes(rangemode="tozero", gridcolor="rgba(148,163,184,.12)")
        st.plotly_chart(figure_style(bar), width="stretch")

    if engine_name == "VADER fallback":
        st.info(
            "Current predictions come from the VADER fallback, not AI-powered analysis."
        )
    with st.expander("About model confidence and analysis"):
        st.write(
            f"Analysis uses **{engine_name}** on the selected `{review_column}` column. "
            "The AI service does not provide class probabilities, so confidence "
            "is shown as unavailable rather than invented. Empty reviews receive Neutral without a model call."
        )


def render_analytics(data, names, engine_name):
    header("Review analytics", "Compare sentiment distribution and prediction methods.")
    counts = sentiment_counts(data, names["ai"])
    total = len(data)
    percent = counts / total * 100 if total else counts.astype(float)
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown('<div class="section-title">Sentiment volume</div>', unsafe_allow_html=True)
        fig = px.bar(x=counts.index, y=counts.values, color=counts.index, color_discrete_map=COLORS, text=counts.values)
        fig.update_traces(textposition="outside")
        st.plotly_chart(figure_style(fig), width="stretch")
    with right:
        st.markdown('<div class="section-title">Sentiment percentages</div>', unsafe_allow_html=True)
        fig = px.bar(x=percent.values, y=percent.index, orientation="h", color=percent.index,
                     color_discrete_map=COLORS, text=[f"{value:.1f}%" for value in percent.values])
        fig.update_traces(textposition="outside")
        fig.update_xaxes(range=[0, 110])
        st.plotly_chart(figure_style(fig), width="stretch")

    rating_column = st.session_state.get("selected_rating_column")
    if rating_column not in data.columns:
        rating_column = None
    if rating_column is not None:
        ratings = pd.to_numeric(data[rating_column], errors="coerce").dropna()
        if not ratings.empty:
            st.markdown('<div class="section-title">Rating distribution</div>', unsafe_allow_html=True)
            fig = px.histogram(x=ratings, nbins=min(10, max(1, ratings.nunique())),
                               labels={"x": str(rating_column), "y": "Reviews"},
                               color_discrete_sequence=["#a78bfa"])
            st.plotly_chart(figure_style(fig, 280), width="stretch")
            rating_sentiment = data[[rating_column, names["ai"]]].copy()
            rating_sentiment["_numeric_rating"] = pd.to_numeric(
                rating_sentiment[rating_column], errors="coerce"
            )
            rating_sentiment = rating_sentiment.dropna(subset=["_numeric_rating"])
            if not rating_sentiment.empty:
                st.markdown('<div class="section-title">Ratings by predicted sentiment</div>', unsafe_allow_html=True)
                fig = px.box(
                    rating_sentiment,
                    x=names["ai"],
                    y="_numeric_rating",
                    color=names["ai"],
                    points="all",
                    category_orders={names["ai"]: ["Positive", "Neutral", "Negative"]},
                    color_discrete_map=COLORS,
                    labels={names["ai"]: "Sentiment", "_numeric_rating": str(rating_column)},
                )
                st.plotly_chart(figure_style(fig, 320), width="stretch")

    st.caption("AI-powered analysis does not return class probabilities; confidence is unavailable.")
    date_column = next(
        (column for column in data.columns if str(column).strip().lower() in {"date", "review_date", "timestamp", "created_at"}),
        None,
    )
    if date_column is not None:
        dated = data.copy()
        dated["_review_date"] = pd.to_datetime(dated[date_column], errors="coerce", utc=True).dt.date
        dated = dated.dropna(subset=["_review_date"])
        if dated["_review_date"].nunique() > 1:
            trend = (
                dated.groupby(["_review_date", names["ai"]], observed=True)
                .size()
                .reset_index(name="Reviews")
            )
            st.markdown('<div class="section-title">Sentiment trend over time</div>', unsafe_allow_html=True)
            fig = px.line(
                trend,
                x="_review_date",
                y="Reviews",
                color=names["ai"],
                markers=True,
                color_discrete_map=COLORS,
                labels={"_review_date": "Date", names["ai"]: "Sentiment"},
            )
            st.plotly_chart(figure_style(fig, 320), width="stretch")

    if names["aspects"] in data.columns:
        st.markdown('<div class="section-title">Aspect breakdown by sentiment</div>', unsafe_allow_html=True)
        aspect_records = []
        for sentiment, raw_aspects in zip(data[names["ai"]], data[names["aspects"]]):
            try:
                details = json.loads(raw_aspects) if isinstance(raw_aspects, str) else []
            except json.JSONDecodeError:
                details = []
            aspect_records.extend(
                {"Aspect": detail["aspect"], "Sentiment": detail["sentiment"]}
                for detail in details
                if isinstance(detail, dict)
                and detail.get("aspect")
                and detail.get("sentiment") in COLORS
            )
        aspect_counts = pd.DataFrame(aspect_records)
        if not aspect_counts.empty:
            aspect_counts = aspect_counts.groupby(["Aspect", "Sentiment"], observed=True).size().reset_index(name="Reviews")
            fig = px.bar(
                aspect_counts,
                x="Aspect",
                y="Reviews",
                color="Sentiment",
                barmode="group",
                color_discrete_map=COLORS,
            )
            fig.update_xaxes(tickangle=-20)
            st.plotly_chart(figure_style(fig, 380), width="stretch")

    product_column = st.session_state.get("selected_product_column")
    if product_column not in data.columns:
        product_column = None
    if product_column is not None and data[product_column].nunique(dropna=True) > 1:
        st.markdown('<div class="section-title">Product comparison</div>', unsafe_allow_html=True)
        product_sentiment = (
            data.dropna(subset=[product_column])
            .groupby([product_column, names["ai"]], observed=True)
            .size()
            .reset_index(name="Reviews")
        )
        fig = px.bar(
            product_sentiment,
            x=product_column,
            y="Reviews",
            color=names["ai"],
            barmode="group",
            color_discrete_map=COLORS,
            labels={product_column: "Product", names["ai"]: "Sentiment"},
        )
        fig.update_xaxes(tickangle=-20)
        st.plotly_chart(figure_style(fig, 380), width="stretch")

    if engine_name == "VADER fallback":
        st.info("AI-versus-VADER comparison is unavailable because VADER is the active prediction engine.")
    else:
        comparison = pd.crosstab(data[names["ai"]], data[names["vader"]]).reindex(
            index=["Positive", "Negative", "Neutral"],
            columns=["Positive", "Negative", "Neutral"],
            fill_value=0,
        )
        comparison_title = (
            "Pre-analyzed demo vs VADER"
            if engine_name == "Pre-analyzed demo"
            else "AI-powered analysis vs VADER"
        )
        st.markdown(f'<div class="section-title">{html.escape(comparison_title)}</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for sentiment in ["Positive", "Negative", "Neutral"]:
            fig.add_bar(name=f"VADER: {sentiment}", x=comparison.index, y=comparison[sentiment], marker_color=COLORS[sentiment])
        fig.update_layout(barmode="group", xaxis_title="AI sentiment", yaxis_title="Reviews")
        st.plotly_chart(figure_style(fig, 300), width="stretch")
        agreement = (data[names["ai"]] == data[names["vader"]]).mean() if len(data) else 0
        st.caption(f"The two prediction sources agree on {agreement:.1%} of reviews. Agreement is not an accuracy measure.")


def render_aspect_explorer(data, review_column, names):
    header(
        "Deep aspect & topic explorer",
        "Explore aspects identified in each AI-powered sentiment analysis.",
    )
    st.caption("Aspect and polarity labels are model-generated predictions, not verified facts.")
    records = []
    for row_index, row in data.iterrows():
        try:
            aspects = json.loads(row[names["aspects"]]) if isinstance(row[names["aspects"]], str) else []
        except json.JSONDecodeError:
            aspects = []
        for aspect in aspects:
            if isinstance(aspect, dict) and aspect.get("sentiment") in COLORS and aspect.get("aspect"):
                records.append(
                    {
                        "row_index": row_index,
                        "Aspect": aspect["aspect"],
                        "Sentiment": aspect["sentiment"],
                        "Review": str(row[review_column]),
                    }
                )
    if not records:
        st.info("Aspect results are not available. Select Analyze sample CSV or upload reviews and run AI-powered analysis.")
        return
    details = pd.DataFrame(records)
    selected_aspect = st.selectbox("Filter to aspect", ["All aspects", *sorted(details["Aspect"].unique())])
    selected = details if selected_aspect == "All aspects" else details[details["Aspect"] == selected_aspect]
    counts = selected.groupby(["Aspect", "Sentiment"], observed=True).size().reset_index(name="Reviews")
    chart = px.bar(counts, x="Aspect", y="Reviews", color="Sentiment", barmode="stack", color_discrete_map=COLORS)
    st.plotly_chart(figure_style(chart, 350), width="stretch")
    negative_reviews = selected.loc[selected["Sentiment"] == "Negative", "Review"].tolist()
    positive_reviews = selected.loc[selected["Sentiment"] == "Positive", "Review"].tolist()
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">Frequent phrases in negative reviews</div>', unsafe_allow_html=True)
        phrases = extract_ngrams(negative_reviews)
        if phrases:
            st.dataframe(pd.DataFrame(phrases, columns=["Phrase", "Frequency"]), hide_index=True, width="stretch")
        else:
            st.info("No recurring multi-word phrases found in this subset.")
    with right:
        st.markdown('<div class="section-title">Frequent phrases in positive reviews</div>', unsafe_allow_html=True)
        phrases = extract_ngrams(positive_reviews)
        if phrases:
            st.dataframe(pd.DataFrame(phrases, columns=["Phrase", "Frequency"]), hide_index=True, width="stretch")
        else:
            st.info("No recurring multi-word phrases found in this subset.")
    st.caption("N-grams are frequency counts, not AI-generated explanations. Product aspects are model predictions.")
    st.dataframe(
        selected[["Review", "Aspect", "Sentiment"]],
        width="stretch",
        hide_index=True,
    )


def render_single_review_sandbox(engine_choice):
    header(
        "Live single-review sandbox",
        "Test a custom review with AI-powered analysis and VADER side by side.",
    )
    st.caption("The review is sent to an external AI service when AI-powered analysis is selected.")
    review = st.text_area(
        "Review text",
        height=150,
        placeholder="For example: The screen looks great, but battery life is disappointing.",
    )
    cloud_consent = (
        engine_choice == "VADER fallback"
        or st.checkbox(
            "I understand this review text will be sent to an external AI service",
            value=False,
            key="sandbox_cloud_consent",
        )
    )
    if st.button("Analyze this review", type="primary", disabled=not cloud_consent):
        st.session_state.pop("sandbox_result", None)
        if not review.strip():
            st.warning("Enter some review text before analyzing.")
        elif len(review) > 4_000:
            st.warning("Please keep a single review under 4,000 characters.")
        else:
            vader_result = analyze_vader([review])[0]
            if engine_choice == "VADER fallback":
                st.session_state["sandbox_result"] = {
                    "ai": vader_result["sentiment"],
                    "reason": "VADER lexicon and rules; no generated explanation.",
                    "aspects": [],
                    "vader": vader_result,
                    "engine": "VADER fallback",
                }
            else:
                try:
                    account_id, api_token, model_name = cloudflare_credentials()
                    with st.spinner("Analyzing the review…"):
                        prediction = analyze_reviews(
                            [review.strip()],
                            account_id=account_id,
                            api_token=api_token,
                            model=model_name,
                        )[0]
                    st.session_state["sandbox_result"] = {
                        "ai": prediction["sentiment"],
                        "reason": prediction["reason"],
                        "aspects": prediction["aspects"],
                        "vader": vader_result,
                        "engine": "AI-powered analysis",
                    }
                except CloudAIError as error:
                    st.error(safe_ai_error(error))
                    st.info("No silent fallback was applied. Select VADER fallback in the sidebar to analyze locally.")
    result = st.session_state.get("sandbox_result")
    if result:
        st.write(f"Active prediction engine: **{result['engine']}**")
        first, second = st.columns(2)
        with first:
            st.metric("AI sentiment" if result["engine"] != "VADER fallback" else "VADER fallback sentiment", result["ai"])
            if result["engine"] == "AI-powered analysis":
                st.caption("Confidence unavailable: no class probabilities are returned.")
                st.write(f"**Explanation:** {result['reason']}")
                if result["aspects"]:
                    st.dataframe(pd.DataFrame(result["aspects"]), hide_index=True, width="stretch")
        with second:
            st.metric("VADER sentiment", result["vader"]["sentiment"])
            st.metric("VADER compound", f"{result['vader']['compound']:+.3f}")
            st.metric("Methods agree", "Yes" if result["ai"] == result["vader"]["sentiment"] else "No")
        st.caption("Agreement is not a measure of correctness.")


def render_ai_analysis(data, names, engine_name):
    if engine_name == "VADER fallback":
        header("Fallback sentiment analysis", "Inspect transparent VADER predictions.")
        st.warning("AI-powered analysis is not active. These labels are produced only by VADER.")
    elif engine_name == "Pre-analyzed demo":
        header("Pre-analyzed sentiment demo", "Inspect the bundled example results without making an AI service request.")
        st.info("These are bundled, precomputed sample predictions; they are not live AI analysis.")
    else:
        header("AI sentiment analysis", "Inspect AI-powered predictions, explanations, and identified aspects.")
        st.info(
            "The AI service returns labels and explanations, not class probabilities. "
            "Confidence is therefore unavailable and is never fabricated."
        )
    columns = [
        names["ai"],
        names["reason"],
        names["aspects"],
        names["source"],
        names["vader"],
        names["agreement"],
    ]
    display_names = {
        names["ai"]: "AI sentiment",
        names["reason"]: "Short explanation",
        names["aspects"]: "Detected aspects",
        names["source"]: "Analysis type",
        names["vader"]: "VADER sentiment",
        names["agreement"]: "Methods agree",
    }
    table = data[columns].rename(columns=display_names)
    st.dataframe(table.head(250), width="stretch", hide_index=True)
    if len(table) > 250:
        st.caption("Showing first 250 predictions here; all rows are available in Review Explorer and the CSV export.")


def render_explorer(data, review_column, names):
    header("Review explorer", "Search reviews, apply sentiment/product/aspect filters, and inspect results.")
    product_column = st.session_state.get("selected_product_column")
    if product_column not in data.columns:
        product_column = None
    rating_column = st.session_state.get("selected_rating_column")
    if rating_column not in data.columns:
        rating_column = None
    parsed_aspects = []
    for raw_aspects in data[names["aspects"]]:
        try:
            parsed_aspects.extend(
                item["aspect"]
                for item in json.loads(raw_aspects)
                if isinstance(item, dict) and isinstance(item.get("aspect"), str)
            )
        except (json.JSONDecodeError, TypeError):
            continue
    controls = st.columns(3)
    with controls[0]:
        sentiment = st.selectbox("AI sentiment", ["All", "Positive", "Negative", "Neutral"])
    with controls[1]:
        aspect_filter = st.selectbox("Aspect", ["All", *sorted(set(parsed_aspects))])
    with controls[2]:
        query = st.text_input("Search review text", placeholder="Type a word or phrase…")
    filtered = data
    if sentiment != "All":
        filtered = filtered[filtered[names["ai"]] == sentiment]
    if aspect_filter != "All":
        filtered = filtered[
            filtered[names["aspects"]].map(
                lambda raw: any(
                    item.get("aspect") == aspect_filter
                    for item in json.loads(raw)
                ) if isinstance(raw, str) else False
            )
        ]
    if query.strip():
        match = filtered[review_column].fillna("").astype(str).str.contains(query.strip(), case=False, regex=False)
        filtered = filtered[match]
    if product_column is not None:
        product = st.selectbox("Product", ["All", *sorted(data[product_column].dropna().astype(str).unique())])
        if product != "All":
            filtered = filtered[filtered[product_column].astype(str) == product]
    if rating_column is not None:
        rating_values = sorted(pd.to_numeric(data[rating_column], errors="coerce").dropna().unique())
        if rating_values:
            rating = st.selectbox("Rating", ["All", *rating_values])
            if rating != "All":
                filtered = filtered[pd.to_numeric(filtered[rating_column], errors="coerce") == rating]
    st.caption(f"Showing {len(filtered):,} of {len(data):,} reviews. Model results should be reviewed by a person.")
    display_columns = [
        review_column,
        names["ai"],
        names["reason"],
        names["aspects"],
        names["source"],
        names["vader"],
        names["agreement"],
        names["duplicate"],
    ]
    table = filtered[display_columns].rename(
        columns={review_column: "Review", names["ai"]: "AI sentiment",
                 names["reason"]: "Short explanation", names["aspects"]: "Detected aspects",
                 names["source"]: "Analysis type",
                 names["vader"]: "VADER sentiment", names["agreement"]: "Methods agree",
                 names["duplicate"]: "Duplicate review"}
    )
    st.dataframe(table, width="stretch", hide_index=True)


def render_insights(data, review_column, names, engine_name):
    header("Customer insights", "Frequent review words provide a lightweight, transparent view of feedback.")
    positive_reviews = data.loc[data[names["ai"]] == "Positive", review_column].fillna("").astype(str).tolist()
    negative_reviews = data.loc[data[names["ai"]] == "Negative", review_column].fillna("").astype(str).tolist()
    positive = extract_ngrams(positive_reviews, limit=10)
    negative = extract_ngrams(negative_reviews, limit=10)
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown('<div class="section-title">Frequently mentioned positive phrases</div>', unsafe_allow_html=True)
        if positive:
            phrase_data = pd.DataFrame(positive, columns=["Phrase", "Mentions"])
            st.plotly_chart(
                figure_style(
                    px.bar(
                        phrase_data,
                        x="Mentions",
                        y="Phrase",
                        orientation="h",
                        color_discrete_sequence=["#10b981"],
                        labels={"Mentions": "Occurrences"},
                    ).update_yaxes(autorange="reversed"),
                    320,
                ),
                width="stretch",
            )
        else:
            st.info("No repeated multi-word phrases found in positive reviews.")
    with b:
        st.markdown('<div class="section-title">Frequently mentioned negative phrases</div>', unsafe_allow_html=True)
        if negative:
            phrase_data = pd.DataFrame(negative, columns=["Phrase", "Mentions"])
            st.plotly_chart(
                figure_style(
                    px.bar(
                        phrase_data,
                        x="Mentions",
                        y="Phrase",
                        orientation="h",
                        color_discrete_sequence=["#fb7185"],
                        labels={"Mentions": "Occurrences"},
                    ).update_yaxes(autorange="reversed"),
                    320,
                ),
                width="stretch",
            )
        else:
            st.info("No repeated multi-word phrases found in negative reviews.")
    st.caption("Phrase frequencies are extracted from the active dataset; they are not generated explanations or verified complaint categories. No themes are invented.")
    st.subheader("Review summary")
    if engine_name != "AI-powered analysis":
        st.info("A generated summary is available after AI-powered analysis.")
        return
    evidence = (
        data[names["reason"]]
        .fillna("")
        .astype(str)
        .loc[lambda values: values.str.strip().ne("")]
        .head(20)
        .tolist()
    )
    st.caption(f"Optional summary uses up to {len(evidence)} existing AI explanations and may consume service quota.")
    account_id, api_token, model_name = cloudflare_credentials()
    summary_key = prediction_signature(evidence, f"summary:{model_name}")
    if st.button("Generate AI summary", type="primary", disabled=not evidence):
        try:
            with st.spinner("Generating a short evidence-based summary…"):
                st.session_state.setdefault("summary_cache", {})[summary_key] = summarize_reviews(
                    evidence,
                    account_id=account_id,
                    api_token=api_token,
                    model=model_name,
                )
        except CloudAIError as error:
            st.error(safe_ai_error(error))
    summary = st.session_state.get("summary_cache", {}).get(summary_key)
    if summary:
        st.info(summary)


def render_evaluation(data, label_column, names):
    header("Model evaluation", "Evaluate predictions only against human-provided sentiment labels.")
    original_columns = [column for column in data.columns if column not in names.values()]
    choices = ["-- Select ground-truth column --"] + original_columns
    default_index = choices.index(label_column) if label_column in choices else 0
    selected_column = st.selectbox("Human-labeled sentiment column", choices, index=default_index)
    if selected_column == choices[0]:
        st.warning("Model evaluation requires ground-truth sentiment labels.")
        st.caption("Choose a column containing human-assigned sentiment labels; predictions are never used as ground truth.")
        return
    if names["ai"].startswith("VADER fallback"):
        st.info("The metrics below evaluate VADER fallback predictions against your human labels, not AI-powered predictions.")
    st.write(f"Ground-truth column: **{selected_column}**")
    try:
        result = evaluate_predictions(data[selected_column], data[names["ai"]])
    except ValueError as error:
        st.warning(str(error))
        return
    metrics = st.columns(4)
    for col, key, title in zip(metrics, ["accuracy", "precision", "recall", "f1"], ["Accuracy", "Macro precision", "Macro recall", "Macro F1"]):
        with col:
            st.metric(title, f"{result[key]:.1%}")
    st.caption(f"Computed from {result['evaluated_rows']:,} rows with recognized human sentiment labels. This is dataset-specific, not a general model accuracy claim.")
    matrix = result["confusion_matrix"]
    fig = px.imshow(matrix, x=result["labels"], y=result["labels"], text_auto=True, aspect="auto",
                    color_continuous_scale=["#171b2a", "#8b5cf6"],
                    labels={"x": "Predicted", "y": "Human label", "color": "Rows"})
    st.plotly_chart(figure_style(fig, 360), width="stretch")


def render_downloads():
    header(
        "Apps & downloads",
        "ReviewSense AI is currently available as a responsive website.",
    )
    st.info(
        "No installable PWA or native application packages have been built yet. "
        "Use the website in a browser on desktop, tablet, or mobile."
    )
    st.link_button(
        "Open ReviewSense AI website",
        "https://reviewsense-ai-nves.onrender.com",
        width="stretch",
    )
    st.subheader("Availability")
    platforms = [
        ("Progressive Web App", "Build required", "The current Streamlit deployment does not include an installable manifest and service worker."),
        ("Windows", "Build required", "No signed .exe or .msi package is available."),
        ("macOS", "Build required", "No .app or .dmg package is available; signing and notarization are also required for distribution."),
        ("Linux", "Build required", "No .AppImage or .deb package is available."),
        ("Android", "Build required", "No .apk or .aab package is available."),
        ("iOS / iPadOS", "Build required", "No iOS build is available; Apple signing and provisioning are required."),
    ]
    for row_start in range(0, len(platforms), 3):
        for column, (platform, status, detail) in zip(
            st.columns(3, gap="small"), platforms[row_start : row_start + 3]
        ):
            with column:
                st.markdown(
                    f"""<div class="metric-card">
                    <div class="metric-label">{html.escape(platform)}</div>
                    <div class="metric-value">{html.escape(status)}</div>
                    <div class="metric-foot">{html.escape(detail)}</div></div>""",
                    unsafe_allow_html=True,
                )
    st.caption(
        "The website is not the same as a native app. Package links will be shown only after real builds are produced and verified."
    )


def render_about(engine_name):
    header("About ReviewSense AI", "Cloud-powered product review intelligence with transparent fallback behavior.")
    st.markdown(
        """
        **Objective:** help users inspect product feedback using server-side, free-tier cloud inference.

        **AI-powered analysis:** Review text is sent from this application to an external AI service;
        access credentials stay on the server and are not exposed in the browser.
        **Fallback:** VADER, a rule- and lexicon-based English sentiment analyzer. Fallback predictions are
        clearly identified and are never presented as AI-powered predictions.

        **Confidence:** unavailable for generative-model labels; the application does not invent a probability.
        **Evaluation:** accuracy and related metrics appear only when a human-labeled sentiment column is provided.
        **Privacy:** uploaded data is held in app memory. When AI-powered analysis is selected, review text is
        transmitted to the external inference service for processing.
        Do not upload personal, customer-identifying, confidential, or regulated information.
        """
    )
    st.write(f"Current analysis mode: **{engine_name}**")
    st.subheader("College report: summary")
    st.markdown(
        """
        - **Abstract / problem:** manual review reading is slow; this dashboard summarizes sentiment while retaining each review.
        - **Proposed system / architecture:** browser → CSV validation → remote AI service or VADER fallback → dashboard and CSV export.
        - **Implementation / results:** predictions, explanations, aspects, method comparison, charts, filters, and labeled-data evaluation
          are calculated from the active dataset. Capture screenshots from this page and the Overview for project submission.
        - **Limitations:** sarcasm, Hinglish, slang, model errors, free quotas, and cloud-host availability can affect performance.
        - **Distribution:** the browser UI works across desktop/mobile OSes; separate native installers are not included.
        - **Future scope / conclusion:** evaluate on a labeled domain dataset and add durable cross-replica rate limiting.
          Predictions should be reviewed by a person before business decisions.
        """
    )
    st.subheader("Viva questions")
    with st.expander("What is the difference between VADER and a generative language model?"):
        st.write("VADER uses a sentiment lexicon and rules. AI-powered analysis interprets review text in context, but can still make errors.")
    with st.expander("What does confidence mean?"):
        st.write("This generative model returns no class probabilities, so the app displays confidence as unavailable instead of inventing a number.")
    with st.expander("How is model accuracy measured?"):
        st.write("Compare predictions with independent human-labeled ground truth and compute metrics such as accuracy, precision, recall, F1, and a confusion matrix.")
    for question, answer in [
        ("Where does inference happen?", "Review text is sent from the hosted application server to the configured remote inference service. Credentials are stored in server-side environment settings."),
        ("How are aspects produced?", "The generative model returns aspect names and sentiment labels in a required JSON shape; the app validates every label and output."),
        ("How do you handle class imbalance?", "Show each class count and use macro precision, recall, and F1 on human-labeled validation data."),
        ("What is a confusion matrix?", "A table that compares each human-labeled class with the corresponding model predictions."),
        ("What is a key limitation?", "Sarcasm, Hinglish, slang, domain-specific meanings, quota limits, and generative-model variability can lead to incorrect predictions."),
        ("What is the VADER compound threshold?", "Scores >= 0.05 are Positive, <= -0.05 are Negative, and values between are Neutral."),
    ]:
        with st.expander(question):
            st.write(answer)


def main():
    styles()
    with st.sidebar:
        if LOGO_FILE.exists():
            st.image(str(LOGO_FILE), width=72)
        st.markdown("## ✦ ReviewSense AI")
        st.caption("Product review intelligence")
        st.divider()
        page = st.radio("Workspace", PAGES, label_visibility="collapsed")
        st.divider()
        source = st.radio(
            "Dataset",
            ["Pre-analyzed demo (instant)", "Analyze sample CSV", "Upload a CSV"],
            help="The pre-analyzed demo is instant. Other sources call the selected analysis engine.",
        )
        uploaded = st.file_uploader("Choose reviews CSV", type=["csv"]) if source == "Upload a CSV" else None
        engine = st.radio(
            "Analysis engine",
            ["Cloudflare Workers AI", "VADER fallback"],
            format_func=lambda value: "AI-powered analysis" if value == "Cloudflare Workers AI" else value,
            help="AI-powered analysis runs remotely. VADER is a clearly labeled local-in-process alternative.",
        )
        if source == "Upload a CSV":
            st.warning(
                "Privacy notice: this application receives the uploaded file. If AI-powered analysis is selected, "
                "review text is also sent to an external inference service. Do not upload personal, identifying, confidential, or regulated data."
            )
            cloud_consent = st.checkbox(
                "I understand this upload is processed by the selected analysis service",
                value=False,
            )
        else:
            cloud_consent = True
        usage_status = st.empty()
        usage_status.caption(usage_caption())
        st.download_button(
            "Download sample CSV",
            data=SAMPLE_FILE.read_bytes(),
            file_name=SAMPLE_FILE.name,
            mime="text/csv",
            width="stretch",
        )

    dataframe, filename = load_dataset(source, uploaded)
    if source == "Upload a CSV" and uploaded is not None and not cloud_consent:
        st.info("Confirm the data-processing notice in the sidebar before analyzing this upload.")
        st.stop()
    if dataframe.empty or len(dataframe.columns) == 0:
        st.error("The dataset has no rows or columns.")
        st.stop()
    suggestion = suggest_review_column(dataframe)
    with st.sidebar:
        review_column = st.selectbox(
            "Review text column", list(dataframe.columns),
            index=list(dataframe.columns).index(suggestion),
        )
        column_options = [None, *dataframe.columns.tolist()]
        product_default = next(
            (
                column
                for column in dataframe.columns
                if str(column).strip().lower() in {"product", "product_name", "product name"}
            ),
            None,
        )
        product_column = st.selectbox(
            "Product name column",
            column_options,
            index=column_options.index(product_default),
            format_func=lambda column: "Not available" if column is None else str(column),
            key="selected_product_column",
        )
        rating_default = detect_rating_column(dataframe)
        rating_column = st.selectbox(
            "Rating column",
            column_options,
            index=column_options.index(rating_default),
            format_func=lambda column: "Not available" if column is None else str(column),
            key="selected_rating_column",
        )
    label_column = detect_label_column(dataframe)
    requested_engine = "Pre-analyzed demo (instant)" if source == "Pre-analyzed demo (instant)" else engine
    requested_signature = None
    if requested_engine == "Cloudflare Workers AI":
        _, _, model_name = cloudflare_credentials()
        requested_signature = prediction_signature(
            prepare_texts(dataframe, review_column),
            model_name,
        )
        analysis_ready = st.session_state.get("cloud_analysis_signature") == requested_signature
        if not analysis_ready:
            with st.sidebar:
                start_analysis = st.button(
                    "Analyze with AI",
                    type="primary",
                    width="stretch",
                )
            if not start_analysis:
                if page == "Upload Reviews":
                    render_upload_page(dataframe, filename, review_column, analysis_ready=False)
                else:
                    st.info("Choose **Analyze with AI** in the sidebar to analyze this dataset.")
                st.stop()
    else:
        analysis_ready = True
    try:
        analyzed, names, engine_name, _ = run_analysis(dataframe, review_column, requested_engine)
    except ValueError as error:
        st.error(str(error))
        st.stop()
    if engine_name == "AI-powered analysis" and requested_signature:
        st.session_state["cloud_analysis_signature"] = requested_signature
    analyzed.attrs["engine_name"] = engine_name

    if page == "Dashboard Overview":
        render_overview(analyzed, names, engine_name, review_column)
    elif page == "Upload Reviews":
        render_upload_page(dataframe, filename, review_column, analysis_ready=True)
    elif page == "AI Sentiment Analysis":
        render_ai_analysis(analyzed, names, engine_name)
    elif page == "Review Analytics":
        render_analytics(analyzed, names, engine_name)
    elif page == "Aspect Analysis":
        render_aspect_explorer(analyzed, review_column, names)
    elif page == "Customer Complaints":
        render_insights(analyzed, review_column, names, engine_name)
    elif page == "Live Single-Review Sandbox":
        render_single_review_sandbox(engine)
    elif page == "Review Explorer":
        render_explorer(analyzed, review_column, names)
    elif page == "Model Evaluation":
        render_evaluation(analyzed, label_column, names)
    elif page == "Apps & Downloads":
        render_downloads()
    else:
        render_about(engine_name)

    with st.sidebar:
        usage_status.caption(usage_caption())
        st.divider()
        st.caption(f"Active data: {filename} · {len(dataframe):,} reviews")
        st.download_button(
            "Download analyzed CSV",
            data=analyzed.to_csv(index=False).encode("utf-8"),
            file_name="reviewsense_analyzed.csv",
            mime="text/csv",
            width="stretch",
        )
    st.caption("ReviewSense AI · Quota-limited cloud inference · English product reviews")


if __name__ == "__main__":
    main()
