"""AEGEAN Customer Experience Intelligence | Streamlit.
Run: streamlit run app.py
Uses existing HF/Gemini ABSA CSVs; never reclassifies reviews.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import cohen_kappa_score

# Application configuration
st.set_page_config(page_title="AEGEAN | Customer Experience Intelligence", page_icon="✈️", layout="wide")
# Resolve paths from the script, not the shell's working directory.
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data" / "Output"
HF_DEFAULT = Path("Data/Output/Hugging Face/df_final_absa.csv")
GEM_DEFAULT = Path("Data/Output/LLM/df_final_absa_gemini.csv")


def project_path(value):
    """Resolve a sidebar path relative to the project root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


FEATURES = [ "cancellation", "delay", "boarding", "baggage",
 "customer_service",
 "cabin_crew", "ground_staff", "complaint_resolution", "seat_comfort", "food_beverage",
 "cabin_cleanliness", "entertainment_wifi", "booking_experience", "check_in",
 "airport_experience", "value_for_money", "pricing_fees", "flight_disruption", "overall_experience"]
# These are dashboard groupings, not a claim about a pre-existing taxonomy.
THEMES = {
 "Flight operations": ["cancellation", "delay", "flight_disruption"],
 "Airport & boarding": ["boarding", "check_in", "airport_experience", "ground_staff", "baggage"],
 "Staff & service": ["customer_service", "cabin_crew", "complaint_resolution"],
 "Onboard experience": ["seat_comfort", "food_beverage", "cabin_cleanliness", "entertainment_wifi"],
 "Booking & value": ["booking_experience", "value_for_money", "pricing_fees"],
 "Overall experience": ["overall_experience"],
}
SEGMENT_FIELDS = {"Cabin class": ["cabin_class", "travel_class", "class", "seat_class", "cabin"],
                  "Flight geography": ["flight_geography", "flight_type", "route_type", "journey_type", "flight_scope"]}
SEGMENT_FLAG_COLUMNS = {"Business": ["business_class", "is_business_class"],
                        "Economy": ["economy_class", "is_economy_class"],
                        "International": ["international", "is_international"],
                        "Domestic": ["domestic", "is_domestic"]}
VALIDATION = pd.DataFrame({"Metric": ["Overall accuracy", "Overall Macro F1", "Aspect detection F1"],
                           "Hugging Face": [0.738, 0.435, 0.310],
                           "Gemini": [0.964, 0.486, 0.821]})


# Existing segment metadata
def segment_series(frame, kind):
    """Read existing metadata/flags only. No new flight or cabin classification."""
    if kind == "cabin":
        candidates, allowed, flags = SEGMENT_FIELDS["Cabin class"], {"business": "Business", "business class": "Business", "economy": "Economy", "economy class": "Economy"}, ("Business", "Economy")
    else:
        candidates, allowed, flags = SEGMENT_FIELDS["Flight geography"], {"international": "International", "domestic": "Domestic"}, ("International", "Domestic")
    col = find_column(frame, candidates)
    if col is not None:
        raw = frame[col].astype("string").str.strip().str.lower()
        result = raw.map(allowed).fillna("Undetermined")
    else:
        result = pd.Series("Undetermined", index=frame.index, dtype="string")
    first = find_column(frame, SEGMENT_FLAG_COLUMNS[flags[0]])
    second = find_column(frame, SEGMENT_FLAG_COLUMNS[flags[1]])
    if first is not None and second is not None:
        a, b = as_bool(frame[first]), as_bool(frame[second])
        result = result.mask(a & ~b, flags[0]).mask(b & ~a, flags[1]).mask(a & b, "Undetermined")
    elif first is not None:
        result = result.mask(as_bool(frame[first]), flags[0])
    elif second is not None:
        result = result.mask(as_bool(frame[second]), flags[1])
    return result


AIRLINES = ["Aegean Airlines", "Turkish Airlines", "Air France", "Lufthansa", "Bangkok Air", "Porter Airlines", "Virgin Australia"]
EUROPE = ["Aegean Airlines", "Turkish Airlines", "Air France", "Lufthansa"]
REGIONAL = ["Aegean Airlines", "Bangkok Air", "Porter Airlines", "Virgin Australia"]
SENTIMENTS = ["positive", "neutral", "negative", "mixed"]
COLORS = {"positive":"#169b8f", "neutral":"#a0a8b4", "negative":"#e26d65", "mixed":"#dca24c"}

st.markdown("""<style>
.block-container{padding-top:1.3rem;padding-bottom:3rem}
[data-testid="stMetric"]{background:#f4f7fa;border:1px solid #e5eaf0;border-radius:12px;padding:12px}
@media(prefers-color-scheme:dark){[data-testid="stMetric"]{background:#202833;border-color:#384452}}
</style>""", unsafe_allow_html=True)


# Data ingestion and normalization
def find_column(df, candidates):
    """Find the first available column using case-insensitive names."""
    lookup = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]
    return None


def as_bool(s):
    """Explicit 1/true/yes only; missing/unclassified -> False."""
    return s.astype("string").str.strip().str.lower().isin(["1", "1.0", "true", "yes", "y"])


def sentiment(s):
    """Normalize recognized sentiment labels; leave unavailable labels missing."""
    v = s.astype("string").str.strip().str.lower()
    v = v.replace({"pos":"positive", "neg":"negative", "neu":"neutral",
                   "very positive":"positive", "very negative":"negative",
                   "not mentioned":pd.NA, "not_mentioned":pd.NA,
                   "none":pd.NA, "nan":pd.NA, "unknown":pd.NA, "":pd.NA})
    return v.where(v.isin(SENTIMENTS))


def feature_flag(s):
    """True for explicit positive binary flag; do not infer from sentiment."""
    return as_bool(s)


def read_csv(path):
    """Read an existing UTF-8 CSV without mixed-type inference warnings."""
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")


def unique_records(df, name):
    """Validate record keys and retain one row per review."""
    if "record_key" not in df.columns:
        raise ValueError(f"{name} is missing 'record_key'.")
    df = df.copy()
    df["record_key"] = df["record_key"].astype("string").str.strip()
    df = df[df["record_key"].notna() & df["record_key"].ne("")]
    duplicates = int(df.duplicated("record_key").sum())
    if duplicates:
        st.warning(f"{name}: {duplicates:,} duplicate record_key rows; retaining first. Verify upstream duplicates.")
        df = df.drop_duplicates("record_key", keep="first")
    return df


def build_data(hf_path, gem_path):
    """Join model exports, apply eligibility rules and prepare dashboard fields."""
    hf = unique_records(read_csv(Path(hf_path)), "Hugging Face")
    gm = unique_records(read_csv(Path(gem_path)), "Gemini")
    # Fail clearly when an incompatible export would otherwise produce false zeros.
    required_hf = ["overall_sentiment"] + [c for f in FEATURES for c in (f, f + "_sentiment")]
    required_gm = ["gemini_overall_sentiment"] + [c for f in FEATURES for c in (f + "_llm", f + "_sentiment_llm")]
    for name, frame, required in (("Hugging Face", hf, required_hf), ("Gemini", gm, required_gm)):
        missing = sorted(set(required) - set(frame.columns))
        if missing:
            raise ValueError(f"{name} export missing prediction columns: {', '.join(missing)}")
    # Keep HF fields as authoritative for review metadata and pre-existing eligibility flags.
    shared = [c for c in gm.columns if c in hf.columns and c != "record_key"]
    gm = gm.drop(columns=shared)
    merged = hf.merge(gm, on="record_key", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("No matching record_key values found between the two datasets.")
    # Prefer HF eligibility columns, fall back to explicit Gemini classification if HF absent.
    long_col = find_column(merged, ["long_haul", "is_long_haul"])
    first_col = find_column(merged, ["first_class", "is_first_class"])
    if long_col is None or first_col is None:
        raise ValueError("Expected existing long_haul and first_class flags were not found. No new classification will be performed.")
    long_ex = as_bool(merged[long_col])
    first_ex = as_bool(merged[first_col])
    eligible = merged.loc[~(long_ex | first_ex)].copy()
    summary = {"HF input":len(hf), "Gemini input":len(gm), "Matched":len(merged),
               "Long-haul flagged":int(long_ex.sum()), "First-class flagged":int(first_ex.sum()),
               "Excluded (union)":int((long_ex | first_ex).sum()), "Eligible":len(eligible),
               "Unclassified long-haul status (included unless first class)":int((~long_ex & merged[long_col].isna() & ~first_ex).sum())}
    metadata = {
        "airline": ["airline", "airline_name", "company"],
        "source": ["source", "review_source", "platform"],
        "date": ["review_date", "publication_date", "date", "published_date"],
        "language": ["original_language", "detected_language", "language", "review_language", "lang"],
        "review": ["review_en", "review_translated", "review_english", "translated_review", "review_text_english", "review", "review_text", "text"],
        "rating": ["rating", "overall_rating", "review_rating"],
    }
    missing = []
    for dest, candidates in metadata.items():
        c = find_column(eligible, candidates)
        if c is None:
            eligible["_"+dest] = pd.NA
            if dest in ("airline", "source", "date"):
                missing.append(dest)
        else:
            eligible["_"+dest] = eligible[c]
    if missing:
        raise ValueError("Required metadata columns not found: " + ", ".join(missing) + ". Check CSV schema.")
    eligible["_date"] = pd.to_datetime(eligible["_date"], errors="coerce", utc=True).dt.tz_localize(None)
    # Enforce the reporting start date globally, before any charts, KPIs, or filters.
    audit_pre_date = len(eligible)
    eligible = eligible.loc[eligible["_date"].ge(pd.Timestamp("2024-01-01"))].copy()
    summary["Before 2024 or missing review date (excluded)"] = audit_pre_date - len(eligible)
    summary["Eligible from 2024-01-01"] = len(eligible)
    eligible["_cabin_segment"] = segment_series(eligible, "cabin")
    eligible["_geography_segment"] = segment_series(eligible, "geography")
    eligible["_year"] = eligible["_date"].dt.year
    eligible["_month"] = eligible["_date"].dt.to_period("M").astype("string")
    eligible["_airline"] = eligible["_airline"].astype("string").str.strip().replace({
        "Aegean":"Aegean Airlines", "aegean_airlines":"Aegean Airlines",
        "air_france":"Air France", "turkish_airlines":"Turkish Airlines",
        "lufthansa":"Lufthansa", "porter_airlines":"Porter Airlines",
        "Bangkok Airways":"Bangkok Air", "bangkok_airways":"Bangkok Air",
        "virgin_australia":"Virgin Australia"})
    eligible = eligible.loc[eligible["_airline"].isin(AIRLINES)].copy()
    summary["Eligible benchmark airlines"] = len(eligible)
    eligible["_source"] = eligible["_source"].astype("string").str.strip()
    eligible["_language"] = eligible["_language"].astype("string").str.strip().replace("", pd.NA).fillna("Unknown")
    eligible["_hf_sentiment"] = sentiment(eligible["overall_sentiment"]) if "overall_sentiment" in eligible else pd.Series(pd.NA, index=eligible.index, dtype="string")
    eligible["_gemini_sentiment"] = sentiment(eligible["gemini_overall_sentiment"]) if "gemini_overall_sentiment" in eligible else pd.Series(pd.NA, index=eligible.index, dtype="string")
    for f in FEATURES:
        for model, flag_col, sent_col in [("hf",f,f+"_sentiment"),("gemini",f+"_llm",f+"_sentiment_llm")]:
            eligible[f"_flag_{model}_{f}"] = feature_flag(eligible[flag_col]) if flag_col in eligible else False
            eligible[f"_sent_{model}_{f}"] = sentiment(eligible[sent_col]) if sent_col in eligible else pd.Series(pd.NA,index=eligible.index,dtype="string")
    return eligible, summary, {"hf":hf.columns.tolist(), "gemini":gm.columns.tolist()}


# Cached loading and analytical helpers
@st.cache_data(show_spinner="Reading and matching ABSA datasets…", ttl=3600)
def cached_build(hf_path, gem_path, hf_mtime, gem_mtime):
    """Cache prepared data; modification times invalidate stale cache entries."""
    return build_data(hf_path, gem_path)


def pct(x, total):
    """Calculate a percentage, returning NaN for an empty denominator."""
    return 100*x/total if total else np.nan


def stats(data, sentiment_col):
    """Summarize recognized overall sentiment labels and net sentiment."""
    valid = data[data[sentiment_col].isin(SENTIMENTS)]
    n = len(valid)
    counts = valid[sentiment_col].value_counts()
    # Never unpack in SENTIMENTS order: its display order differs from metric names.
    pos = int(counts.get("positive", 0))
    neg = int(counts.get("negative", 0))
    neutral = int(counts.get("neutral", 0))
    mixed = int(counts.get("mixed", 0))
    return {"n":n, "positive":pct(pos,n), "negative":pct(neg,n), "neutral":pct(neutral,n),
            "mixed":pct(mixed,n), "net":pct(pos-neg,n)}


def grouped_sentiment(data, groups, column):
    """Calculate sentiment distributions within the requested groups."""
    subset = data[data[column].isin(SENTIMENTS)]
    if subset.empty:
        return pd.DataFrame(columns=groups+["n", "positive", "neutral", "negative", "mixed", "net"])
    tab = subset.groupby(groups+[column], dropna=False).size().unstack(fill_value=0)
    for s in SENTIMENTS:
        if s not in tab: tab[s] = 0
    tab["n"] = tab[SENTIMENTS].sum(axis=1)
    for s in SENTIMENTS: tab[s] = 100*tab[s]/tab["n"]
    tab["net"] = tab["positive"]-tab["negative"]
    return tab.reset_index()


def bars(df, x, y, title, color=None, orientation="v", height=390):
    """Render a labeled Plotly bar chart for a nonempty selection."""
    if df.empty:
        st.info("No data for this selection."); return
    fig = px.bar(df, x=x, y=y, color=color, orientation=orientation, title=title, height=height,
                 text=y if orientation == "v" else x)
    fig.update_traces(texttemplate="%{text:,.1f}" if pd.api.types.is_float_dtype(df[y if orientation == "v" else x]) else "%{text:,}",
                      textposition="outside", cliponaxis=False)
    fig.update_layout(margin=dict(l=15,r=45,t=55,b=45), legend_title_text="", uniformtext_minsize=9, uniformtext_mode="hide")
    st.plotly_chart(fig, use_container_width=True)


def heatmap(data, index, columns, values, title):
    """Render a matrix without treating missing observations as zero."""
    if data.empty: st.info("No observations for this view."); return
    matrix = data.pivot(index=index, columns=columns, values=values)
    fig = px.imshow(matrix, color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                    aspect="auto", title=title, text_auto=".1f")
    fig.update_layout(height=max(380, 34*len(matrix)+130))
    st.plotly_chart(fig, use_container_width=True)


def sentiment_visual(data, col, group, metric):
    """Display grouped sentiment using the selected reporting metric."""
    agg = grouped_sentiment(data, [group], col)
    if agg.empty: st.info("No classified sentiment in this selection."); return
    if metric == "Full distribution":
        long = agg.melt(id_vars=[group,"n"], value_vars=SENTIMENTS, var_name="Sentiment", value_name="Percent")
        fig = px.bar(long, x=group, y="Percent", color="Sentiment", color_discrete_map=COLORS,
                     barmode="stack", title="Sentiment distribution (%)", custom_data=["n"], text="Percent")
        fig.update_traces(texttemplate="%{y:.1f}%", textposition="inside",
                          insidetextanchor="middle",
                          hovertemplate="%{x}<br>%{fullData.name}: %{y:.1f}%<extra></extra>")
        fig.update_layout(uniformtext_minsize=9, uniformtext_mode="hide")
        st.plotly_chart(fig, use_container_width=True)
    else:
        key = {"Net sentiment":"net", "Positive share":"positive", "Negative share":"negative"}[metric]
        agg = agg.sort_values(key)
        bars(agg, key, group, metric+" by "+group.lstrip("_"), orientation="h")
    st.dataframe(agg.round(2), use_container_width=True, hide_index=True)


def feature_table(data, model):
    """Summarize explicit feature mentions and recognized aspect sentiment."""
    records=[]
    n=len(data)
    for f in FEATURES:
        flags = data[f"_flag_{model}_{f}"]
        sent = data.loc[flags, f"_sent_{model}_{f}"]
        counts = sent.value_counts()
        known = int(sent.isin(SENTIMENTS).sum())
        mentions = int(flags.sum())
        pos = int(counts.get("positive",0)); neg=int(counts.get("negative",0))
        records.append({"feature":f,"theme":next((t for t,fs in THEMES.items() if f in fs),"Other"),
                        "mentions":mentions,"mention_rate":pct(mentions,n),"classified_mentions":known,
                        "negative_mentions":neg,"negative_share":pct(neg,known),
                        "positive_share":pct(pos,known),"net":pct(pos-neg,known)})
    return pd.DataFrame(records)


def review_explorer(data, model, feature=None):
    """Display and export review evidence for the selected segment."""
    st.subheader("Review evidence explorer")
    d = data.copy()
    if feature and feature != "All features": d = d[d[f"_flag_{model}_{feature}"]]
    search = st.text_input("Search review text", key="search_"+model+str(feature))
    if search:
        d = d[d["_review"].astype("string").str.contains(re.escape(search), case=False, na=False, regex=True)]
    cols = ["record_key", "_date", "_airline", "_source", "_language", "_review", "_hf_sentiment", "_gemini_sentiment"]
    if feature and feature != "All features": cols.append(f"_sent_{model}_{feature}")
    st.caption(f"{len(d):,} matching reviews. Text is unedited; review authors are not displayed.")
    st.dataframe(d[cols].head(1500), use_container_width=True, hide_index=True)
    st.download_button("Export selected reviews (CSV)", d[cols].to_csv(index=False).encode("utf-8-sig"),
                       "aegean_review_evidence.csv", "text/csv", key="export_"+model+str(feature))


# Sidebar controls and shared filters
st.sidebar.title("✈️ AEGEAN Intelligence")
with st.sidebar.expander("Data file paths", expanded=False):
    hf_path = st.text_input("Hugging Face CSV", str(HF_DEFAULT))
    gem_path = st.text_input("Gemini CSV", str(GEM_DEFAULT))
    st.caption("Relative paths are resolved from the project root; absolute overrides are optional.")
try:
    hf_file, gm_file = project_path(hf_path), project_path(gem_path)
    df, audit, schemas = cached_build(str(hf_file),str(gm_file),hf_file.stat().st_mtime,gm_file.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not load the ABSA data: {exc}")
    st.info("Verify the paths in the sidebar and inspect the source column names. The app does not generate placeholder data.")
    st.stop()

period = st.sidebar.selectbox("Review period", ["Entire period", "2024", "2025", "2026", "2024 + 2025"])
benchmark = st.sidebar.selectbox("Benchmark", ["European", "Global regional", "All seven airlines"])
choices = EUROPE if benchmark=="European" else REGIONAL if benchmark=="Global regional" else AIRLINES
present = [a for a in choices if a in df["_airline"].dropna().unique()]
airlines = st.sidebar.multiselect("Airlines", present, default=present)
sources = sorted(df["_source"].dropna().unique().tolist())
selected_sources = st.sidebar.multiselect("Review sources", sources, default=sources)
cabin_choice = st.sidebar.selectbox("Cabin class", ["All", "Business", "Economy", "Undetermined"], index=0)
geography_choice = st.sidebar.selectbox("Flight geography", ["All", "International", "Domestic", "Undetermined"], index=0)
model_choice = st.sidebar.selectbox("Sentiment model", ["Gemini", "Hugging Face", "Compare both"])
metric = st.sidebar.selectbox("Sentiment metric", ["Net sentiment", "Positive share", "Negative share", "Full distribution"])
minimum_n = st.sidebar.number_input("Small-cell warning threshold", min_value=1, max_value=1000, value=30, step=5)

mask = df["_airline"].isin(airlines) & df["_source"].isin(selected_sources)
if period == "2024": mask &= df["_year"].eq(2024)
elif period == "2025": mask &= df["_year"].eq(2025)
elif period == "2026": mask &= df["_year"].eq(2026)
elif period == "2024 + 2025": mask &= df["_year"].isin([2024,2025])
if cabin_choice != "All": mask &= df["_cabin_segment"].eq(cabin_choice)
if geography_choice != "All": mask &= df["_geography_segment"].eq(geography_choice)
d = df.loc[mask].copy()
active_model = "hf" if model_choice=="Hugging Face" else "gemini"
active_col = "_hf_sentiment" if active_model=="hf" else "_gemini_sentiment"

st.title("AEGEAN | Customer Experience Intelligence")
st.caption("Public review sample from 01/01/2024 · Matched Hugging Face & Gemini ABSA · Excludes flagged long-haul and first-class reviews")
if len(d)==0: st.warning("The selected filters return no reviews.")
page = st.sidebar.radio("Dashboard", ["Executive Overview", "EDA & Model Validation", "Overall Sentiment", "Feature Analysis", "Strategic Insights", "Thank You"])

# Dashboard pages
if page == "Executive Overview":
    st.header("Executive overview")
    ae = d[d["_airline"]=="Aegean Airlines"]
    ae_s = stats(ae,active_col)
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Eligible matched reviews", f"{len(d):,}")
    c2.metric("AEGEAN reviews", f"{len(ae):,}")
    c3.metric("AEGEAN positive", f"{ae_s['positive']:.1f}%" if ae_s["n"] else "N/A")
    c4.metric("AEGEAN negative", f"{ae_s['negative']:.1f}%" if ae_s["n"] else "N/A")
    c5.metric("AEGEAN net sentiment", f"{ae_s['net']:+.1f} pp" if ae_s["n"] else "N/A")
    left,right = st.columns(2)
    with left: sentiment_visual(d,active_col,"_airline",metric)
    with right:
        trend = grouped_sentiment(d.dropna(subset=["_date"]),["_month","_airline"],active_col)
        if not trend.empty:
            fig = px.line(trend, x="_month",y="net",color="_airline",markers=True,
                          title="Monthly net sentiment (percentage points)")
            st.plotly_chart(fig,use_container_width=True)
    st.subheader("AEGEAN feature opportunities")
    ft = feature_table(ae,active_model)
    focus = ft[~ft["feature"].isin(["overall_experience"])]
    st.dataframe(focus.sort_values("negative_mentions",ascending=False).head(10).round(2),hide_index=True,use_container_width=True)
    st.caption("Descriptive public-review signals, not representative passenger satisfaction or causal findings.")

elif page == "EDA & Model Validation":
    st.header("Matched sample: EDA and model validation")
    with st.expander("Eligibility and matching audit",expanded=True):
        st.dataframe(pd.DataFrame([{"Measure":k,"Reviews":v} for k,v in audit.items()]),hide_index=True,use_container_width=True)
        st.caption("Unknown long-haul classifications are included unless first class is explicitly flagged. Exclusions use existing columns only.")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Filtered reviews",f"{len(d):,}")
    c2.metric("Airlines",d["_airline"].nunique())
    c3.metric("Sources",d["_source"].nunique())
    c4.metric("Languages",d.loc[d["_language"].ne("Unknown"), "_language"].nunique())
    a,b = st.columns(2)
    with a:
        bars(d["_airline"].value_counts().rename_axis("Airline").reset_index(name="Reviews"),"Airline","Reviews","Reviews by airline")
        bars(d["_year"].value_counts().sort_index().rename_axis("Year").reset_index(name="Reviews").assign(Year=lambda x: x["Year"].astype(str)),"Year","Reviews","Reviews by year")
    with b:
        bars(d["_source"].value_counts().rename_axis("Source").reset_index(name="Reviews"),"Source","Reviews","Reviews by source")
        lang = d.loc[d["_language"].ne("Unknown"), "_language"].value_counts()
        lang = pd.concat([lang.head(7),pd.Series({"Other":int(lang.iloc[7:].sum())})]) if len(lang)>7 else lang
        bars(lang.rename_axis("Language").reset_index(name="Reviews"),"Language","Reviews","Original language distribution")
    st.subheader("Human-annotated validation — previously reported results")
    st.caption("Fixed results from the earlier evaluation, not recalculated for the current airline/source/cabin/geography/date filters. Accuracy requires human ground truth; agreement below does not.")
    st.dataframe(VALIDATION.style.format({"Hugging Face":"{:.3f}", "Gemini":"{:.3f}"}), hide_index=True, use_container_width=True)
    validation_long = VALIDATION.melt(id_vars="Metric", var_name="Model", value_name="Score")
    fig = px.bar(validation_long, x="Metric", y="Score", color="Model", barmode="group", text="Score", title="Human-validation performance: Hugging Face vs Gemini")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_yaxes(range=[0,1.12])
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Hugging Face vs Gemini: overall sentiment agreement")
    paired = d.dropna(subset=["_hf_sentiment","_gemini_sentiment"])
    if len(paired):
        agree = (paired["_hf_sentiment"]==paired["_gemini_sentiment"]).mean()*100
        kappa = cohen_kappa_score(paired["_hf_sentiment"],paired["_gemini_sentiment"],labels=SENTIMENTS)
        c1,c2,c3=st.columns(3)
        c1.metric("Paired classifications",f"{len(paired):,}")
        c2.metric("Agreement",f"{agree:.1f}%")
        c3.metric("Cohen's kappa",f"{kappa:.3f}" if np.isfinite(kappa) else "N/A")
        matrix = pd.crosstab(paired["_hf_sentiment"],paired["_gemini_sentiment"]).reindex(index=SENTIMENTS,columns=SENTIMENTS,fill_value=0)
        fig=px.imshow(matrix, text_auto=True, title="Confusion matrix — HF rows, Gemini columns",aspect="auto")
        st.plotly_chart(fig,use_container_width=True)
        by = paired.assign(agree=paired["_hf_sentiment"].eq(paired["_gemini_sentiment"])).groupby("_airline").agg(n=("agree","size"),agreement=("agree","mean")).reset_index()
        by["agreement"]*=100
        st.dataframe(by.round(2),use_container_width=True,hide_index=True)
        with st.expander("Model disagreements — inspect reviews"):
            st.dataframe(paired.loc[paired["_hf_sentiment"]!=paired["_gemini_sentiment"],
                         ["record_key","_airline","_source","_review","_hf_sentiment","_gemini_sentiment"]].head(1000),hide_index=True,use_container_width=True)
    else: st.warning("No paired, recognized overall sentiment labels. Inspect sentiment mappings.")
    with st.expander("Column schema diagnostics"):
        st.write("HF input columns",schemas["hf"])
        st.write("Gemini-specific input columns",schemas["gemini"])
        st.write("Missing original dates",int(d["_date"].isna().sum()))
        st.write("Missing review texts",int(d["_review"].isna().sum()))

elif page == "Overall Sentiment":
    st.header("Overall experience sentiment")
    st.caption("Positive, neutral, negative and mixed are distinct. Net = positive share − negative share; denominator = recognized model labels.")
    if model_choice=="Compare both":
        l,r=st.columns(2)
        with l:
            st.subheader("Gemini")
            sentiment_visual(d,"_gemini_sentiment","_airline",metric)
        with r:
            st.subheader("Hugging Face")
            sentiment_visual(d,"_hf_sentiment","_airline",metric)
    else: sentiment_visual(d,active_col,"_airline",metric)
    st.subheader("Source-adjusted comparison")
    agg = grouped_sentiment(d,["_airline","_source"],active_col)
    heatmap(agg,"_airline","_source","net","Net sentiment by airline and source (pp)")
    st.caption("Cells without observations are missing, not zero. Compare airline-source cells with sufficient review counts.")
    st.dataframe(agg.round(2),hide_index=True,use_container_width=True)
    st.subheader("Time evolution")
    trend = grouped_sentiment(d.dropna(subset=["_date"]),["_month","_airline"],active_col)
    if not trend.empty:
        fig=px.line(trend,x="_month",y="net",color="_airline",markers=True,hover_data=["n"],title="Monthly net sentiment")
        st.plotly_chart(fig,use_container_width=True)
        small = trend[trend["n"]<minimum_n]
        if len(small): st.warning(f"{len(small)} monthly airline cells have fewer than {minimum_n} classified reviews.")
    st.subheader("Source-standardized benchmark (common-source equal weighting)")
    if not agg.empty:
        pivot=agg.pivot(index="_source",columns="_airline",values="net")
        common=pivot.dropna(axis=0,how="any")
        if len(common):
            adjusted=common.mean().sort_values(ascending=False).rename("Mean net sentiment (pp)").reset_index()
            st.dataframe(adjusted.round(2),hide_index=True,use_container_width=True)
            st.caption(f"Equal weight per shared source, using {len(common)} source(s) represented by every selected airline. Descriptive, not causal adjustment.")
        else: st.info("No source is shared by every selected airline. Standardized benchmark is unavailable.")
    review_explorer(d,active_model)

elif page == "Feature Analysis":
    st.header("Feature-level customer experience")
    st.caption("Passenger-experience ABSA features only. Cabin class and flight geography are sidebar segments, not sentiment features; long-haul and first-class are eligibility exclusions.")
    if model_choice=="Compare both":
        st.info("Feature charts show Gemini by default. Select Hugging Face to switch; model comparisons are shown below.")
    ft = feature_table(d,active_model)
    theme = st.selectbox("Theme",["All themes"]+list(THEMES))
    feature_options=[f for f in FEATURES if theme=="All themes" or f in THEMES[theme]]
    selected_feature=st.selectbox("Feature",["All features"]+feature_options)
    view = ft[ft["feature"].isin(feature_options)]
    if selected_feature!="All features": view=view[view["feature"]==selected_feature]
    a,b=st.columns(2)
    with a: bars(view.sort_values("mentions",ascending=False),"mentions","feature","Feature mention counts",orientation="h",height=520)
    with b: bars(view.sort_values("negative_mentions",ascending=False),"negative_mentions","feature","Negative feature mentions",orientation="h",height=520)
    st.dataframe(view.round(2),hide_index=True,use_container_width=True)
    st.subheader("Theme mentions (reviews may mention multiple themes)")
    tdf = view.groupby("theme",as_index=False)[["mentions","negative_mentions"]].sum().sort_values("mentions",ascending=False)
    bars(tdf,"theme","mentions","Theme mentions (sum of feature mentions)")
    st.subheader("Feature × airline heatmap")
    records=[]
    for airline, subset in d.groupby("_airline"):
        for _, row in feature_table(subset,active_model).iterrows():
            if row["feature"] in feature_options:
                records.append({"Airline":airline,"Feature":row["feature"],"Net sentiment":row["net"],"n":row["classified_mentions"]})
    heatmap(pd.DataFrame(records),"Feature","Airline","Net sentiment","Feature net sentiment (pp)")
    st.subheader("Feature prioritization matrix")
    matrix=view.dropna(subset=["negative_share","mention_rate"])
    if not matrix.empty:
        fig=px.scatter(matrix,x="negative_share",y="mention_rate",size="negative_mentions",hover_name="feature",
                       color="theme",size_max=45,title="Negative share vs mention rate; bubble = negative mentions",
                       hover_data=["mentions","classified_mentions"])
        st.plotly_chart(fig,use_container_width=True)
    st.subheader("Feature detection: model agreement")
    agreement=[]
    for f in feature_options:
        a=d[f"_flag_hf_{f}"]; b=d[f"_flag_gemini_{f}"]
        agreement.append({"Feature":f,"HF mentions":int(a.sum()),"Gemini mentions":int(b.sum()),
                          "Binary agreement (%)":100*float((a==b).mean()) if len(d) else np.nan})
    st.dataframe(pd.DataFrame(agreement).round(2),hide_index=True,use_container_width=True)
    review_explorer(d,active_model,selected_feature)

elif page == "Strategic Insights":
    st.header("Strategic insights and priorities")
    st.caption("Exploratory ranking of customer pain points. Priority weights are configurable assumptions, not validated causal effects.")
    a,b,c=st.columns(3)
    with a: volume_w=st.slider("Negative volume weight",0,100,40)
    with b: gap_w=st.slider("Competitor gap weight",0,100,35)
    with c: trend_w=st.slider("Deterioration weight",0,100,25)
    ae=d[d["_airline"]=="Aegean Airlines"]
    peers=d[d["_airline"]!="Aegean Airlines"]
    if ae.empty:
        st.warning("Select AEGEAN in the airline filter to calculate AEGEAN priorities.")
    else:
        ft=feature_table(ae,active_model)
        peer_ft=feature_table(peers,active_model).set_index("feature") if len(peers) else pd.DataFrame()
        ft=ft[~ft["feature"].isin(["overall_experience"])].copy()
        ft["competitor_gap"] = ft.apply(lambda r: (r["negative_share"]-peer_ft.loc[r["feature"],"negative_share"]) if not peer_ft.empty and r["feature"] in peer_ft.index else np.nan,axis=1)
        # Difference in negative share between 2026 and 2024+2025 (if present)
        prior=feature_table(ae[ae["_year"].isin([2024,2025])],active_model).set_index("feature")
        recent=feature_table(ae[ae["_year"].eq(2026)],active_model).set_index("feature")
        ft["deterioration"] = ft["feature"].map(recent["negative_share"]-prior["negative_share"])
        def norm(v):
            """Scale nonnegative priority components to a maximum of 100."""
            v=pd.to_numeric(v,errors="coerce").clip(lower=0)
            mx=v.max(skipna=True)
            return 100*v/mx if pd.notna(mx) and mx>0 else v*0
        ft["volume_score"]=norm(ft["negative_mentions"])
        ft["gap_score"]=norm(ft["competitor_gap"])
        ft["trend_score"]=norm(ft["deterioration"])
        weight_sum=volume_w+gap_w+trend_w
        if not weight_sum: st.warning("At least one priority weight must be positive.")
        else:
            # Missing gap/trend = zero contribution; preserve visibility of missing evidence.
            ft["priority_score"]=(volume_w*ft["volume_score"].fillna(0)+gap_w*ft["gap_score"].fillna(0)+trend_w*ft["trend_score"].fillna(0))/weight_sum
            ft=ft.sort_values("priority_score",ascending=False)
            st.dataframe(ft[["feature","mentions","negative_mentions","negative_share","competitor_gap","deterioration","priority_score"]].round(2),hide_index=True,use_container_width=True)
            bars(ft.head(12),"priority_score","feature","Prioritized experience issues",orientation="h",height=510)
            st.caption("Competitor gap is AEGEAN negative share minus pooled peers; positive means AEGEAN has more negative feature sentiment. Trend compares 2026 (partial) to 2024–2025 combined. Missing evidence contributes zero, not proof of no problem.")
            if (ft["classified_mentions"]<minimum_n).any():
                st.warning(f"Some feature scores rely on fewer than {minimum_n} classified AEGEAN mentions; review denominators before action.")
        st.subheader("Evidence supporting an issue")
        chosen=st.selectbox("Investigate feature",ft["feature"].tolist())
        review_explorer(ae,active_model,chosen)

elif page == "Thank You":
    st.markdown("<div style='height:7vh'></div>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align:center;font-size:4rem;color:#C9A45C'>Thank You!</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center'>Michalis Fessas, PhD</h2>", unsafe_allow_html=True)
    qr_path = Path(__file__).parent / "assets" / "linkedin_qr.svg"
    if qr_path.exists():
        import base64
        qr = base64.b64encode(qr_path.read_bytes()).decode("ascii")
        st.markdown(f"<div style='text-align:center;margin-top:2rem'><img src='data:image/svg+xml;base64,{qr}' width='260' alt='LinkedIn QR code' /></div>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center'>Connect with me on LinkedIn</p>", unsafe_allow_html=True)
    else:
        st.warning("LinkedIn QR image not found in assets/linkedin_qr.svg")

st.sidebar.divider()
st.sidebar.caption(f"Eligible matched dataset (from 2024-01-01): {len(df):,} reviews | Current selection: {len(d):,}")
st.sidebar.caption("2026 is a partial year. Public review data are not a representative passenger survey.")
