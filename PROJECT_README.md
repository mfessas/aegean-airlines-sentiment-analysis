# AEGEAN Airlines — Technical project documentation

**Public Perception and Sentiment Benchmarking against Regional and European Airlines**  
Author: Michalis Fessas, PhD  
Documentation baseline: November 2026

## 1. Purpose and scope

This prototype collects public airline reviews, harmonizes heterogeneous sources, translates non-English text, detects passenger-experience aspects and compares sentiment across airlines. It supports descriptive competitive benchmarking and identification of potential improvement priorities. It uses no internal airline data and does not estimate representative passenger satisfaction or causal effects.

| European benchmark | Global regional benchmark |
|---|---|
| Aegean Airlines | Aegean Airlines |
| Turkish Airlines | Bangkok Air (Bangkok Airways) |
| Air France | Porter Airlines |
| Lufthansa | Virgin Australia |

The analytical reporting window starts on 2024-01-01. The 2026 observations constitute a partial year. Raw and intermediate datasets may retain earlier records for auditability.

## 2. Architecture and directory layout

```text
 aegean-airlines/
 ├── README.md
 ├── PROJECT_README.md
 ├── Python Code/
 │   ├── Scraping/
 │   ├── Data Cleaning/
 │   ├── Hugging Face/
 │   ├── LLM/
 │   └── Model Validation/
 ├── Data/
 │   ├── Scraping/
 │   └── Output/
 │       ├── Cleaning data/
 │       │   └── Legacy/
 │       ├── Hugging Face/
 │       └── LLM/
 │           ├── Pilot/
 │           └── validation_results/ (generated)
 └── Streamlit/
     ├── app.py
     ├── README.md
     ├── requirements.txt
     └── assets/linkedin_qr.svg
```

The three main directories are retained. Notebooks locate the project root from their working directory; start Jupyter within the project. The Streamlit application resolves its root from its script location. Create the Data and Streamlit directories if they are absent after a Git clone (Git does not track empty directories).

## 3. Environment and installation

Python 3.10+ is a reasonable starting point; the full environment has not been version-pinned or integration-tested. Use an isolated environment. From the repository root:

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` on Windows PowerShell or `source .venv/bin/activate` on macOS/Linux.

For dashboard-only use:

```bash
python -m pip install -r Streamlit/requirements.txt
python -m streamlit run Streamlit/app.py
```

For full pipeline development, install the following import-derived dependency groups as needed. These are indicative, not a tested lockfile:

| Stage | Packages / prerequisites |
|---|---|
| Shared | jupyterlab, pandas, numpy, tqdm, openpyxl |
| Scraping | requests, beautifulsoup4, lxml, selenium, playwright, apify-client; Chrome for Selenium |
| Cleaning and translation | lingua-language-detector, openai, airportsdata |
| Hugging Face | torch, transformers, airportsdata, scikit-learn |
| LLM | google-genai, anthropic |
| Validation | scikit-learn, matplotlib, seaborn, openpyxl |
| Dashboard | Streamlit/requirements.txt |

Install Chromium for Playwright with `python -m playwright install chromium`. Select the appropriate PyTorch build for the machine and CUDA configuration. Start notebooks with `python -m jupyterlab`. Review notebook-specific install cells and SDK compatibility before running.

Credentials are needed only for applicable stages: `APIFY_TOKEN`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY` (pilot). Set these in the environment or use the notebooks' interactive prompts. Never commit credentials or notebook outputs containing them. Confirm API model availability, quota and pricing before execution; model identifiers below are those configured in the delivered code, not independently verified provider offerings.

## 4. Execution order and data contracts

### 4.1 Collection

Run the six notebooks under `Python Code/Scraping/` selectively. Each source has different pagination, coverage and access constraints. Respect site terms, rate limits and applicable privacy requirements; do not bypass access restrictions.

| Notebook | Consolidated input to cleaning (relative to Data/Scraping/) |
|---|---|
| Airline Ratings scraper.ipynb | airlineratings_data/all_airlines_reviews.csv |
| Flight Review Scraper.ipynb | flight_report_data/all_airlines_reviews.csv |
| eSky Scraper.ipynb | esky/esky_all_reviews_v2.csv |
| Kayak Scraping.ipynb | kayak/kayak_all_airline_reviews.csv |
| TripAdvisor_scraper_apify.ipynb | tripadvisor_reviews/tripadvisor_all_airlines.csv |
| Trustpilot - Apify Scraper.ipynb | trustpilot_reviews/trustpilot_all_airlines.csv |

The AirlineRatings notebook delegates to `airlineratings_multi_airline_paginated_v2.py`. Kayak and eSky use Selenium; Flight-Report uses requests/BeautifulSoup; AirlineRatings uses Playwright; TripAdvisor and Trustpilot use Apify. Preserve per-airline exports and checkpoints for recovery, but ingest consolidated files only to avoid double counting overlapping exports. Verify source/airline coverage after every run.

### 4.2 Cleaning and translation

Run `Python Code/Data Cleaning/Airline_Reviews_Pipeline_Updated(1).ipynb`. Harmonize schemas and airline names, parse dates, deduplicate using stable record keys, detect original language, translate non-English reviews using the configured `gpt-4.1-mini` model, and preserve translation status. Output directory: `Data/Output/Cleaning data/`.

Key outputs: `airline_reviews_cleaned.csv`, `airline_reviews_translated.csv`, `df_final.csv`, `translation_checkpoint.pkl`, `airline_review_summary.xlsx`.

`Clean_Data(1).ipynb` is a legacy workflow with isolated outputs under `Legacy/`. Do not run it after the main cleaning stage. `Incremental_reviews_pipeline.ipynb` is an alternative maintenance workflow; inspect `NEW_FILES`, retain backups and check record-key stability before appending. It uses `incremental_translation_checkpoint.csv`. The legacy `airline_translation_checkpoint.pkl` is not interchangeable with the main checkpoint; never unpickle untrusted files.

### 4.3 Feature engineering

Run `Python Code/Hugging Face/Airline_Feature_Engineering_v2(1).ipynb` with `airline_features_v2.py` alongside it. Input: `Data/Output/Cleaning data/df_final.csv`. The module uses contextual regular expressions, negation/hypothetical checks, airport references and approximate flight duration. It retains evidence, uncertainty and conflict diagnostics. Evidence-strength scores are heuristic, not calibrated probabilities.

The 25 classifications are:

| Group | Fields |
|---|---|
| Eligibility | long_haul, first_class |
| Cabin and geography | economy_class, business_class, international, domestic |
| Flight operations | cancellation, delay, flight_disruption |
| Airport and boarding | boarding, check_in, airport_experience, ground_staff, baggage |
| Staff and service | customer_service, cabin_crew, complaint_resolution |
| Onboard | seat_comfort, food_beverage, cabin_cleanliness, entertainment_wifi |
| Booking and value | booking_experience, value_for_money, pricing_fees |
| Overall | overall_experience |

Outputs under `Data/Output/Hugging Face/`: `df_final_features.csv`, `df_final_features_no_first_class_no_long_haul.csv`, `classification_diagnostics.csv`, `classification_conflicts.csv`, `classification_validation_sample.csv`, `feature_summary.csv`, `feature_summary_by_airline.csv`, `feature_summary_by_source.csv`. Diagnostic summaries may describe the unfiltered engineering dataset.

### 4.4 Hugging Face baseline

Run `NoLLM_Sentiment_Analysis.ipynb` in the Hugging Face directory. Input: `df_final_features.csv`. The configured model is `cardiffnlp/twitter-roberta-base-sentiment-latest`; rule-based aspect evidence is combined with sentiment inference and aggregation. CPU/GPU selection is automatic in the notebook. The name NoLLM denotes the non-generative baseline, not the absence of a pretrained language model.

Outputs: `df_final_absa.csv`, `df_final_absa_no_first_class_no_long_haul.csv`, `absa_diagnostics.csv`, `absa_summary.csv`, `absa_airline_sentiment_summary.csv` in the same output directory. Preserve the complete export for downstream matching.

### 4.5 Generative LLM analysis

Run selected sections of `Python Code/LLM/LLM_sentiment_analysis.ipynb`. Input: `Data/Output/Hugging Face/df_final_absa.csv`. The pilot compares configured identifiers `claude-haiku-5-5` and `gemini-3.1-flash-lite`; full production classification uses the latter. The full run is configured for 30 concurrent requests. Review costs and limits before starting. **Do not use Run All indiscriminately**: historical pilot, production and recovery cells coexist.

Pilot exports/checkpoints belong under `Data/Output/LLM/Pilot/`. Production outputs under `Data/Output/LLM/` include `df_final_absa_gemini.csv`, `df_final_absa_gemini_complete.csv`, `gemini_full_checkpoint.jsonl`, `gemini_full_failures.jsonl`, `gemini_diagnostics.csv`, `gemini_cost_summary.csv`, `gemini_vs_huggingface.csv`. The successful-classification export is the dashboard input. Resume only with compatible record keys, payload hashes and prompt versions. Failed/unclassified requests are not neutral sentiment; inspect failure logs before retrying.

### 4.6 Human validation

Run `Python Code/Model Validation/model_validation.ipynb` using HF and Gemini exports, optional Haiku pilot predictions and the manually completed `Data/Output/LLM/model_validation_100_reviewsv3.xlsx`. The human labels must be supplied separately; the notebook does not invent ground truth. Match records and report coverage, accuracy, macro F1, aspect detection, agreement, error analyses and uncertainty. Outputs include `model_validation_results.xlsx`, `presentation_validation_summary.xlsx`, CSV tables and charts under `Data/Output/LLM/validation_results/`.

Previously reported values retained in the dashboard:

| Metric | Hugging Face | Gemini |
|---|---:|---:|
| Overall accuracy | 0.738 | 0.964 |
| Overall macro F1 | 0.435 | 0.486 |
| Aspect detection F1 | 0.310 | 0.821 |

These historical evaluation values were not recomputed for this documentation and are not filter-specific. Accuracy and agreement are different quantities; inter-model agreement does not establish correctness. The reported scores favor Gemini on these three measures, but do not establish universal superiority.

### 4.7 Streamlit

The dashboard needs only the two production ABSA CSVs at the paths above. No API credentials, GPU, raw reviews from separate sources or validation workbook are required to launch it. The six pages are Executive Overview, EDA & Model Validation, Overall Sentiment, Feature Analysis, Strategic Insights and Thank You. The sidebar controls reporting period, benchmark, airline, source, cabin, geography, model and sentiment metric. The app is read-only with respect to source CSVs; review evidence can be downloaded as CSV. See [Streamlit README](Streamlit/README.md).

## 5. Core data dictionary

| Field / pattern | Meaning |
|---|---|
| record_key | Stable source-record identifier; unique, non-null join key |
| airline, source | Normalized airline and review platform |
| review_date, rating | Publication date and source-provided rating; rating scales may differ |
| review, review_en | Original review and English analysis text |
| original_language, translation_status | Language detection and translation processing state |
| origin, destination, route, flight_number | Optional journey metadata when available |
| long_haul, first_class | Existing binary eligibility flags |
| regional, european | Benchmark membership flags |
| <feature>, <feature>_sentiment | HF aspect detection and sentiment |
| overall_sentiment | HF overall review sentiment |
| <feature>_llm, <feature>_sentiment_llm | Gemini aspect detection and sentiment |
| gemini_overall_sentiment | Gemini overall review sentiment |

This is a logical dictionary, not an exhaustive physical schema. Inspect CSV headers and diagnostics before integration. Aspect detection is distinct from sentiment polarity. Unknown binary evidence must not be interpreted as confirmed absence.

## 6. Analytical rules and metrics

The dashboard inner-joins HF and Gemini by `record_key`, uses existing HF eligibility flags preferentially, excludes explicit long-haul or first-class flags, retains unknown haul status unless first class is flagged, excludes invalid or pre-2024 dates, and restricts to the seven airlines. Do not reclassify reviews at dashboard time. Recalculate the matched sample dynamically; do not assume the historical count is unchanged.

The project uses positive, negative, neutral, mixed and not_mentioned labels. Dashboard sentiment distributions use the four recognized polarity labels; not_mentioned/unclassified is excluded from sentiment denominators. Aspect mention rate = flagged reviews / selected reviews. Aspect positive or negative share = respective sentiment count / classified flagged mentions. Net sentiment = positive share minus negative share (percentage points). Source-standardized net sentiment uses equal weights across shared sources. Strategic scores combine normalized negative volume, pooled-peer gap and deterioration with configurable weights; missing components contribute zero, not evidence of absence.

## 7. Reproducibility, maintenance and limitations

- **Known threshold discrepancy:** intended long-haul definition is at least six hours; historical rule code and prompts use strictly greater than six hours. Preserve existing flags for published-result reproducibility. Quantify affected records and version any future correction before rerunning models.
- Public reviews are self-selected, platform coverage is unequal, translations and models can introduce errors, and route duration is approximate. Missing classifications are not verified negatives. Compare denominators and source mix; 2026 is incomplete.
- Preserve source snapshots, collection dates, model identifiers, prompt hashes, checkpoints and validation workbook versions privately. Record package versions and runtime environment for each reproducible run. Use `python -m pip freeze` to capture the actual tested environment, not as a substitute for testing.
- Before updates, back up existing data; verify unique keys, per-source counts, date range, translation coverage, feature schema, failure counts, matched sample size and exclusion counts. Confirm all six dashboard pages and empty-filter behavior locally. Reconcile metrics with presentation tables before publication.
- The delivered code was previously statically checked and selected dashboard data functions tested on synthetic data; this documentation does not certify live scraping, paid API execution, real-data accuracy or browser behavior. Production CSVs and human labels are not included in the inspected code archive.

## 8. Repository publication and governance

Publish code and documentation only after reviewing rights, site terms and privacy obligations. Keep raw/translated reviews, reviewer identifiers, human annotations, checkpoints, credentials, browser dumps and cost logs out of public Git history. Review notebook cell outputs and the LinkedIn QR asset before publishing. Use private data storage and sanitized examples only if authorized. No license is asserted by this document; establish appropriate code and data permissions before redistribution.

Suggested `.gitignore` entries (adapt to your repository):

```gitignore
.venv/
.env
.env.*
!.env.example
__pycache__/
.ipynb_checkpoints/
.streamlit/secrets.toml
Data/Scraping/**
Data/Output/**
*.pkl
*.jsonl
```

Ignoring files does not remove already tracked content or historical secrets. Review Git history and rotate exposed credentials if necessary. Keep the two README files at repository root; retain the separate Streamlit README for application-specific details.
