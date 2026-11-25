# AEGEAN Customer Experience Intelligence

## Installation and launch

From the project root (the folder containing `Python Code`, `Data`, and `Streamlit`), run:

```sh
python -m pip install -r Streamlit/requirements.txt
python -m streamlit run Streamlit/app.py
```

Python 3.10+ is recommended. The requirements file covers the dashboard only, not scraping or model inference. Paths are resolved from the location of `app.py`, so the project can be moved. Sidebar overrides are project-relative by default.

## Required inputs

- `Data/Output/Hugging Face/df_final_absa.csv`
- `Data/Output/LLM/df_final_absa_gemini.csv` (full successful production export, not the pilot or complete failure-inclusive export)
- `Streamlit/assets/linkedin_qr.svg` (closing page)

No comparison CSV, API key, model inference or new classification is required. Both model exports must contain unique review keys and the expected overall and aspect prediction columns. Duplicate keys produce a warning and retain the first row.

## Reporting methodology

The app inner joins on `record_key`, uses existing Hugging Face eligibility flags (Gemini fallback if absent), excludes explicit long-haul and first-class flags, retains unknown haul status, excludes dates before 2024-01-01 and invalid dates, and restricts reporting to the seven benchmark airlines. No duration threshold is recomputed. Existing classification discrepancies remain upstream. The EDA audit is global; page charts use sidebar filters. Unknown language is excluded from the language count and top-seven chart; other known languages are grouped as Other.

Overall sentiment shares use recognized positive, negative, neutral and mixed labels only. Unmentioned/unclassified sentiment is not a neutral label. Aspect shares require an explicit mention flag and recognized sentiment. The 19 experience aspects exclude six cabin, geography and eligibility fields from the original 25 classifications. The six theme groups are dashboard groupings. Fixed human-validation metrics are historical and are not recomputed by filters. Agreement is not accuracy. Priority scores are exploratory; missing evidence contributes zero. A single-year selection may prevent calculating the multi-year deterioration component. Compare both uses Gemini for single-model views unless otherwise indicated.

## Dashboard pages

Executive Overview; EDA & Model Validation; Overall Sentiment; Feature Analysis; Strategic Insights; Thank You.

## Verification

The app is syntax-checked and its data preparation is tested with synthetic fixtures. Actual ABSA CSVs and a live Streamlit runtime are required for end-to-end verification. The requirements are minimum versions, not a tested lockfile. Public review samples are not representative passenger surveys; 2026 is a partial year.
