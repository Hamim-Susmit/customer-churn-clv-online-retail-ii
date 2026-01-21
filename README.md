# Customer Churn Risk + CLV Proxy (Online Retail II)

## Business problem
Understanding which customers are at risk of churn **and** which customers are likely to generate meaningful value in the near future lets teams focus retention budgets on the right segments. This project turns transaction-level data into churn predictions and a **future value proxy** so business users can prioritize outreach and revenue recovery.

## Dataset summary + quirks
- Source: UCI "Online Retail II" Excel file (`data/raw/online_retail_II.xlsx`).
- **Cancellations:** invoices starting with `"C"` are cancellations.
- **Returns:** negative quantities represent returns.
- **Missing IDs:** rows can have missing `CustomerID`.
- Multiple sheets in the Excel file are concatenated with a `source_sheet` column for traceability.

## Approach overview
1. **Audit**: load and profile each sheet, standardize columns, inspect missingness.
2. **Clean**: drop missing `CustomerID`, remove non-positive prices, flag cancellations/returns.
3. **Features**: build customer-level RFM, monetary, diversity, and behavior metrics.
4. **Time-aware split**: train on pre-cutoff history, label churn based on post-cutoff activity.
5. **Model**: compare logistic regression vs. gradient boosting.
6. **Insights**: inspect feature importances.
7. **Action list**: combine churn risk with value proxy into recommended actions.

## Results summary (update after running notebooks)
> The metrics below are **placeholders**. Run the notebooks to generate final values and update this section.

- **Churn definition:** churned if no purchase in the 90-day window after the cutoff.
- **Final model:** HistGradientBoostingClassifier.
- **Metrics:** ROC-AUC = **TBD**, Precision = **TBD**, Recall = **TBD**, F1 = **TBD**.
- **Top 5 drivers (to confirm via permutation importance):**
  1. Recency (days since last purchase)
  2. Frequency (invoice count)
  3. Revenue per month
  4. Average order value
  5. Return/cancellation rates

### Business recommendations
1. **High risk + high value:** prioritize personal outreach and targeted incentives to reduce churn risk.
2. **High risk + low value:** run low-cost reactivation campaigns (email/ads) to test uplift.
3. **Low risk + high value:** focus on loyalty/upsell programs to increase share-of-wallet.

### Limitations + next steps
- This project uses a **value proxy** (future 90-day revenue), not a full CLV model.
- The dataset is historical and may not reflect current seasonality or promotions.
- Next steps: productionize with scheduled data refreshes, add behavioral/web features, and A/B test retention offers.

## Project structure
```
├── data/
│   ├── raw/online_retail_II.xlsx
│   └── processed/
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_cleaning_feature_engineering.ipynb
│   ├── 03_modeling_churn.ipynb
│   └── 04_value_proxy_and_actions.ipynb
├── reports/
│   ├── figures/
│   └── customer_action_list.csv
├── src/
└── requirements.txt
```

## Reproducibility
- Python 3.11+
- Deterministic seed (`RANDOM_STATE = 42`)
- Dependencies in `requirements.txt` or `pyproject.toml`

## Setup
Install dependencies in a virtual environment:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies and project in editable mode
pip install -r requirements.txt
pip install -e .
```

## Run the full pipeline

### Option 1: Run all notebooks in sequence (Interactive)
```bash
source .venv/bin/activate  # or activate.bat on Windows
jupyter notebook
```
Then open and run each notebook in order:
1. `01_data_audit.ipynb` — Load raw Excel data, profile columns, check for missing values and anomalies
2. `02_cleaning_feature_engineering.ipynb` — Clean transactions (remove invalid rows), build customer-level features (RFM, diversity, behavior metrics)
3. `03_modeling_churn.ipynb` — Train churn prediction models (Logistic Regression vs HistGradientBoosting), evaluate performance
4. `04_value_proxy_and_actions.ipynb` — Train value regression model, generate churn + value scores, create action list

### Option 2: Run the final step only (Script)
If earlier notebooks have already generated `data/processed/transactions_clean.parquet` and `data/processed/customer_features.parquet`:

```bash
cd "/home/dhaka/Main Project/customer-churn-clv-online-retail-ii"
source .venv/bin/activate
python scripts/run_04.py
```

This trains churn and value models with proper train/validation split and writes the action list to `reports/customer_action_list.csv`.

The script will output:
- **Churn model validation AUC** — Evaluated on holdout customers
- **Value model validation RMSE** — Evaluated on holdout customers

## What we're finding

**Data Quality & Leakage Prevention (Fixed):**
- **Time-based leakage eliminated:** Features are now computed only from transactions up to a snapshot date, preventing future behavior from leaking into training labels.
- **Train/validation split implemented:** Models are trained on earlier customer cohorts and evaluated on later ones to assess true generalization.
- **Unused variables cleaned:** Removed redundant `col_candidates` variable in data ingestion.

**Churn Risk Prediction:**
- Identifies customers likely to become inactive (no purchases for 90 days post-cutoff).
- Uses features: recency (days since last purchase), frequency (invoice count), monetary value, purchase diversity, return rates, etc.
- Model: HistGradientBoostingClassifier trained on historical purchase patterns.
- Output: `churn_probability` (0–1 score, higher = more likely to churn).

**Value Proxy (Future Revenue):**
- Estimates how much revenue each customer will generate in the next 90 days.
- Uses the same customer features to predict future purchase behavior.
- Model: Ridge regression.
- Output: `value_score` (predicted future revenue); `future_revenue_90d` (actual 90-day revenue sum from holdout period, if labeled).

**Action List:**
Segments customers by risk and value and recommends actions:
- **High risk + High value:** Retention incentive + personal outreach (expensive, targeted)
- **High risk + Low value:** Low-cost reactivation email (test if campaigns can recover them)
- **Low risk + High value:** Loyalty/upsell offer (maximize share-of-wallet)
- **Low risk + Low value:** Nurture / standard campaigns (maintain baseline engagement)

## Outputs

After running the pipeline, you will have:

- **`data/processed/transactions_clean.parquet`** — Cleaned transaction-level data with flags for cancellations, returns, and line revenue.
- **`data/processed/customer_features.parquet`** — Customer-level aggregated features (RFM, behavioral, diversity metrics, snapshot date).
- **`reports/customer_action_list.csv`** — Final output with columns:
  - `CustomerID`
  - `churn_probability` — Risk of churn (0–1)
  - `value_score` — Predicted future value
  - `future_revenue_90d` — Actual future revenue (from holdout period)
  - `segment` — One of {High risk + High value, High risk + Low value, Low risk + High value, Low risk + Low value}
  - `recommended_action` — Business action to take

Use this CSV to prioritize retention budgets and design targeted outreach campaigns.
