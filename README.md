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
- Dependencies in `requirements.txt`

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the notebooks
```bash
jupyter notebook
```
Run notebooks in order:
1. `01_data_audit.ipynb`
2. `02_cleaning_feature_engineering.ipynb`
3. `03_modeling_churn.ipynb`
4. `04_value_proxy_and_actions.ipynb`

## Makefile commands
```bash
make setup
make notebook
```

## Outputs
- `data/processed/transactions_clean.parquet`
- `data/processed/customer_features.parquet`
- `reports/figures/*.png`
- `reports/customer_action_list.csv`
