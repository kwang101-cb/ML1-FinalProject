# Notebook Coding Best Practices (agent.md)

## Goals
- Make notebooks **reproducible**, **readable**, and **safe to rerun top-to-bottom**.
- Reduce hidden state and “it worked yesterday” issues.

## Structure
- Start with a short **Purpose** and **Outline** markdown cell.
- Use clear sections:
  - Data loading
  - Cleaning / feature engineering
  - EDA / plots
  - Modeling
  - Evaluation
  - Conclusions / next steps
- Keep each code cell focused on one logical step.

## Reproducibility & State
- Ensure the notebook runs **top-to-bottom** without manual tweaks.
- Avoid relying on variables created “somewhere above” without obvious dependencies.
- Prefer **functions** for reusable logic (feature creation, plotting, evaluation).
- Set random seeds where relevant (NumPy / sklearn).
- Print or log key shapes and assumptions (`df.shape`, column lists).

## Data Loading & Paths
- Use `Path` from `pathlib` and **relative paths** from project root when possible.
- Validate inputs early (file exists, required columns present).
- Never hardcode machine-specific absolute paths.

## Missing Data & Types
- Make dtype conversions explicit (`pd.to_datetime`, numeric coercion).
- Decide on a missing-data strategy explicitly (drop vs impute), and document it.
- For categorical features, avoid arbitrary integer encoding unless truly ordinal.

## Preprocessing & Leakage
- When predicting a target, **do not include target-derived features** in `X`.
- Use sklearn **Pipeline + ColumnTransformer** so preprocessing is fit only on train data.
- Keep feature engineering consistent between train/validation/test.

## Performance (Large Data)
- Prefer vectorized pandas operations over Python loops.
- Avoid printing huge objects; use `.head()`, `.describe()`, sampled views.
- Consider sampling for EDA; keep full-data operations minimal and deliberate.
- Cache expensive intermediate results (e.g., save cleaned parquet) when appropriate.

## Visualization
- Label axes/titles and units; include legends when needed.
- Keep plots readable (reasonable figure size, rotate ticks if needed).
- Don’t create dozens of near-duplicate plots—factor into a plotting function.

## Modeling & Evaluation
- Use a clear metric (e.g., RMSE/MAE/R²) and report it consistently.
- Use a validation split or cross-validation; avoid tuning on the test set.
- Track baselines (simple model first) before adding complexity.

## Notebook Hygiene
- Restart kernel + Run All before finalizing.
- Remove dead/duplicated cells and large accidental outputs.
- Keep secrets out of notebooks (API keys, tokens).
- If the notebook is a “report,” keep it narrative; if it’s “workbench,” keep it modular.

## Exportability
- Keep core logic in functions so it can be moved to a `.py` module later.
- Avoid notebook-only magic for critical steps (or provide a non-magic alternative).