# Ola Driver Churn Prediction

Predicting which Ola drivers will leave the platform, using monthly driver records (Jan 2019 to Dec 2020). Four tree-based models (Random Forest, Gradient Boosting, XGBoost, LightGBM) are tuned and compared inside leakage-safe pipelines.

## Problem

Driver attrition is expensive: every driver who leaves has to be replaced and onboarded. The goal is to classify each driver as **churned (1)** or **retained (0)** and find which factors are linked to leaving.

## Dataset

- 19,104 monthly records for 2,381 unique drivers (`ola.csv`)
- Columns: Driver_ID, Age, Gender, City, Education_Level, Income, Dateofjoining, LastWorkingDate, Joining Designation, Grade, Total Business Value, Quarterly Rating
- A driver is labelled **churned** if `LastWorkingDate` is present. About **68% of drivers (1,616) are churned** and 765 are retained, so churn is the *majority* class here.
- The dataset is not included in this repo. Add the source link here before publishing.

## Approach

1. **Cleaning:** parsed dates, dropped the index column and `City`, filled missing `Gender` with the mode.
2. **Driver-level aggregation:** collapsed monthly rows into one row per driver (2,381 rows, 15 features), for example:
   - `Income_Mean`, `Income_Latest`, `Grade_Change`
   - `Total_Business_Value`, `Avg_Monthly_Business_Value`
   - `Initial_Rating`, `Latest_Rating`, `Rating_Difference`
   - `Months_Active` (number of monthly records)
3. **Split first:** 80/20 stratified train/test split (1,904 / 477) before any preprocessing. The test set has 324 churned and 153 retained drivers.
4. **Pipeline per model:** `StandardScaler`, then `SMOTE`, then the classifier, inside `imblearn.pipeline.Pipeline`. Scaling and oversampling are fitted only on the training part of each CV fold, so validation folds stay untouched (no leakage). SMOTE comes after scaling because it relies on nearest-neighbour distances; the tree models themselves do not need scaling.
5. **Tuning:** `GridSearchCV`, 5-fold CV, scored on ROC-AUC.

   | Model | Combinations | Total fits (x5 folds) |
   |---|---|---|
   | Random Forest | 108 | 540 |
   | Gradient Boosting | 54 | 270 |
   | XGBoost | 32 | 160 |
   | LightGBM | 48 | 240 |

6. **Final evaluation:** the fitted `GridSearchCV` object (already refit on the full training set with the best parameters) is evaluated once on the untouched test set, using raw `X_test`.

## Results

| Model | CV ROC-AUC | Test ROC-AUC | Test accuracy | Churn F1 | No-churn F1 |
|---|---|---|---|---|---|
| Random Forest | 0.8665 | 0.8997 | 0.851 | 0.89 | 0.77 |
| Gradient Boosting | **0.8702** | **0.9062** | 0.855 | 0.89 | 0.78 |
| XGBoost | 0.8677 | 0.9037 | 0.853 | 0.89 | 0.77 |
| LightGBM | 0.8695 | 0.9030 | 0.855 | 0.89 | 0.77 |

Models are ranked by CV ROC-AUC (the test set is kept for the final report only). Gradient Boosting is marginally first, but **all four models are practically tied**: CV AUC spans only 0.004 and test AUC only 0.011. On a 477-driver test set, differences this small are within noise.

Best parameters:

- **Random Forest:** `n_estimators=120, max_depth=8, min_samples_leaf=3, max_features=0.5, class_weight=None`
- **Gradient Boosting:** `n_estimators=100, max_depth=3, min_samples_leaf=5, max_features=0.5`
- **XGBoost:** `n_estimators=200, max_depth=3, learning_rate=0.05, subsample=1.0, colsample_bytree=1.0`
- **LightGBM:** `n_estimators=200, num_leaves=7, learning_rate=0.05, min_child_samples=10` (see the `subsample` note under Limitations)

Random Forest confusion matrix on the test set (rows = actual, columns = predicted):

| | Pred: No churn | Pred: Churn |
|---|---|---|
| **Actual: No churn** | 119 | 34 |
| **Actual: Churn** | 37 | 287 |

## Key Findings

**About drivers and churn**

- **About 68% of drivers have left**, so attrition is the norm in this dataset, not the exception.
- **Rating is the strongest signal.** In the Random Forest, `Latest_Rating` (0.25) and `Rating_Difference` (0.11) are among the top features. How a driver is rated, and whether the rating moved over time, matters more than age, gender or education.
- **Tenure matters.** `Months_Active` is the second most important feature (0.22). Short-tenure drivers look very different from long-tenure ones (see the leakage note under Limitations).
- **Churned drivers earn less.** Average monthly income is about 59k for churned drivers vs about 73k for retained drivers (roughly 19% lower).
- **Churned drivers bring in far less business.** Retained drivers generated about 7.36B in total business value across 765 drivers, versus 3.56B across 1,616 churned drivers. That is roughly 4x less per driver.
- **Income and joining designation do not change within a driver's history** in this data (the income and designation differences computed per driver are 0), so income growth carries no information here. Demographics (age, gender, education) rank low in importance.

**About the models**

- All four models predict churn well: about **0.85 test accuracy**, **0.87 CV ROC-AUC** and **0.90 test ROC-AUC**. Churn F1 is 0.89 and no-churn F1 is 0.77 to 0.78 for every model.
- **Retained drivers are harder to identify than churned ones** (F1 0.77 vs 0.89), which is expected because they are the smaller class.
- **Model choice barely matters here.** Random Forest, Gradient Boosting, XGBoost and LightGBM end up within noise of each other, which suggests the limit is the information in the features, not the algorithm.
- **The tuned models are all on the simple side.** Every grid picked a shallow or small setting (Gradient Boosting `max_depth=3`, XGBoost `max_depth=3`, LightGBM `num_leaves=7`, Random Forest `max_depth=8`, the smallest allowed in its grid), which fits a small dataset of about 1,900 training rows.
- **Preprocessing had to sit inside the pipeline.** Scaling and SMOTE were placed inside each CV fold so validation folds stay untouched. Doing them before cross-validation would have produced an optimistic score.

## Recommendations

**For the business** (these are hypotheses from correlations, not proven causes; test them before acting)

1. **Watch ratings as an early-warning signal.** Drivers whose quarterly rating is low or falling are the group the models rely on most. Flag them for a check-in, coaching or support before they leave.
2. **Focus retention effort on the first months.** Since tenure is a top feature and attrition is high, a structured onboarding and early-tenure support programme is the most likely place to improve retention.
3. **Review earnings for at-risk drivers.** Churned drivers earn noticeably less and generate much less business. Incentives or better ride allocation for low-earning drivers are worth piloting, ideally as an A/B test.
4. **Do not rely on demographics for targeting.** Age, gender and education add little predictive value here, so retention programmes should be based on performance and earnings, not on who the driver is.
5. **Use the model to rank, not to decide.** Use predicted churn probabilities to prioritise outreach, and measure whether the outreach actually reduces churn.

**For the model (next steps)**

1. **Rebuild features with a cut-off date.** Use only data before a chosen date to predict who leaves afterwards. This removes the leakage risk and gives a realistic estimate of future performance.
2. **Compare against a baseline without SMOTE** (or with `class_weight`) to check whether SMOTE actually helps.
3. **Use repeated or nested cross-validation** instead of a single 80/20 split, since the test set is small.
4. **Extend the grids toward simpler models** (for example `max_depth=2` or smaller `num_leaves`), since the best values sit at the edge of several grids.
5. **Calibrate predicted probabilities** before using them to prioritise drivers.

## Limitations

- **Possible target leakage.** `Months_Active` and `Latest_Rating` are computed over the whole history, and drivers who leave have shorter records. Part of the models' accuracy may come from recognising "already left" rather than predicting a future exit.
- **Class balance.** Churn is the majority class (about 68%), so SMOTE oversamples the *retained* drivers. A baseline without SMOTE was not compared.
- **Small test set.** 477 drivers and one split. Differences of one or two points between models are within noise. Test ROC-AUC (about 0.90) is higher than CV ROC-AUC (about 0.87) for every model; since all models share the same test set, this likely reflects that particular split rather than a real advantage, and CV is the safer estimate.
- **Grid edges.** Several best parameters sit at the edge of their grid (see Recommendations), so a wider search might find slightly different settings.
- **LightGBM `subsample` had no effect.** In LightGBM, `subsample` only takes effect when `subsample_freq` is above 0, which was not set. That grid dimension just duplicated fits, so its reported best value is not meaningful.
- **Row-level averages.** The average-income and business-value comparisons are computed over monthly rows, not per driver.

## Project structure

```
ola_churn_prediction.ipynb   # full analysis and models
requirements.txt
README.md
```

## How to run

```bash
pip install -r requirements.txt
jupyter notebook ola_churn_prediction.ipynb
```

Place `ola.csv` in the same folder as the notebook.

## Tech stack

Python, pandas, NumPy, scikit-learn, imbalanced-learn, XGBoost, LightGBM, seaborn, matplotlib
# Ola_Driver_Churn_Prediction

