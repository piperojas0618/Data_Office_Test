# Davivienda Data Office Test - Bank Marketing Classification

Binary classification task predicting whether a client will subscribe to a term deposit (`y`) based on demographic, financial, and campaign-related features.

## Dataset

- **Train:** 26,373 records, 20 features + target (`y`)
- **Test:** 6,594 records
- **Class imbalance:** ~11% positive class (`y=1`)
- **No null values** in either dataset

## Pipeline Overview

### 1. EDA

- Null checks, data types, unique value counts, and descriptive statistics
- Boxplots for `Edad` and `euribor3m`; histograms for all numeric and categorical variables
- Outlier removal (age > 80, low-frequency categories in multiple variables)
- Pearson correlation for numeric variables; Chi-Square (p-value) for categorical variables

### 2. Feature Selection

- Removed `Edad` (low linear correlation with target)
- Removed `Consumo` and `Vivienda` (high p-value with target in Chi-Square test)
- Removed `Estado_Civil = 'unknown'` and `Incumplimiento = 'yes'` (rare categories)

### 3. Preprocessing

- **Categorical:** Ordinal Encoding for all categorical variables
- **Numerical:** StandardScaler normalization
- **Class balancing:** SMOTE (sampling_strategy=0.8)

### 4. Models Trained (with manual hyperparameter grid search)

| Model                              | Best ROC-AUC    |
| ---------------------------------- | --------------- |
| Logistic Regression (SMOTE)        | 0.717           |
| Random Forest (balanced_subsample) | **0.731** |
| Gradient Boosting (SMOTE)          | 0.719           |
| XGBoost (SMOTE)                    | 0.718           |

### 5. Final Model

**Random Forest** (n_estimators=25, max_depth=5, max_features='sqrt', class_weight='balanced_subsample') trained on original (non-SMOTE) data. Final validation: ROC-AUC = 0.731, Accuracy = 83%.

---

## Enhanced Notebook - Key Improvements

The enhancement was made primarily to demonstrate my improvement in organizing and writing code, as well as my ability to address a simple classification model with proper data and model handling. 

The [Enhanced_Test.ipynb](Enhanced_Test.ipynb) introduces several improvements over the [original notebook](Prueba%20Oficina%20de%20Datos%20-%20Davivienda.ipynb):

### Code Quality

| Aspect              | Original                                                   | Enhanced                                                                                                                                                         |
| ------------------- | ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Encoding            | Ordinal Encoding (imposes artificial order)                | **One-Hot Encoding** (no ordinal assumptions)                                                                                                              |
| Feature engineering | No new features created                                    | **5 derived features** (price_conf_ratio, conf_nr_employed_ratio, price_employed_ratio, age_euribor_ratio, total_num_contacts)                             |
| Feature selection   | Manual removal based on heatmap inspection                 | **Automated** removal of features with \|corr\| < 0.05 with target + removal of highly correlated pairs (> 0.9) keeping the one more correlated with `y` |
| Data cleaning       | Aggressive row removal (~988 rows across multiple filters) | **No rows removed** - all data preserved                                                                                                                   |
| Model functions     | Inline code with results scattered                         | **Reusable functions** (`lr_model`, `rf_model`, `gb_model`, `xgb_model`) with standardized output                                                  |
| Results tracking    | Manual variable tracking                                   | **Dictionaries** (`results`, `results_balanced`) for systematic comparison                                                                             |
| EDA visualizations  | Subplots grouped in single figures                         | **Individual plots** per variable with `hue='y'` showing target distribution                                                                             |
| Final prediction    | Uses validation split model                                | **Retrains on full dataset** before predicting test set                                                                                                    |

### Feature/Variable Handling

| Aspect                      | Original                                            | Enhanced                                                              |
| --------------------------- | --------------------------------------------------- | --------------------------------------------------------------------- |
| Variables removed pre-model | `Edad`, `Consumo`, `Vivienda` + row filtering | 38 low-correlation features + 5 high-correlation features (automated) |
| Final feature count         | 16 (ordinal-encoded)                                | **24** (one-hot + engineered + numeric)                         |
| `Edad`                    | Dropped entirely                                    | Kept indirectly via `age_euribor_ratio`                             |
| `Consumo`, `Vivienda`   | Dropped entirely                                    | One-hot encoded, then auto-dropped if low correlation                 |
| Rows used for training      | ~25,385 (after filtering)                           | **26,373** (all rows preserved)                                 |

### Results Comparison

| Model                          | Original (ROC-AUC) | Enhanced (ROC-AUC)  |
| ------------------------------ | ------------------ | ------------------- |
| Logistic Regression (balanced) | 0.717              | **0.747**     |
| Random Forest (balanced)       | 0.731              | **0.746**     |
| Gradient Boosting (balanced)   | 0.719              | 0.741               |
| XGBoost (balanced)             | 0.718              | **0.744**     |
| **Best overall**         | RF: 0.731          | **LR: 0.747** |

### Key Takeaways

1. **One-Hot Encoding > Ordinal Encoding** for nominal variables - avoids imposing false ordinal relationships
2. **Feature engineering** (ratio features) added predictive signal, enabling even Logistic Regression to outperform the original's best model
3. **Preserving all rows** instead of aggressive filtering retained more training signal
4. **Automated feature selection** based on correlation thresholds is more reproducible than manual inspection
5. The enhanced notebook achieves a **+1.6 pp improvement** in best ROC-AUC (0.747 vs 0.731) with a simpler final model (Logistic Regression vs Random Forest)
