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
| Encoding            | Ordinal Encoding (imposes artificial order)                | **One-Hot Encoding** for nominal variables; ordinal mapping for `Mes` and `Dias`                                                                           |
| Feature engineering | No new features created                                    | **5 derived ratio features** + log-transforms for skewed columns + quantile/equal-width bins for key numeric variables                                     |
| Feature selection   | Manual removal based on heatmap inspection                 | **4-step automated pipeline**: high-corr pair removal → Chi-Square → L1 SelectFromModel → F-Classifier                                                    |
| Data cleaning       | Aggressive row removal (~988 rows across multiple filters) | **No rows removed** - all data preserved                                                                                                                   |
| Model functions     | Inline code with results scattered                         | **Reusable functions** (`lr_model`, `rf_model`, `gb_model`, `xgb_model`) with standardized output                                                  |
| Results tracking    | Manual variable tracking                                   | **Dictionaries** (`results`, `results_balanced`) for systematic comparison                                                                             |
| EDA visualizations  | Subplots grouped in single figures                         | **Individual plots** per variable with `hue='y'` showing target distribution                                                                             |
| Final prediction    | Uses validation split model                                | **Retrains on full dataset** before predicting test set                                                                                                    |

### Feature Engineering Steps

1. **One-Hot Encoding** — nominal categorical variables (`Tipo_Trabajo`, `Estado_Civil`, `Educacion`, `Incumplimiento`, `Vivienda`, `Consumo`, `Contacto`, `Resultado_Anterior`) → 38 features
2. **Derived ratio features** — `price_conf_ratio`, `conf_nr_employed_ratio`, `price_employed_ratio`, `age_euribor_ratio`, `total_num_contacts` + original numeric columns → 48 features
3. **Ordinal encoding** — `Mes` (jan=1 … dec=12) and `Dias` (mon=1 … fri=5) → 51 features
4. **Log-transforms** — `log1p` applied to skewed columns (`Edad`, `Campana`, `Dias_Ultima_Camp`, `No_Contactos`, `cons_price_idx`) → 56 features
5. **Binning** — quantile and equal-width bins for `Edad`, `emp_var_rate`, `cons_conf_idx`, `nr_employed`, `cons_price_idx`, `euribor3m` → 67 features

### Feature/Variable Handling

| Aspect                      | Original                                            | Enhanced                                                                     |
| --------------------------- | --------------------------------------------------- | ---------------------------------------------------------------------------- |
| Variables removed pre-model | `Edad`, `Consumo`, `Vivienda` + row filtering | Automated 4-step selection pipeline (correlation, chi2, L1, F-Classifier)    |
| Final feature count         | 16 (ordinal-encoded)                                | **34** (after full feature engineering + selection pipeline)           |
| `Edad`                    | Dropped entirely                                    | Kept indirectly via `age_euribor_ratio`, log-transform, and age bins        |
| `Consumo`, `Vivienda`   | Dropped entirely                                    | One-hot encoded, then auto-dropped if low chi2 score                         |
| Rows used for training      | ~25,385 (after filtering)                           | **26,373** (all rows preserved)                                        |

### Results Comparison

| Model                          | Original (ROC-AUC) | Enhanced (ROC-AUC)  |
| ------------------------------ | ------------------ | ------------------- |
| Logistic Regression (balanced) | 0.717              | **0.746**     |
| Random Forest (balanced)       | 0.731              | **0.746**     |
| Gradient Boosting (balanced)   | 0.719              | 0.738               |
| XGBoost (balanced)             | 0.718              | 0.733               |
| **Best overall**         | RF: 0.731          | **RF ≈ LR: 0.746** |

### Key Takeaways

1. **One-Hot Encoding > Ordinal Encoding** for nominal variables — avoids imposing false ordinal relationships
2. **Richer feature engineering** (ratio features, log-transforms, binning) provided broader signal coverage, allowing all models to improve over the original
3. **Preserving all rows** instead of aggressive filtering retained more training signal
4. **Automated 4-step feature selection** (correlation → chi2 → L1 → F-Classifier) is more systematic and reproducible than manual inspection
5. The enhanced notebook achieves a **+1.5 pp improvement** in best ROC-AUC (0.746 vs 0.731); RF and LR are essentially tied at the top, with Logistic Regression chosen as the final model
