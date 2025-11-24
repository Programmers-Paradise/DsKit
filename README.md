# ✅ DSKit – A Unified Wrapper Library for Data Science & ML 

**DSKit** is a community-driven, open-source Python library that wraps complex Data Science and ML operations into easy, user-friendly 1-line commands.

Instead of writing hundreds of lines for cleaning, EDA, plotting, preprocessing, modeling, evaluation, and explainability, DSKit makes everything **simple**, **readable**, **reusable**, and **production-friendly**.

The goal is to bring an **end-to-end Data Science ecosystem** in one place with wrapper-style functions and classes.

---

## 🚀 Project Objective

To create a Python library that lets users perform full Data Science workflows with minimal code:

```python
from dskit import DSKit

kit = DSKit(df)
kit.clean().visualize().train().explain()
```

The library should remain:
- ✅ Simple
- ✅ Extensible
- ✅ Modular
- ✅ Beginner-friendly
- ✅ Powerful for experts

---

## 📦 Core Modules

DSKit includes modules for:

- 📂 Data loading
- 🧹 Data cleaning
- 📊 EDA
- 🔍 Missing value handling
- 📉 Outlier handling
- 📈 Visualization & plotting
- 📐 Statistical utilities
- ⚙️ Preprocessing
- 🔧 Feature engineering
- 🤖 ML modeling
- 📋 Model evaluation
- 🧠 Explainability (XAI)
- 🔤 Optional NLP/CV tools

---

## 🧩 TASK LIST (Complete & Clear)

Each task below is numbered and written in simple language with enough theory so that any contributor — even new ones — can understand exactly what to build.

---

### **Task 1 — Implement `load()` function**

**Goal:** Build a function that loads different file formats (CSV, Excel, JSON) into a pandas DataFrame.

**Theory:** Beginners often struggle with loading different file types. This function should automatically detect file type from the extension and load it properly. If the file path is wrong or unreadable, the function should show a clean error message instead of a confusing traceback.

**Expected:**
```python
df = dskit.load("data.csv")
```

---

### **Task 2 — Implement `read_folder()`**

**Goal:** Load multiple data files from a folder and combine them into a single DataFrame.

**Theory:** Many datasets come split into multiple files. Users should not manually loop through files. The function should: list files → load each → concatenate. Provide the option to filter by file type.

**Expected:**
```python
df = dskit.read_folder("datasets/")
```

---

### **Task 3 — Implement `fix_dtypes()`**

**Goal:** Auto-detect correct column types (numeric, categorical, datetime) and convert them.

**Theory:** Data often comes with wrong types (e.g., numbers stored as text). This function should detect patterns (like digits) and convert columns accordingly. It should also convert date-like strings into datetime.

**Expected:**
```python
df = dskit.fix_dtypes(df)
```

---

### **Task 4 — Implement `rename_columns_auto()`**

**Goal:** Clean column names so they're analysis-friendly.

**Theory:** Raw datasets may have spaces, special characters, uppercase letters, etc. The function should convert them to lowercase, replace spaces with underscores, and remove problematic characters.

**Expected:**
```python
df = dskit.rename_columns_auto(df)
```

---

### **Task 5 — Implement `replace_specials()`**

**Goal:** Remove or replace unwanted special characters from text columns.

**Theory:** Text fields may contain symbols like @, #, %, $ that interfere with processing. The function should let users choose whether to remove or replace them, and should not modify numeric fields.

**Expected:**
```python
df = dskit.replace_specials(df)
```

---

### **Task 6 — Implement `missing_summary()`**

**Goal:** Provide a table that shows how many missing values each column has.

**Theory:** Missing data is critical to understand before cleaning. The function should show both the count and percentage of missing values. Optional sorting should be available.

**Expected:**
```python
dskit.missing_summary(df)
```

---

### **Task 7 — Implement `plot_missingness()`**

**Goal:** Visualize missing data patterns in a dataset.

**Theory:** Heatmaps and bar charts help identify systematic missingness. The visualization should clearly show where data is missing and which columns are most affected.

**Expected:**
```python
dskit.plot_missingness(df)
```

---

### **Task 8 — Implement `fill_missing()`**

**Goal:** Provide simple missing value handling with strategies like mean, median, mode, and forward fill.

**Theory:** Imputation is a common preprocessing step. This function should automatically decide whether to use numeric or categorical strategies based on column type.

**Expected:**
```python
df = dskit.fill_missing(df, strategy="mean")
```

---

### **Task 9 — Implement `outlier_summary()`**

**Goal:** Detect outliers in numeric columns using IQR or Z-score.

**Theory:** Outliers can bias models. This function should calculate outliers per column and return a summary table indicating how many extreme values exist.

**Expected:**
```python
dskit.outlier_summary(df)
```

---

### **Task 10 — Implement `remove_outliers()`**

**Goal:** Remove rows containing extreme values based on statistical rules.

**Theory:** After detecting outliers, users may want to remove them. Use rules like IQR or Z-score thresholds to identify which rows to drop.

**Expected:**
```python
df = dskit.remove_outliers(df)
```

---

### **Task 11 — Implement `plot_histograms()`**

**Goal:** Automatically generate histograms for all numeric features.

**Theory:** Histograms help understand data distribution. The function should loop through numeric columns and plot each histogram clearly with labels and titles.

**Expected:**
```python
dskit.plot_histograms(df)
```

---

### **Task 12 — Implement `plot_boxplots()`**

**Goal:** Create boxplots for numeric columns to show spread and outliers.

**Theory:** Boxplots visually highlight skewness and outliers. The function should generate boxplots for each numeric column in a structured layout.

**Expected:**
```python
dskit.plot_boxplots(df)
```

---

### **Task 13 — Implement `plot_correlation_heatmap()`**

**Goal:** Show correlation between numeric variables.

**Theory:** Correlation helps understand relationships in data. The heatmap should include labels, a colorbar, and an option to hide the upper triangle.

**Expected:**
```python
dskit.plot_correlation_heatmap(df)
```

---

### **Task 14 — Implement `plot_pairplot()`**

**Goal:** Generate a pairplot for multi-feature relationships.

**Theory:** Pairplots visually show pairwise relationships between features and are essential for understanding interactions.

**Expected:**
```python
dskit.plot_pairplot(df)
```

---

### **Task 15 — Implement `basic_stats()`**

**Goal:** Create a summary of essential statistics.

**Theory:** Provide count, mean, median, mode, std deviation, variance, min, max, IQR, etc. This helps users quickly understand dataset characteristics.

**Expected:**
```python
dskit.basic_stats(df)
```

---

### **Task 16 — Implement `auto_encode()`**

**Goal:** Automatically encode categorical variables.

**Theory:** ML models require numeric data. The function should detect categorical columns and choose label-encoding or one-hot encoding depending on the number of unique categories.

**Expected:**
```python
df = dskit.auto_encode(df)
```

---

### **Task 17 — Implement `auto_scale()`**

**Goal:** Automatically scale numeric data using StandardScaler or MinMaxScaler.

**Theory:** Scaling helps models like SVM, KNN, and Logistic Regression work better. Detect numeric columns and apply scaling only where needed.

**Expected:**
```python
df = dskit.auto_scale(df)
```

---

### **Task 18 — Implement `train_test_auto()`**

**Goal:** Automatically split data into training and testing sets.

**Theory:** Many users struggle with selecting correct columns. The function should detect the target feature automatically (if specified) and return split data.

**Expected:**
```python
X_train, X_test, y_train, y_test = dskit.train_test_auto(df, target="label")
```

---

### **Task 19 — Implement `QuickModel` class**

**Goal:** Create a class that trains ML models with simple commands.

**Theory:** Model training should be easy. Users should specify the model name, and the class should handle initialization, training, prediction, and evaluation.

**Expected:**
```python
model = QuickModel("random_forest").fit(X, y)
```

---

### **Task 20 — Implement `compare_models()`**

**Goal:** Train multiple ML models and compare their performance.

**Theory:** Model selection is difficult for beginners. This function should train a set of common models and generate a performance leaderboard.

**Expected:**
```python
dskit.compare_models(X, y)
```

---

### **Task 21 — Implement `auto_hpo()`**

**Goal:** Perform automatic hyperparameter tuning.

**Theory:** Hyperparameter tuning improves model performance but is often complex. Support grid search and random search with clean output.

**Expected:**
```python
best_model = dskit.auto_hpo(model, param_grid)
```

---

### **Task 22 — Implement `evaluate_model()`**

**Goal:** Provide evaluation metrics for a trained model.

**Theory:** Include accuracy, precision, recall, F1, ROC-AUC for classification and RMSE, MAE, R² for regression.

**Expected:**
```python
dskit.evaluate_model(model, X_test, y_test)
```

---

### **Task 23 — Implement `error_analysis()`**

**Goal:** Analyze wrong predictions and show diagnostic insights.

**Theory:** Helps understand model weaknesses by identifying misclassified points or high-error points.

**Expected:**
```python
dskit.error_analysis(model, X_test, y_test)
```

---

### **Task 24 — Implement `explain_shap()`**

**Goal:** Provide SHAP-based explainability for ML models.

**Theory:** SHAP shows feature contributions. This function should compute SHAP values and generate summary/force plots.

**Expected:**
```python
dskit.explain_shap(model, X)
```

---

### **Task 25 — Implement `quick_eda()`**

**Goal:** Generate a complete EDA report with one function.

**Theory:** Should include missingness, statistics, outliers, correlations, and plots in a structured format.

**Expected:**
```python
dskit.quick_eda(df)
```

---



## 📄 License

This project is licensed under the MIT License.

