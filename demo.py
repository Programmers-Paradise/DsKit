import pandas as pd
from dskit import dskit

# 1. Load Data
print("--- Loading Data ---")
kit = dskit.load("dummy_data.csv")
print(kit.df.head())

# 2. EDA
print("\n--- Quick EDA ---")
kit.quick_eda()

# 3. Cleaning
print("\n--- Cleaning Data ---")
# Chain methods: fix types -> rename columns -> fill missing -> remove outliers
kit.fix_dtypes().rename_columns_auto().fill_missing().remove_outliers()
print("Data after cleaning:")
print(kit.df.head())

# 4. Feature Engineering / Preprocessing
print("\n--- Preprocessing ---")
# Auto encode categorical variables and scale numeric ones
kit.auto_encode().auto_scale()
print("Data after preprocessing:")
print(kit.df.head())

# 5. Train/Test Split
print("\n--- Train/Test Split ---")
# Assuming 'salary' is the target for a regression task, or 'performance_score'
# Let's try to predict 'salary' (regression)
X_train, X_test, y_train, y_test = kit.train_test_auto(target='salary')
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# 6. Modeling
print("\n--- Training Model ---")
# Train a Random Forest Regressor
kit.train(model_name='random_forest', task='regression')

# 7. Evaluation
print("\n--- Evaluation ---")
kit.evaluate(task='regression')

# 8. Explainability
print("\n--- Explainability ---")
# This might pop up a plot window
# kit.explain() 
print("Explainability plot generated (if enabled).")

print("\n--- Done ---")
