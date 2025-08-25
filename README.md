# 📊Data Science Job Dataset – Missing Value Imputation
This project explores the Data Science Job dataset (data_science_job.csv) and demonstrates how to clean and preprocess missing values using pandas and scikit-learn.
The goal is to build a reliable preprocessing pipeline that improves data quality for downstream tasks such as machine learning modeling and job market analytics.

# 📖 Project Overview

Real-world datasets often contain missing values in critical fields such as:
Demographics (e.g., gender, education level, major discipline)
Professional details (e.g., company size, company type, relevant experience)
Training-related features (e.g., training hours, city_development_index)
This project demonstrates mean and median imputation strategies to handle missing values effectively.

# 🚀 Workflow
Load the dataset
import pandas as pd
df = pd.read_csv("data_science_job.csv")

Explore missing values
df.isnull().sum()

Apply imputers
Mean Imputation → for normally distributed numeric features (e.g., training hours).
Median Imputation → for skewed numeric features or those with outliers (e.g., city_development_index).

Validate results
Ensure no missing values remain and distributions remain consistent.

# 🛠️ Implementation
🔹 Pandas Example
# Mean imputation
df["training_hours"] = df["training_hours"].fillna(df["training_hours"].mean())

 Median imputation
df["city_development_index"] = df["city_development_index"].fillna(df["city_development_index].median())

🔹 Scikit-learn Example
from sklearn.impute import SimpleImputer
 Mean Imputer
mean_imputer = SimpleImputer(strategy="mean")
df["training_hours_imputed"] = mean_imputer.fit_transform(df[["training_hours"]])

 Median Imputer
median_imputer = SimpleImputer(strategy="median")
df["city_development_index"] = median_imputer.fit_transform(df[["city_development_index"]])

📊 When to Use What?

✅ Mean Imputation → Good for continuous data with no extreme outliers.

✅ Median Imputation → Robust against skewed distributions and outliers.
