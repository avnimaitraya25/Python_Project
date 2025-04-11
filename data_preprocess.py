import pandas as pd

# Load the dataset
df = pd.read_csv("RBA_Final_dataset.csv")

# Display first 5 rows
print("\n📌 First 5 Rows:\n")
print(df.head())

# Info about the dataset
print("\n📌 DataFrame Info:\n")
print(df.info())

# Summary statistics
print("\n📌 Descriptive Statistics:\n")
print(df.describe(include='all'))

# Check missing values
print("\n📌 Missing Values Per Column:\n")
print(df.isna().sum())

# Unique values per column
print("\n📌 Unique Value Counts Per Column:\n")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")

# Print sample data from object columns
print("\n📌 Sample values from object (string) columns:\n")
for col in df.select_dtypes(include='object').columns:
    print(f"{col}: {df[col].unique()[:5]}")

# Optional: Fill missing numeric values with median
numeric_cols = df.select_dtypes(include='number').columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Optional: Drop rows with any missing values (uncomment if needed)
# df.dropna(inplace=True)

# Save the cleaned version (if you want)
# df.to_csv("cleaned_RBA_dataset.csv", index=False)
