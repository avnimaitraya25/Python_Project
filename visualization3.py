import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("RBA_Final_dataset.csv")

# Convert columns to numeric where possible
df_numeric = df.copy()
for col in df_numeric.columns:
    df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

# Drop non-numeric and NaN columns
df_numeric = df_numeric.dropna(axis=1, how='all')

# Melt the dataframe to long format for seaborn boxplot
df_melted = df_numeric.melt(var_name="Feature", value_name="Value")

# Plot the boxplots
plt.figure(figsize=(14, 8))
sns.boxplot(x="Feature", y="Value", data=df_melted, palette="Set3")

plt.xticks(rotation=45, ha='right')
plt.title("Boxplot of Numeric Features", fontsize=16)
plt.tight_layout()
plt.show()
