import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("RBA_Final_dataset.csv",low_memory=False)

# Convert columns to numeric where possible
df_numeric = df.copy()
for col in df_numeric.columns:
    df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

# Drop columns that are all NaN (non-numeric)
df_numeric.dropna(axis=1, how='all', inplace=True)

# Check and plot
if df_numeric.shape[1] == 0:
    print("No numeric data available for correlation heatmap.")
else:
    corr = df_numeric.corr()

    plt.figure(figsize=(14, 12))  # Wider and taller for clarity
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5,
                annot_kws={"size": 8})
    
    # Rotate axis labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.title("Correlation Heatmap", fontsize=16)
    plt.tight_layout()
    plt.show()
