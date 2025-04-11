import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("RBA_Final_dataset.csv")

# Convert relevant columns to numeric
numeric_cols = ["Round-Trip Time [ms]", "Is Attack IP", "Is Account Takeover", "Login Successful"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df_clean = df.dropna(subset=numeric_cols)

# Set up a 2x2 subplot layout
plt.figure(figsize=(15, 12))

# 1️⃣ Scatter Plot
plt.subplot(2, 2, 1)
sns.scatterplot(data=df_clean, x="Round-Trip Time [ms]", y="Is Attack IP", hue="Is Attack IP", palette="coolwarm")
plt.title("Scatter: Round-Trip Time vs Is Attack IP")
plt.grid(True)

# 2️⃣ Box Plot
plt.subplot(2, 2, 2)
sns.boxplot(data=df_clean, x="Login Successful", y="Round-Trip Time [ms]", palette="Set2")
plt.title("Boxplot: Round-Trip Time by Login Success")
plt.grid(True)

# 3️⃣ Correlation Heatmap
plt.subplot(2, 2, 3)
corr = df_clean[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")

# 4️⃣ Pie Chart for Login Success
plt.subplot(2, 2, 4)
login_counts = df_clean["Login Successful"].value_counts()
labels = ['Success', 'Failed']
plt.pie(login_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=["#66bb6a", "#ef5350"])
plt.title("Login Success Distribution")

plt.tight_layout()
plt.show()
