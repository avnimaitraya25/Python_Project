import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("RBA_Final_dataset.csv", low_memory=False)

# General setup
plt.style.use("dark_background")
sns.set_palette("cool")

# Pie Chart: Login Success vs Failure
def plot_login_success_pie():
    login_counts = df['Login Successful'].value_counts()
    labels = login_counts.index.astype(str)  # Ensure labels match the data
    counts = login_counts.values

    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Login Success Distribution')
    plt.show()


