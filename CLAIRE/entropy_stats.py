import pandas as pd

df = pd.read_csv("semantic_entropy_3d_new.csv")
print(df.describe())
# Group by Success and calculate descriptive statistics
stats = df.groupby("Success")["Entropy"].describe()
percentile_95 = df['Entropy'].quantile(0.95)
print(f"95% {percentile_95}")
print(stats)