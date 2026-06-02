import numpy as np
import pandas as pd

df = pd.read_csv('data/GiveMeSomeCredit-training.csv')

# Basic checks
print("Dataset Shape:")
print(df.shape)

print("\nTarget Distribution:")
print(df.iloc[:, 0].value_counts(normalize=True))

print("\nMissing Values:")
print(df.isnull().sum())


# DebtRatio Analysis
col = 'DebtRatio'
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1
upper_fence = Q3 + 1.5 * IQR


print("\n--- DebtRatio Outlier Analysis ---")
print(f"Q1: {Q1}")
print(f"Q3: {Q3}")
print(f"IQR: {IQR}")
print(f"Upper Fence: {upper_fence}")
print(f"Values above fence: {(df[col] > upper_fence).sum()}")
print(f"Max value: {df[col].max()}")

# Percentiles
print(f"95th percentile: {df['DebtRatio'].quantile(0.95)}")
print(f"99th percentile: {df['DebtRatio'].quantile(0.99)}")
print(f"Values above 99th: {(df['DebtRatio'] > df['DebtRatio'].quantile(0.99)).sum()}")

# Summary stats
print("\nDebtRatio Summary:")
print(df['DebtRatio'].describe())
print()
for p in [0.90, 0.95, 0.97, 0.99]:
    val = df['DebtRatio'].quantile(p)
    count = (df['DebtRatio'] > val).sum()
    print(f"{p*100}th percentile: {val:.2f} | rows above: {count}")

# RevolvingUtilization Analysis
print("\n--- RevolvingUtilization Analysis ---")
print(df['RevolvingUtilizationOfUnsecuredLines'].describe())
print(f"Values above 1: {(df['RevolvingUtilizationOfUnsecuredLines'] > 1).sum()}")
print(f"Max value: {df['RevolvingUtilizationOfUnsecuredLines'].max()}")
