import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the predictions file
df = pd.read_csv('predictions_vs_true.csv')

# Analyze true values
print("True Values Statistics:")
print("Min:", df['true_value'].min())
print("Max:", df['true_value'].max())
print("Mean:", df['true_value'].mean())
print("Median:", df['true_value'].median())
print("Standard deviation:", df['true_value'].std())

# Plot distribution of true values
plt.figure(figsize=(10, 5))
plt.hist(df['true_value'], bins=50)
plt.title('Distribution of Entanglement Negativity')
plt.xlabel('Value')
plt.ylabel('Count')
plt.savefig('negativity_distribution.png')
plt.show()
