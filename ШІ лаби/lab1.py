import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Завантаження набору даних
advertising = pd.read_csv('advertising.csv')

# Попередній аналіз даних
print("Перші 5 рядків датасету:")
print(advertising.head())
print("\nІнформація про датасет:")
print(advertising.info())

# Очищення даних: видалення пропущених значень
advertising.dropna(inplace=True)

# Визначення кореляційної матриці
advertising_correlation = advertising.corr()

# Візуалізація кореляційної матриці
plt.figure(figsize=(12, 10))
heatmap = sns.heatmap(advertising_correlation, annot=True, cmap='coolwarm', fmt=".2f")
heatmap.set_title('Матриця кореляції ознак Advertising Dataset', fontsize=16)
plt.show()
