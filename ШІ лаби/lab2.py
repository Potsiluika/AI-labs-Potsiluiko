import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Завантаження набору даних
advertising = pd.read_csv('advertising.csv')

# Видалення рядків з NaN
advertising.dropna(inplace=True)

# Вибір числових ознак для кластеризації
X = advertising.select_dtypes(include=['float64', 'int64'])

# Перевірка, чи є достатньо ознак для кластеризації
if X.shape[1] < 2:
    raise ValueError("Необхідно хоча б 2 числові ознаки для кластеризації")

# Стандартизація даних
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Використання алгоритму k-means з 3 кластерами
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# Додавання кластерних міток до датасету
advertising['Cluster'] = kmeans.labels_

# Виведення кількості зразків у кожному кластері
print(advertising['Cluster'].value_counts())

# Візуалізація кластерів на діаграмі розсіювання
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=advertising['Cluster'], palette='viridis')
plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])
plt.title('Кластеризація за допомогою K-Means')
plt.legend(title='Cluster')
plt.show()
