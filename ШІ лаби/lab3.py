import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Завантаження даних
advertising = pd.read_csv('advertising.csv')

# Перевірка на наявність пропущених значень
advertising.dropna(inplace=True)

# Вибір числових ознак для прогнозування
X = advertising.drop('Sales', axis=1)  # 'Sales' — цільова змінна
y = advertising['Sales']  # 'Sales' — це те, що ми будемо прогнозувати

# Розділення на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Стандартизація даних (якщо потрібно)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Побудова та навчання моделі дерева рішень
decision_tree = DecisionTreeRegressor(random_state=42)
decision_tree.fit(X_train_scaled, y_train)

# Прогнозування на тестових даних
y_pred = decision_tree.predict(X_test_scaled)

# Обчислення середньоквадратичної помилки (MSE) та коефіцієнта детермінації (R^2)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Виведення результатів
print(f"Середньоквадратична помилка (MSE): {mse}")
print(f"Коефіцієнт детермінації (R^2): {r2}")

# Побудова графіку дерева рішень
plt.figure(figsize=(20, 10))
plot_tree(decision_tree, feature_names=X.columns, filled=True, rounded=True)
plt.show()
