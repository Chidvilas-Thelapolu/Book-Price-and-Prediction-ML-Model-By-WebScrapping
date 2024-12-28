import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the data
data = pd.read_csv('books_data_with_ratings.csv')

# Clean and preprocess the data
data['Price'] = data['Price'].str.replace('£', '').astype(float)  # Remove £ and convert to float
data['Rating'] = data['Rating'].str.replace(' Stars', '').str.replace(' Star', '').astype(int)  # Convert Rating to numerical

# Features and target (simplify to only use Rating for now)
X = data[['Rating']]  # Features
y = data['Price']  # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear Regression Model
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Linear Regression Metrics
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Random Forest Regression
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Random Forest Metrics
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Print Metrics
print("\nLinear Regression Metrics:")
print(f"RMSE: {rmse_lr:.2f}, MAE: {mae_lr:.2f}, R²: {r2_lr:.2f}")

print("\nRandom Forest Regression Metrics:")
print(f"RMSE: {rmse_rf:.2f}, MAE: {mae_rf:.2f}, R²: {r2_rf:.2f}")

# Determine the better model
if r2_lr > r2_rf:
    print("\nLinear Regression performs better.")
else:
    print("\nRandom Forest Regression performs better.")

# Visualization: Actual vs Predicted for Random Forest
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.7, label='Random Forest Predictions', color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Actual vs Predicted (Random Forest)')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.legend()
plt.show()

# Visualization: Residual Distribution
plt.figure(figsize=(12, 6))
sns.histplot(y_test - y_pred_rf, kde=True, color='blue', label='Random Forest Residuals', bins=30)
plt.title('Residual Distribution (Random Forest)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.legend()
plt.show()
