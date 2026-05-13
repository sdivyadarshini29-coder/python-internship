import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
import pickle

# 1. Load Data
df = pd.read_csv('employee_salary_dataset.csv')

# --- DATA MODIFICATION FOR CLASSIFICATION ---
# Salary-ah moonu category-ah pirikkurom (Low, Medium, High)
def classify_salary(sal):
    if sal < 50000: return 'Low'
    elif sal < 90000: return 'Medium'
    else: return 'High'

df['Salary_Category'] = df['Salary'].apply(classify_salary)

# 2. Features and Targets
X = df[['Years_Experience', 'Education_Level', 'Age']]
y_reg = df['Salary']          # For Regression (Linear & Random Forest)
y_clf = df['Salary_Category']  # For Classification (Logistic)

# 3. Split Data
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)

# --- A. LINEAR REGRESSION ---
lr_model = LinearRegression()
lr_model.fit(X_train, y_train_reg)
lr_preds = lr_model.predict(X_test)
print(f"Linear Regression Error: {mean_squared_error(y_test_reg, lr_preds)}")

# --- B. LOGISTIC REGRESSION (Salary Category Prediction) ---
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_clf, y_train_clf)
log_preds = log_model.predict(X_test_clf)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test_clf, log_preds) * 100}%")

# --- C. RANDOM FOREST (Better Performance) ---
rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X_train, y_train_reg)
rf_preds = rf_model.predict(X_test)
print(f"Random Forest Error: {mean_squared_error(y_test_reg, rf_preds)}")

# --- VISUALIZATION (Graphs) ---
plt.figure(figsize=(12, 5))

# Graph 1: Actual vs Predicted (Random Forest)
plt.subplot(1, 2, 1)
plt.scatter(y_test_reg, rf_preds, color='blue')
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'k--', lw=2)
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Actual vs Predicted (Random Forest)')

# Graph 2: Salary Categories Distribution
plt.subplot(1, 2, 2)
sns.countplot(x='Salary_Category', data=df, palette='viridis')
plt.title('Salary Category Distribution')

plt.tight_layout()
plt.savefig('model_performance.png') # Graph-ah save pannum
plt.show()

# Save the best model (Random Forest) for App
with open('salary_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

print("\nAll Models Trained and Visualization Saved as 'model_performance.png'!")