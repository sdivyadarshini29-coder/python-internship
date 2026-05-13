import pandas as pd
import numpy as np

# Create dummy data
data = {
    'Years_Experience': np.random.randint(1, 20, 100),
    'Education_Level': np.random.randint(1, 4, 100), # 1: Bachelor, 2: Master, 3: PhD
    'Age': np.random.randint(22, 60, 100),
    'Salary': []
}

# Simple logic for salary calculation
for i in range(100):
    salary = (data['Years_Experience'][i] * 5000) + (data['Education_Level'][i] * 10000) + 20000
    data['Salary'].append(salary)

df = pd.DataFrame(data)
df.to_csv('employee_salary_dataset.csv', index=False)
print("CSV File Created Successfully!")