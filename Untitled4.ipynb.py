#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[4]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Loading the dataset
file_path = 'D:/Student Prediction/studentper.csv'
df = pd.read_csv(file_path)

# Display a sample of the data
print(df.sample(10))

# Selecting features
features = ['failures', 'Medu', 'studytime', 'absences', 'G1', 'G2']
target = 'G3'

# Splitting the data
x_train, x_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.3, random_state=0)

# Scaling the dataset
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Fitting the base linear regression model
base_lr_model = LinearRegression()
base_lr_model.fit(x_train_scaled, y_train)

# BaggingRegressor
bagging_model = BaggingRegressor(base_lr_model, n_estimators=10, random_state=42)
bagging_model.fit(x_train_scaled, y_train)

# Evaluate models
def evaluate_model(model, x_test, y_test, model_name):
    y_test_pred = model.predict(x_test)
    rms_error = mean_squared_error(y_test, y_test_pred, squared=False)
    r2_score_value = r2_score(y_test, y_test_pred)
    print(f"{model_name}:")
    print(f"Root mean squared error: {rms_error}")
    print(f"R2-score: {r2_score_value}")
    return r2_score_value

# Calculate R2 scores
r2_score_base = evaluate_model(base_lr_model, x_test_scaled, y_test, "Base Linear Regression Model")
r2_score_bagging = evaluate_model(bagging_model, x_test_scaled, y_test, "Bagging Model")

# Select the best model based on R2 score
best_model = base_lr_model if r2_score_base > r2_score_bagging else bagging_model

# Make predictions on the test set using the best model
y_test_pred_best = best_model.predict(x_test_scaled)

# Evaluate the best model
rms_error_best = mean_squared_error(y_test, y_test_pred_best, squared=False)
r2_score_value_best = r2_score(y_test, y_test_pred_best)

print("Best Model Performance:")
print(f"Root mean squared error: {rms_error_best}")
print(f"R2-score: {r2_score_value_best}")

# Calculate the absolute difference between actual and predicted values
absolute_difference = np.abs(y_test - y_test_pred_best)

# Displaying the plots with a secondary y-axis for the absolute difference
f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

# Plotting actual vs predicted values
ax1.plot(np.arange(len(y_test)), y_test, label='Actual')
ax1.plot(np.arange(len(y_test_pred_best)), y_test_pred_best, label='Prediction')
ax1.set_title('Actual vs Predicted Values')
ax1.legend()

# Bar plot for the absolute difference with a secondary y-axis
ax2.bar(np.arange(len(absolute_difference)), absolute_difference, color='red', alpha=0.7)
ax2.set_title('Absolute Difference between Actual and Predicted Values')

# Plotting actual vs predicted values in a single plot with a secondary y-axis for the absolute difference
ax3.plot(np.arange(len(y_test)), y_test, label='Actual')
ax3.plot(np.arange(len(y_test_pred_best)), y_test_pred_best, label='Prediction')
ax3.set_ylabel('Actual/Predicted Values')
ax3_twin = ax3.twinx()
ax3_twin.plot(np.arange(len(absolute_difference)), absolute_difference, label='Absolute Difference', linestyle='--', color='red')
ax3_twin.set_ylabel('Absolute Difference')
ax3.set_title('Actual vs Predicted Values with Absolute Difference')
ax3.legend(loc='upper left')
ax3_twin.legend(loc='upper right')

# Adjust x-axis limits for better visibility in all subplots
x_range = 50  # Adjust this value based on your preference
for ax in [ax1, ax2, ax3]:
    ax.set_xlim(0, x_range)

# Displaying the plots
plt.show()


# In[ ]:




