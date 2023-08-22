import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the dataset from the CSV file
df = pd.read_csv("Admission_Predict.csv")
df.head(10).T

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Assume 'GRE', 'TOEFL', 'University_Rating', 'CGPA' are features and 'Chance of Admit' is the target column
X = df[["GRE Score", "TOEFL Score", "University Rating", "CGPA"]]
y = df["Chance of Admit "]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the target values on the test set
y_pred = model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error:", mse)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Predict the target values on the test set
y_pred = model.predict(X_test)

# Calculate various evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared:", r2)

# Since we are working with a regression problem, a confusion matrix is not applicable, as it's used for classification problems. The metrics we have calculated, including Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared, are appropriate for evaluating the performance of your regression model.

# If we want to visualize the performance of your regression model's predictions, we could create scatter plots of the predicted values versus the actual values. This can help us to visually understand how well your model's predictions align with the true values.


import matplotlib.pyplot as plt
import numpy as np

# Assuming you have already trained your regression model 'model' and made predictions 'y_pred'
# y_test contains the actual target values from your test set

# Create scatter plot for predicted vs. actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color="blue", alpha=0.5)
plt.title("Predicted vs. Actual Values")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()
# In this code, replace `y_test` with your actual target values and `y_pred` with the predicted values obtained from your regression model. This scatter plot will show how well your model's predictions align with the true values. Points along the diagonal line indicate accurate predictions.

# You might also consider adding a line of best fit (regression line) to the scatter plot to visualize the trend more clearly:

# Calculate the coefficients of the regression line
coef = np.polyfit(y_test, y_pred, 1)
poly1d_fn = np.poly1d(coef)

# Create scatter plot with regression line
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color="blue", alpha=0.5)
plt.plot(y_test, poly1d_fn(y_test), color="red")
plt.title("Predicted vs. Actual Values with Regression Line")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()

# # This will show the relationship between predicted and actual values more clearly, along with the linear regression line representing the trend.
# In this code, replace y_test with your actual target values and y_pred with the predicted values obtained from our regression model. This scatter plot will show how well our model's predictions align with the true values. Points along the diagonal line indicate accurate predictions.

# We might also consider adding a line of best fit (regression line) to the scatter plot to visualize the trend more clearly:

# Calculate the coefficients of the regression line
coef = np.polyfit(y_test, y_pred, 1)
poly1d_fn = np.poly1d(coef)

# Create scatter plot with regression line
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot(y_test, poly1d_fn(y_test), color='red')
plt.title('Predicted vs. Actual Values with Regression Line')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

# 01) Dividing Data into Categories:
# You can create a new column in your DataFrame to categorize university rankings into U1, U2, U3, U4, and U5. You can use the pd.cut() function to create these categories based on the university ranking values.

# 02) Creating Supportive Figures and Text:
# You can create visualizations or summaries that show the distribution of acceptance chances for each university ranking category. This can help provide insights into how university ranking might affect admission chances.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ... (Your existing imports and code)

# Step 1: Categorize University Ranking
df['University Rank Category'] = pd.cut(df['University Rating'], bins=[0, 1, 2, 3, 4, 5], labels=['U1', 'U2', 'U3', 'U4', 'U5'])

# Step 2: Create Supportive Figures and Text
plt.figure(figsize=(12, 6))
sns.boxplot(x='University Rank Category', y='Chance of Admit ', data=df)
plt.title('Distribution of Admission Chances by University Rank Category')
plt.xlabel('University Rank Category')
plt.ylabel('Chance of Admission')
plt.show()

# Print summary statistics for each university rank category
summary = df.groupby('University Rank Category')['Chance of Admit '].describe()
print(summary)

