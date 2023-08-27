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

# After performing the steps to split your data, calculate feature importance, and select important features, you can proceed with model selection and analysis. For each model, you can train, predict, and evaluate performance for each category separately.

                    # import pandas as pd
                    # from sklearn.model_selection import train_test_split
                    # from sklearn.ensemble import RandomForestClassifier
                    # from sklearn.linear_model import LogisticRegression
                    # from sklearn.metrics import confusion_matrix, f1_score

                    # # Load and preprocess your dataset
                    # # df = pd.read_csv('your_dataset.csv')
                    # # ... Preprocessing steps ...

                    # # Define the categorical columns you want to one-hot encode
                    # categorical_columns = ['University Rank Category']  # Add more columns if needed

                    # # Perform one-hot encoding on categorical columns
                    # df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

                    # # Define features and target variable
                    # X = df_encoded.drop('Chance of Admit ', axis=1)
                    # y = df_encoded['Chance of Admit ']

                    # # Split the dataset into train and test sets
                    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # # Initialize models for analysis
                    # models = {
                    #     'Random Forest': RandomForestClassifier(),
                    #     'Logistic Regression': LogisticRegression()
                    # }

                    # # Loop through each University Rank Category
                    # for category in df['University Rank Category'].unique():
                    #     print(f"University Rank Category: {category}")

                    #     # Filter data for the current category
                    #     category_data = df_encoded[df_encoded['University Rank Category_' + category] == 1]

                    #     # Prepare datasets for the current category
                    #     X_train_category = X_train[X_train.index.isin(category_data.index)]
                    #     X_test_category = X_test[X_test.index.isin(category_data.index)]
                    #     y_train_category = y_train[y_train.index.isin(category_data.index)]
                    #     y_test_category = y_test[y_test.index.isin(category_data.index)]

                    #     # Train and evaluate selected models
                    #     for model_name, model in models.items():
                    #         print(f"\nModel: {model_name}")

                    #         # Train the model
                    #         model.fit(X_train_category, y_train_category)

                    #         # Make predictions
                    #         y_pred = model.predict(X_test_category)

                    #         # Evaluate performance using confusion matrix and F1 score
                    #         conf_matrix = confusion_matrix(y_test_category, y_pred)
                    #         f1 = f1_score(y_test_category, y_pred)

                    #         print("Confusion Matrix:\n", conf_matrix)
                    #         print("F1 Score:", f1)

                                    # import pandas as pd
                                    # from sklearn.model_selection import train_test_split
                                    # from sklearn.ensemble import RandomForestRegressor
                                    # from sklearn.linear_model import LinearRegression
                                    # from sklearn.metrics import mean_squared_error

                                    # # Load your dataset
                                    # df = pd.read_csv('Admission_Predict.csv')  # Replace with your dataset path

                                    # # Define your features and target variable
                                    # X = df[['GRE Score', 'TOEFL Score', 'SOP', 'LOR ', 'CGPA', 'Research', 'University Rating']]
                                    # y = df['Chance of Admit ']

                                    # # Perform one-hot encoding on 'University Rating' column
                                    # categorical_columns = ['University Rating']
                                    # X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

                                    # # Split the dataset into train and test sets
                                    # X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

                                    # # Initialize models for analysis
                                    # models = {
                                    #     'Linear Regression': LinearRegression(),
                                    #     'Random Forest': RandomForestRegressor(),
                                    # }

                                    # # Loop through each University Rating
                                    # for category in X_encoded['University Rating'].unique():
                                    #     print(f"University Rating: {category}")
                                        
                                    #     # Prepare datasets for the current category
                                    #     X_train_category = X_train[X_train['University Rating'] == category]
                                    #     y_train_category = y_train[X_train_category.index]
                                    #     X_test_category = X_test[X_test['University Rating'] == category]
                                    #     y_test_category = y_test[X_test_category.index]

                                    #     # Train and evaluate models
                                    #     for model_name, model in models.items():
                                    #         print(f"\nModel: {model_name}")
                                            
                                    #         # Train the model
                                    #         model.fit(X_train_category, y_train_category)
                                            
                                    #         # Make predictions
                                    #         y_pred = model.predict(X_test_category)
                                            
                                    #         # Calculate Mean Squared Error
                                    #         mse = mean_squared_error(y_test_category, y_pred)
                                    #         print("Mean Squared Error:", mse)
# --------------------------------------------------------------------------------------------------------------------------------
# Creating a new column 'University Rank Category' that categorizes the university ranking based on specified bins. Using a box plot to visualize the distribution of admission chances across different university rank categories. Printing summary statistics for each university rank category using the describe() function.
# --------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# Load your dataset and preprocess it
df = pd.read_csv('Admission_Predict.csv')
# ... Perform preprocessing steps ...

# Define your features and target variable
X = df.drop('Chance of Admit ', axis=1)
y = df['Chance of Admit ']

# Perform one-hot encoding on your features
X_encoded = pd.get_dummies(X, drop_first=True)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Step 1: Feature Importance Selection using RandomForestRegressor
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

# Get feature importances
feature_importances = rf_model.feature_importances_

# Create a list of feature names
feature_names = list(X_train.columns)

# Create a dictionary to store feature importance scores
feature_importance_dict = dict(zip(feature_names, feature_importances))

# Select the top 3 or 4 features with the highest importance scores
important_features = sorted(feature_importance_dict, key=feature_importance_dict.get, reverse=True)[:3]

# Step 2: Model Selection and Analysis
# Prepare datasets with only important features
X_train_selected = X_train[important_features]
X_test_selected = X_test[important_features]

# Initialize models for analysis
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
}

# ... Rest of your code ...

# Train and evaluate selected models
for model_name, model in models.items():
    # Train the model on the selected features
    model.fit(X_train_selected, y_train)

    # Make predictions
    y_pred = model.predict(X_test_selected)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"{model_name} RMSE: {rmse:.2f}")



                    # import pandas as pd
                    # from sklearn.model_selection import train_test_split
                    # from sklearn.ensemble import RandomForestRegressor
                    # from sklearn.linear_model import LinearRegression
                    # from sklearn.metrics import mean_absolute_error, mean_squared_error

                    # # Load and preprocess your dataset
                    # # df = pd.read_csv('your_dataset.csv')
                    # # ... Preprocessing steps ...
                    # # Assuming 'University Rating' is a categorical feature
                    # df['University Rank Category'] = 'U' + df['University Rating'].astype(str)

                    # # Define the categorical columns you want to one-hot encode
                    # categorical_columns = ['University Rank Category']  # Add more columns if needed

                    # # Perform one-hot encoding on categorical columns
                    # df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

                    # # Define features and target variable
                    # X = df_encoded.drop('Chance of Admit ', axis=1)
                    # y = df_encoded['Chance of Admit ']

                    # # Split the dataset into train and test sets
                    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # # Initialize models for analysis
                    # models = {
                    #     'Random Forest': RandomForestRegressor(),
                    #     'Linear Regression': LinearRegression()
                    # }

                    # # ... (previous code)

                    # # Loop through each University Rank Category
                    # for category in df['University Rank Category'].unique():
                    #     print(f"University Rank Category: {category}")

                    #     # Filter data for the current category
                    #     column_name = 'University Rank Category_' + category
                    #     # category_data = df_encoded[df_encoded[column_name] == 1]

                    #     # Prepare datasets for the current category
                    #     X_train_category = X_train[X_train.index.isin(category_data.index)]
                    #     X_test_category = X_test[X_test.index.isin(category_data.index)]
                    #     y_train_category = y_train[y_train.index.isin(category_data.index)]
                    #     y_test_category = y_test[y_test.index.isin(category_data.index)]

                    #     # Train and evaluate selected models
                    #     for model_name, model in models.items():
                    #         print(f"\nModel: {model_name}")

                    #         # Train the model
                    #         model.fit(X_train_category, y_train_category)

                    #         # Make predictions
                    #         y_pred = model.predict(X_test_category)

                    #         # Evaluate performance using MAE, MSE, and RMSE
                    #         mae = mean_absolute_error(y_test_category, y_pred)
                    #         mse = mean_squared_error(y_test_category, y_pred)
                    #         rmse = mean_squared_error(y_test_category, y_pred, squared=False)

                    #         print("Mean Absolute Error:", mae)
                    #         print("Mean Squared Error:", mse)
                    #         print("Root Mean Squared Error:", rmse)


#  Now I created a bar chart showing the Mean Squared Error (MSE) for each model and each University Rank
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load and preprocess your dataset
# df = pd.read_csv('your_dataset.csv')
# ... Preprocessing steps ...
# Assuming 'University Rating' is a categorical feature
df['University Rank Category'] = 'U' + df['University Rating'].astype(str)

# Define the categorical columns you want to one-hot encode
categorical_columns = ['University Rank Category']  # Add more columns if needed

# Perform one-hot encoding on categorical columns
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Define features and target variable
X = df_encoded.drop('Chance of Admit ', axis=1)
y = df_encoded['Chance of Admit ']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models for analysis
models = {
    'Random Forest': RandomForestRegressor(),
    'Linear Regression': LinearRegression()
}

# Create a dictionary to store MSE values for each model and category
mse_results = {model_name: [] for model_name in models.keys()}

# Loop through each University Rank Category
for category in df['University Rank Category'].unique():
    print(f"University Rank Category: {category}")

    # Filter data for the current category
    column_name = 'University Rank Category_' + category
    # category_data = df_encoded[df_encoded[column_name] == 1]

    # Prepare datasets for the current category
    X_train_category = X_train[X_train.index.isin(category_data.index)]
    X_test_category = X_test[X_test.index.isin(category_data.index)]
    y_train_category = y_train[y_train.index.isin(category_data.index)]
    y_test_category = y_test[y_test.index.isin(category_data.index)]

    # Train and evaluate selected models
    for model_name, model in models.items():
        print(f"\nModel: {model_name}")

        # Train the model
        model.fit(X_train_category, y_train_category)

        # Make predictions
        y_pred = model.predict(X_test_category)

        # Calculate MSE
        mse = mean_squared_error(y_test_category, y_pred)
        mse_results[model_name].append(mse)

# Create a DataFrame from the mse_results dictionary
mse_df = pd.DataFrame(mse_results)

# Plot a bar chart
mse_df.plot(kind='bar', xlabel='University Rank Category', ylabel='Mean Squared Error',
            title='Mean Squared Error by Model and University Rank Category')
plt.xticks(rotation=0)
plt.legend(title='Model')
plt.show()
 
# use the Pandas library to store the Mean Squared Error (MSE) values for each model and University Rank Category in a DataFrame. Here's how you can m

                        import pandas as pd
                        from sklearn.model_selection import train_test_split
                        from sklearn.ensemble import RandomForestRegressor
                        from sklearn.linear_model import LinearRegression
                        from sklearn.metrics import mean_squared_error

                        # Load and preprocess your dataset
                        # df = pd.read_csv('your_dataset.csv')
                        # ... Preprocessing steps ...
                        # Assuming 'University Rating' is a categorical feature
                        df['University Rank Category'] = 'U' + df['University Rating'].astype(str)

                        # Define the categorical columns you want to one-hot encode
                        categorical_columns = ['University Rank Category']  # Add more columns if needed

                        # Perform one-hot encoding on categorical columns
                        df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

                        # Define features and target variable
                        X = df_encoded.drop('Chance of Admit ', axis=1)
                        y = df_encoded['Chance of Admit ']

                        # Split the dataset into train and test sets
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                        # Initialize models for analysis
                        models = {
                            'Random Forest': RandomForestRegressor(),
                            'Linear Regression': LinearRegression()
                        }

                        # Create a dictionary to store MSE values for each model and category
                        mse_results = {model_name: [] for model_name in models.keys()}

                        # Loop through each University Rank Category
                        categories = df['University Rank Category'].unique()
                        for category in categories:
                            print(f"University Rank Category: {category}")

                            # Filter data for the current category
                            column_name = 'University Rank Category_' + category
                            # category_data = df_encoded[df_encoded[column_name] == 1]

                            # Prepare datasets for the current category
                            X_train_category = X_train[X_train.index.isin(category_data.index)]
                            X_test_category = X_test[X_test.index.isin(category_data.index)]
                            y_train_category = y_train[y_train.index.isin(category_data.index)]
                            y_test_category = y_test[y_test.index.isin(category_data.index)]

                            # Train and evaluate selected models
                            for model_name, model in models.items():
                                print(f"\nModel: {model_name}")

                                # Train the model
                                model.fit(X_train_category, y_train_category)

                                # Make predictions
                                y_pred = model.predict(X_test_category)

                                # Calculate MSE
                                mse = mean_squared_error(y_test_category, y_pred)
                                mse_results[model_name].append(mse)

                        # Create a DataFrame for the performance analysis table
                        performance_df = pd.DataFrame(mse_results, index=categories)
                        performance_df.index.name = 'University Rank Category'

                        # Display the performance analysis table
                        print(performance_df)



# convey the performance analysis more effectively. One way to visualize the MSE values for each model and University Rank Category is by using a bar plot. Here's how you can modify the code to create a bar plot using the Seaborn library:

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load and preprocess your dataset
# df = pd.read_csv('your_dataset.csv')
# ... Preprocessing steps ...
# Assuming 'University Rating' is a categorical feature
df['University Rank Category'] = 'U' + df['University Rating'].astype(str)

# Define the categorical columns you want to one-hot encode
categorical_columns = ['University Rank Category']  # Add more columns if needed

# Perform one-hot encoding on categorical columns
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Define features and target variable
X = df_encoded.drop('Chance of Admit ', axis=1)
y = df_encoded['Chance of Admit ']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models for analysis
models = {
    'Random Forest': RandomForestRegressor(),
    'Linear Regression': LinearRegression()
}

# Create a dictionary to store MSE values for each model and category
mse_results = {model_name: [] for model_name in models.keys()}

# Loop through each University Rank Category
categories = df['University Rank Category'].unique()
for category in categories:
    print(f"University Rank Category: {category}")

    # Filter data for the current category
    column_name = 'University Rank Category_' + category
    # category_data = df_encoded[df_encoded[column_name] == 1]

    # Prepare datasets for the current category
    X_train_category = X_train[X_train.index.isin(category_data.index)]
    X_test_category = X_test[X_test.index.isin(category_data.index)]
    y_train_category = y_train[y_train.index.isin(category_data.index)]
    y_test_category = y_test[y_test.index.isin(category_data.index)]

    # Train and evaluate selected models
    for model_name, model in models.items():
        print(f"\nModel: {model_name}")

        # Train the model
        model.fit(X_train_category, y_train_category)

        # Make predictions
        y_pred = model.predict(X_test_category)

        # Calculate MSE
        mse = mean_squared_error(y_test_category, y_pred)
        mse_results[model_name].append(mse)

# Create a DataFrame for the performance analysis table
performance_df = pd.DataFrame(mse_results, index=categories)
performance_df.index.name = 'University Rank Category'

# Display the performance analysis table
print(performance_df)

# Create a bar plot using Seaborn
plt.figure(figsize=(10, 6))
sns.barplot(data=performance_df, x=performance_df.index, y='Random Forest', label='Random Forest')
sns.barplot(data=performance_df, x=performance_df.index, y='Linear Regression', label='Linear Regression')
plt.xlabel('University Rank Category')
plt.ylabel('Mean Squared Error')
plt.title('Performance Analysis by Model and University Rank Category')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
