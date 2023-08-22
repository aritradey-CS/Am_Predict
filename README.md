# Am_Predict
Admission Prediction The provided code seems to be an analysis and modeling script for a dataset related to admission predictions. The code performs various data exploration, visualization, and machine learning tasks. Here's a breakdown of what the code does:

1. **Import Libraries:** The code begins by importing necessary libraries such as numpy, pandas, matplotlib, seaborn, and other required modules.

2. **Load and Explore Data:** The script reads a CSV file named "Admission_Predict.csv" into a pandas DataFrame and displays the first 10 rows using `.head()`. It also checks for null values and displays the columns of the DataFrame.

3. **Data Visualization:** Various visualizations are created using seaborn and matplotlib. These include:
   - A heatmap showing correlations between numerical features.
   - Distribution plots (histograms and KDE) for 'GRE Score' and 'TOEFL Score'.
   - A scatterplot showing the relationship between 'University Rating' and 'CGPA'.
   - Bar plots displaying the relationship between 'GRE Score' and 'Chance of Admit' and 'TOEFL Score' and 'Chance of Admit'.
   - A bar plot of 'University Ratings' for candidates with an acceptance chance of 75% or higher.

4. **Data Statistics:** The script calculates and prints the average GRE Score, TOEFL Score, CGPA, and chance of getting admitted.

5. **Mount Google Drive:** It mounts Google Drive using the `google.colab` library (assuming you're using this script in Google Colab).

6. **Top Applicants:** Identifies top applicants based on specific criteria for GRE Score, TOEFL Score, and CGPA.

7. **Data Preprocessing:** Splits the data into features (X) and target ('Chance of Admit') (y). It also normalizes the features using `preprocessing.normalize()` and then splits the data into training and testing sets using `train_test_split()`.

8. **Modeling and Evaluation:** The script evaluates multiple regression models and classification models. It uses a variety of algorithms and measures to assess model performance. For regression:
   - Linear Regression
   - Extra Trees Regression
   - K-Neighbors Regression
   - Support Vector Regression

   For classification:
   - Logistic Regression
   - Extra Trees Classification
   - K-Neighbors Classification
   - Support Vector Classification

   Model performance is evaluated using mean squared error (for regression) and accuracy (for classification). Bar plots are used to visualize the performance of different models.

Overall, this code combines data preprocessing, visualization, and machine learning techniques to analyze and model the admission prediction dataset. The specific dataset "Admission_Predict.csv" contains features such as GRE Score, TOEFL Score, University Rating, and CGPA, along with the target variable 'Chance of Admit'. The code aims to predict the likelihood of admission based on these features using various machine learning models.
