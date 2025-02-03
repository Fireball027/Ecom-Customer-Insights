## Overview

The **E-commerce Customer Analysis** project is designed to understand customer behavior by applying **linear regression**. It utilizes an e-commerce dataset to find relationships between features like time spent on the website, session duration, and annual spending. The goal is to build a predictive model that estimates customer spending.

---

## Key Features

- **Data Preprocessing**: Cleans and prepares customer data.
- **Exploratory Data Analysis (EDA)**: Identifies patterns and correlations.
- **Linear Regression Model**: Predicts customer spending based on input features.
- **Visualization**: Generates graphs to display regression results and data insights.

---

## Project Files

### 1. `LinerRegression_Project.py`
This script performs data analysis, model training, and visualization.

#### Key Components:

- **Data Loading**:
  - Reads the `Ecommerce Customers` dataset.

- **Exploratory Data Analysis**:
  - Uses Seaborn and Matplotlib to visualize relationships.

- **Model Training**:
  - Splits the dataset into training and test sets.
  - Fits a **Linear Regression** model to predict customer spending.

- **Model Evaluation**:
  - Measures performance using **R-squared** and **Mean Absolute Error (MAE)**.

#### Example Code:
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv('Ecommerce Customers')

# Split data into features and target
X = data[['Time on Website', 'Time on App', 'Length of Membership']]
y = data['Yearly Amount Spent']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print(f'R-squared: {r2_score(y_test, y_pred)}')
print(f'MAE: {mean_absolute_error(y_test, y_pred)}')
```

### 2. `Ecommerce Customers`
This dataset contains customer behavior data with features like:

- **Email**: Customer email ID.
- **Time on Website**: Duration spent on the website.
- **Time on App**: App usage time.
- **Length of Membership**: How long the customer has been with the service.
- **Yearly Amount Spent**: The target variable representing customer spending.

---

## How to Run the Project

### Step 1: Install Dependencies
Ensure you have Python installed, then install the required libraries:
```bash
pip install pandas seaborn matplotlib scikit-learn
```

### Step 2: Run the Script
Execute the main script:
```bash
python LinerRegression_Project.py
```

### Step 3: View Insights
- Model performance metrics (R-squared, MAE)
- Relationship graphs (EDA)
- Predicted vs Actual Spending comparisons

---

## Future Enhancements

- **Polynomial Regression**: Improve predictions with non-linear relationships.
- **Feature Engineering**: Add more relevant customer metrics.
- **Automated Reporting**: Generate summary reports on customer behavior.
- **Deployment**: Integrate with a web app for live predictions.

---

## Conclusion

The **E-commerce Customer Analysis** project is an insightful application of machine learning in business analytics. It provides valuable insights into customer spending behavior and builds a predictive model to enhance marketing and customer engagement strategies.

---

**Happy Analyzing!**

