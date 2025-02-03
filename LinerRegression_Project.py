import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load and Explore Data
customers = pd.read_csv('Ecommerce Customers')

print(customers.head())
print(customers.info())
print(customers.describe())


# Exploratory Data Analysis (EDA)
# Create a jointplot to compare the Time on Website and Yearly Amount Spent columns
sns.jointplot(data=customers, x='Time on Website', y='Yearly Amount Spent', kind='scatter')
plt.show()

# Same for Time on App column
sns.jointplot(data=customers, x='Time on App', y='Yearly Amount Spent', kind='scatter')
plt.show()

# Use jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership
sns.jointplot(data=customers, x='Time on App', y='Length of Membership', kind='hex')
plt.show()

# Explore types of relationships across entire data set
sns.pairplot(customers)
plt.show()

# Create a linear model plot of Yearly Amount Spent vs. Length of Membership
sns.lmplot(data=customers, x='Length of Membership', y='Yearly Amount Spent')
plt.show()


# Train-Test Split
# Split data into training and testing sets
y = customers['Yearly Amount Spent']
X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# Train the Linear Regression Model
# Create and train the model
lm = LinearRegression()
lm.fit(X_train, y_train)

# Print model coefficients
print("\nModel Coefficients:\n", pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient']))


# Predict Test Data
# Make predictions
predictions = lm.predict(X_test)

# Create a scatterplot of the real values vs. the predicted values
plt.figure(figsize=(8, 5))
plt.scatter(y_test, predictions)
plt.xlabel('Y Test (True Values)')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.show()


# Model Evaluation
# Calculate the Mean Absolute Error, Mean Squared Error and Root Mean Squared Error
mae = metrics.mean_absolute_error(y_test, predictions)
mse = metrics.mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
explained_variance = metrics.explained_variance_score(y_test, predictions)

# Print evaluation metrics
print(f"\nModel Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Explained Variance Score: {explained_variance:.4f}")


# Residuals Analysis
# Plot a histogram of the residuals
plt.figure(figsize=(8, 5))
sns.histplot((y_test - predictions), bins=50, kde=True)
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()
