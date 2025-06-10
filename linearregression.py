# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 2. Load and Preprocess Dataset
df = pd.read_csv('Housing.csv')
print("✅ Dataset loaded successfully.\n")

# Display basic info
print("🔍 Dataset Info:")
print(df.info())
print("\n🔍 Missing Values:")
print(df.isnull().sum())

# Convert categorical columns if needed (example: 'mainroad')
# df = pd.get_dummies(df, drop_first=True)  # One-hot encode if required

# Select numeric columns for regression
df_numeric = df.select_dtypes(include=[np.number])
df_numeric = df_numeric.dropna()  # Drop rows with missing values

# Choose an independent variable (e.g., 'area') and a target (e.g., 'price')
X = df_numeric[['area']]  # Feature
y = df_numeric['price']   # Target

# 3. Split into Train-Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\n✅ Data split into training and test sets.")

# 4. Fit Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
print("✅ Linear Regression model trained.")

5# Predict
y_pred = model.predict(X_test)

# 5. Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Model Evaluation Metrics:")
print(f"MAE (Mean Absolute Error): {mae:.2f}")
print(f"MSE (Mean Squared Error): {mse:.2f}")
print(f"R² (R-squared): {r2:.2f}")

# 6. Coefficients Interpretation
print(f"\n🔍 Regression Coefficients:")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Slope for area → price: {model.coef_[0]:.2f}")

# 7. Plot Regression Line
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test['area'], y=y_test, label="Actual Price", color="blue")
plt.plot(X_test, y_pred, color="red", label="Regression Line")
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Linear Regression: Price vs Area")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
