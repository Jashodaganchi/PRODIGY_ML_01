import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
# Load dataset
data = pd.read_csv("train.csv")
# Select only required features
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
target = 'SalePrice'
# Remove missing values
data = data[features + [target]].dropna()
X = data[features]
y = data[target]
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Train model
model = LinearRegression()
model.fit(X_train, y_train)
# Predictions
predictions = model.predict(X_test)
# Evaluation
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print("Model trained successfully")
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)
# Example prediction
example= pd.DataFrame({
    'GrLivArea': [2000],
    'BedroomAbvGr': [3],
    'FullBath': [2]
})
predicted_price = model.predict(example)
print("Predicted price for house:", predicted_price[0])