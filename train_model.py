import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def load_data(filepath):
    """Load the dataset from a CSV file."""
    return pd.read_csv(filepath)

def train_model(train_filepath, test_filepath):
    """Train a model and evaluate it."""
    # Load and split the data
    train = load_data(train_filepath)
    test = load_data(test_filepath)
    
    # Separate features and target variable
    X_train = train.drop('Price', axis=1)  # Assuming 'Price' is the target variable
    y_train = train['Price']
    X_test = test.drop('Price', axis=1)
    y_test = test['Price']
    
    # Train a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    
    return model

if __name__ == "__main__":
    # Update the file paths as needed
    train_filepath = '/Users/nirmeetrao/house_price_prediction/data/processed_housing_train.csv'
    test_filepath = '/Users/nirmeetrao/house_price_prediction/data/processed_housing_test.csv'
    
    model = train_model(train_filepath, test_filepath)
    print("Model training completed.")
