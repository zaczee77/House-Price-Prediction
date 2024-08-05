import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Load the dataset from a CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Perform preprocessing steps on the dataset."""
    # Example preprocessing: Drop rows with missing values
    df = df.dropna()
    
    # Example preprocessing: Encode categorical features
    df = pd.get_dummies(df, drop_first=True)
    
    # Example preprocessing: Standardize numerical features
    scaler = StandardScaler()
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    return df

def save_data(df, filepath):
    """Save the processed data to a new CSV file."""
    df.to_csv(filepath, index=False)

def split_and_save_data(df, train_filepath, test_filepath):
    """Split data into training and testing sets and save them to CSV files."""
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv(train_filepath, index=False)
    test_df.to_csv(test_filepath, index=False)

if __name__ == "__main__":
    # Update the file paths as needed
    input_filepath = '/Users/nirmeetrao/house_price_prediction/data/USA_Housing.csv'
    processed_filepath = '/Users/nirmeetrao/house_price_prediction/data/processed_housing.csv'
    train_filepath = '/Users/nirmeetrao/house_price_prediction/data/processed_housing_train.csv'
    test_filepath = '/Users/nirmeetrao/house_price_prediction/data/processed_housing_test.csv'
    
    df = load_data(input_filepath)
    df = preprocess_data(df)
    save_data(df, processed_filepath)
    split_and_save_data(df, train_filepath, test_filepath)
    print("Data preprocessing and splitting completed.")
