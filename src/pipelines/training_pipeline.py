import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

# Define the paths
DATASET_URL = "https://raw.githubusercontent.com/sunnysavita10/credit_card_pw_hindi/main/creditCardFraud_28011964_120214.csv"
MODEL_SAVE_PATH = "artifacts/credit_card_fault_model.pkl"

# Step 1: Load Dataset
def load_data():
    """
    Load the dataset from the given URL or local path.
    """
    print("Loading dataset...")
    df = pd.read_csv(DATASET_URL)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    return df

# Step 2: Data Preprocessing
def preprocess_data(df):
    """
    Preprocess the dataset: Handle missing values, scale numerical features, etc.
    """
    print("Starting preprocessing...")

    # Handle missing values if any
    if df.isnull().sum().any():
        print("Handling missing values...")
        df.fillna(df.median(), inplace=True)

    # Rename target column for convenience
    df.rename(columns={'default payment next month': 'target'}, inplace=True)

    # Features and target separation
    X = df.drop(columns=['target'])
    y = df['target']

    print("Preprocessing completed.")
    return X, y

# Step 3: Split Data
def split_data(X, y):
    """
    Split data into training and testing sets.
    """
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Data split completed.")
    return X_train, X_test, y_train, y_test

# Step 4: Build Model Pipeline
def build_pipeline():
    """
    Build a machine learning pipeline with preprocessing and model training.
    """
    print("Building the model pipeline...")
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    print("Pipeline built successfully.")
    return pipeline

# Step 5: Train the Model
def train_model(pipeline, X_train, y_train):
    """
    Train the machine learning model.
    """
    print("Training the model...")
    pipeline.fit(X_train, y_train)
    print("Model training completed.")
    return pipeline

# Step 6: Evaluate Model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test dataset.
    """
    print("Evaluating the model...")
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    print(f"Model Accuracy: {acc:.2f}")
    print("Classification Report:\n", report)

# Step 7: Save the Trained Model
def save_model(model, path=MODEL_SAVE_PATH):
    """
    Save the trained model as a pickle file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Saving the model to {path}...")
    with open(path, 'wb') as file:
        pickle.dump(model, file)
    print("Model saved successfully.")

# Main Training Pipeline
def main():
    # Step 1: Load the dataset
    data = load_data()

    # Step 2: Preprocess the data
    X, y = preprocess_data(data)

    # Step 3: Split the data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 4: Build the model pipeline
    pipeline = build_pipeline()

    # Step 5: Train the model
    trained_model = train_model(pipeline, X_train, y_train)

    # Step 6: Evaluate the model
    evaluate_model(trained_model, X_test, y_test)

    # Step 7: Save the trained model
    save_model(trained_model)

if __name__ == "__main__":
    main()
