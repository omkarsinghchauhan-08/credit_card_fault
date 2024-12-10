import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def save_object(file_path, obj):
    """
    Save an object to a specified file path using pickle.

    Parameters:
    file_path (str): The path to save the serialized object.
    obj (object): The Python object to serialize and save.

    Raises:
    CustomException: If any error occurs during the save operation.
    """
    try:
        # Ensure the directory exists before saving
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Serialize and save the object
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
    except Exception as e:
        raise Exception(e)

def load_dataset(filepath):
    """
    Load the dataset from a CSV file.
    Args:
        filepath (str): Path to the dataset CSV file.
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        data = pd.read_csv(filepath)
        return data
    except Exception as e:
        raise Exception(f"Error loading dataset: {e}")

def preprocess_data(data):
    """
    Preprocess the dataset for model training or prediction.
    Args:
        data (pd.DataFrame): The raw dataset.
    Returns:
        pd.DataFrame: Processed dataset.
    """
    try:
        # Ensure categorical variables are properly encoded
        categorical_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
        data[categorical_cols] = data[categorical_cols].astype('category')
        
        # Handle missing values (if any)
        data = data.fillna(0)
        
        return data
    except Exception as e:
        raise Exception(f"Error preprocessing data: {e}")

def split_data(data, target_column, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    Args:
        data (pd.DataFrame): The dataset to split.
        target_column (str): Name of the target column.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random state for reproducibility.
    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    try:
        X = data.drop(columns=[target_column])
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise Exception(f"Error splitting data: {e}")

def save_model(model, filepath):
    """
    Save a trained model to a file.
    Args:
        model: Trained model object.
        filepath (str): Filepath to save the model.
    """
    try:
        with open(filepath, 'wb') as file:
            pickle.dump(model, file)
    except Exception as e:
        raise Exception(f"Error saving model: {e}")
    
import pickle


def load_object(file_path):
    """
    Load a Python object from a file using pickle deserialization.

    Parameters:
    file_path (str): The path where the object is stored.

    Returns:
    object: The deserialized Python object.

    Raises:
    CustomException: If any error occurs during the load operation.
    """
    try:
        # Check if the file exists before attempting to load
        if not os.path.exists(file_path):
            raise Exception(f"File not found at {file_path}")

        # Open the file and load the object
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
        
        return obj
    except Exception as e:
        raise Exception(f"Error loading object from {file_path}: {e}")


def load_model(filepath):
    """
    Load a trained model from a file.
    Args:
        filepath (str): Path to the saved model file.
    Returns:
        Model object: Loaded model.
    """
    try:
        with open(filepath, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {e}")

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data.
    Args:
        model: Trained model object.
        X_test (pd.DataFrame): Features for testing.
        y_test (pd.Series): True labels for testing.
    Returns:
        dict: Evaluation metrics (accuracy, classification report).
    """
    try:
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        return {"accuracy": accuracy, "classification_report": report}
    except Exception as e:
        raise Exception(f"Error evaluating model: {e}")

def get_feature_summary(data):
    """
    Generate a summary of the dataset features.
    Args:
        data (pd.DataFrame): Dataset to summarize.
    Returns:
        pd.DataFrame: Summary of dataset features.
    """
    try:
        summary = data.describe().transpose()
        return summary
    except Exception as e:
        raise Exception(f"Error generating feature summary: {e}")
