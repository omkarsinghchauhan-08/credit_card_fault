import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object 

class PredictPipeline:
    """
    Prediction Pipeline to handle model loading and prediction.
    """

    def __init__(self):
        # Paths to the saved preprocessor and model
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        self.model_path = os.path.join('artifacts', 'model.pkl')

    def predict(self, features):
        """
        Predicts the target label for the given features using the trained model and preprocessor.
        """
        try:
            logging.info("Loading preprocessor and model...")
            
            # Load the preprocessor and model
            preprocessor = load_object(self.preprocessor_path)
            model = load_object(self.model_path)

            logging.info("Transforming the features using the preprocessor...")
            # Transform the features using the preprocessor
            data_scaled = preprocessor.transform(features)

            logging.info("Making predictions using the model...")
            # Make predictions using the model
            predictions = model.predict(data_scaled)

            return predictions

        except Exception as e:
            logging.error("Exception occurred during prediction.")
            raise CustomException(e, sys)

class CustomData:
    """
    A class to handle user input data and format it into a DataFrame
    for the prediction pipeline.
    """

    def __init__(
        self,
        LIMIT_BAL: float,
        SEX: int,
        EDUCATION: int,
        MARRIAGE: int,
        AGE: int,
        PAY_0: int,
        PAY_2: int,
        PAY_3: int,
        PAY_4: int,
        PAY_5: int,
        PAY_6: int,
        BILL_AMT1: float,
        BILL_AMT2: float,
        BILL_AMT3: float,
        BILL_AMT4: float,
        BILL_AMT5: float,
        BILL_AMT6: float,
        PAY_AMT1: float,
        PAY_AMT2: float,
        PAY_AMT3: float,
        PAY_AMT4: float,
        PAY_AMT5: float,
        PAY_AMT6: float
    ):
        self.LIMIT_BAL = LIMIT_BAL
        self.SEX = SEX
        self.EDUCATION = EDUCATION
        self.MARRIAGE = MARRIAGE
        self.AGE = AGE
        self.PAY_0 = PAY_0
        self.PAY_2 = PAY_2
        self.PAY_3 = PAY_3
        self.PAY_4 = PAY_4
        self.PAY_5 = PAY_5
        self.PAY_6 = PAY_6
        self.BILL_AMT1 = BILL_AMT1
        self.BILL_AMT2 = BILL_AMT2
        self.BILL_AMT3 = BILL_AMT3
        self.BILL_AMT4 = BILL_AMT4
        self.BILL_AMT5 = BILL_AMT5
        self.BILL_AMT6 = BILL_AMT6
        self.PAY_AMT1 = PAY_AMT1
        self.PAY_AMT2 = PAY_AMT2
        self.PAY_AMT3 = PAY_AMT3
        self.PAY_AMT4 = PAY_AMT4
        self.PAY_AMT5 = PAY_AMT5
        self.PAY_AMT6 = PAY_AMT6

    def get_data_as_dataframe(self):
        """
        Converts the input data into a pandas DataFrame for prediction.
        """
        try:
            logging.info("Converting custom data into a DataFrame...")
            custom_data_input_dict = {
                'LIMIT_BAL': [self.LIMIT_BAL],
                'SEX': [self.SEX],
                'EDUCATION': [self.EDUCATION],
                'MARRIAGE': [self.MARRIAGE],
                'AGE': [self.AGE],
                'PAY_0': [self.PAY_0],
                'PAY_2': [self.PAY_2],
                'PAY_3': [self.PAY_3],
                'PAY_4': [self.PAY_4],
                'PAY_5': [self.PAY_5],
                'PAY_6': [self.PAY_6],
                'BILL_AMT1': [self.BILL_AMT1],
                'BILL_AMT2': [self.BILL_AMT2],
                'BILL_AMT3': [self.BILL_AMT3],
                'BILL_AMT4': [self.BILL_AMT4],
                'BILL_AMT5': [self.BILL_AMT5],
                'BILL_AMT6': [self.BILL_AMT6],
                'PAY_AMT1': [self.PAY_AMT1],
                'PAY_AMT2': [self.PAY_AMT2],
                'PAY_AMT3': [self.PAY_AMT3],
                'PAY_AMT4': [self.PAY_AMT4],
                'PAY_AMT5': [self.PAY_AMT5],
                'PAY_AMT6': [self.PAY_AMT6],
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("Custom data successfully converted to DataFrame.")
            return df

        except Exception as e:
            logging.error("Exception occurred while converting custom data into DataFrame.")
            raise CustomException(e, sys)
