import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    """Stores the path for train and test data"""

    # artifacts folder store the outputs
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Initiates the data ingestion process by fetching the data from the source,
        splitting it into training and testing sets, saving the sets to disk,
        and returning the paths to the training and testing sets.

        :return: A tuple containing the paths to the training and testing sets.
        """

        logging.info("Entered data ingestion method")

        try:
            # fetch dataset from UCI
            heart_disease = fetch_ucirepo(id=45)
            X = heart_disease.data.features
            y = heart_disease.data.targets

            """
            From the repository
            The "goal" field refers to the presence of heart disease in the 
            patient.  
            It is integer valued from 0 (no presence) to 4. 
            Experiments with the Cleveland database have concentrated on 
            simply attempting to distinguish presence (values 1,2,3,4) 
            from absence (value 0). 
            """

            X = pd.DataFrame(X, columns=heart_disease.feature_names)

            # making y binary where 0 means absence and 1 means presence
            y_processed = pd.DataFrame(np.where(y == 0, 0, 1), columns=["num"])

            df = pd.concat([X, y_processed], axis=1)

            logging.info("Read the dataset as dataframe")
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )

            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )

            logging.info("Ingestion of data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()
    except Exception as e:
        raise CustomException(e, sys)
