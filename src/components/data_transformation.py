import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "proprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, train_path):
        """
        This function is responsible for data trnasformation

        """
        try:
            logging.info(f"INside get data transformer object")

            target_column_name = "num"
            train_df = pd.read_csv(train_path)
            X_train = train_df.drop(columns=[target_column_name], axis=1)

            numeric_feat = ["age", "trestbps", "chol", "thalach", "oldpeak"]
            categorical_feat = [
                "sex",
                "restecg",
                "thal",
                "slope",
                "cp",
                "exang",
                "ca",
                "fbs",
            ]

            # Categorical Features PipeLines
            categorical__feat_pipeline = Pipeline(
                steps=[("knn_imputer", KNNImputer(n_neighbors=1))]
            )
            logging.info(f"Categorical features: {categorical_feat}")

            # Numeric Features PipeLines
            numeric_feat_pipeline = Pipeline(steps=[("scaler", RobustScaler())])

            logging.info(f"Numerical columns: {numeric_feat}")

            preprocessor1 = ColumnTransformer(
                transformers=[
                    ("knn_imputation", categorical__feat_pipeline, ["ca", "thal"]),
                    ("numeric_pipeline", numeric_feat_pipeline, numeric_feat),
                ],
                verbose_feature_names_out=False,
                remainder="passthrough",
            )

            preprocessor = Pipeline(
                steps=[
                    ("feature_eng_steps", preprocessor1),
                    (
                        "feature_selection_steps",
                        SelectKBest(score_func=mutual_info_classif, k=10),
                    ),
                ]
            ).set_output(transform="pandas")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object(train_path)

            target_column_name = "num"
            numeric_feat = ["age", "trestbps", "chol", "thalach", "oldpeak"]
            categorical_feat = [
                "sex",
                "restecg",
                "thal",
                "slope",
                "cp",
                "exang",
                "ca",
                "fbs",
            ]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            print(input_feature_train_df)

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df, target_feature_train_df
            )

            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
