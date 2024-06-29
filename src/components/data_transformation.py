import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer,SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler,FunctionTransformer,RobustScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from src.utils import save_object
from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self,train_path):
        '''
        This function is responsible for data trnasformation
        
        '''
        try:
            target_column_name="num"
            train_df=pd.read_csv(train_path)
            X_train=train_df.drop(columns=[target_column_name],axis=1)


            numeric_feat = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
            categorical_feat = ['sex', 'restecg', 'thal', 'slope', 'cp', 'exang', 'ca','fbs']

            # Categorical Features PipeLines
            categorical__feat_pipeline = Pipeline(
                steps = [
                ("knn_imputer", KNNImputer(n_neighbors=1))
                ]
            )
            logging.info(f"Categorical columns: {categorical_feat}")


            # Numeric Features PipeLines
            numeric_feat_pipeline = Pipeline(
                steps = [
                ("scaler", RobustScaler())
                ]
            )

            logging.info(f"Numerical columns: {numeric_feat}")


            preprocessor1 = ColumnTransformer(
                transformers = [
                    ("knn_imputation", categorical__feat_pipeline, ['ca','thal']),
                    ("numeric_pipeline", numeric_feat_pipeline, numeric_feat)
                ],verbose_feature_names_out = False,
                remainder = "passthrough"
            )
            preprocessor1

            print("Index of categorical features in X_train")
            L = []
            for i in range(len(X_train.columns)):
                if X_train.columns[i] in categorical_feat:
                    L.append(i)
            print(L)

            def custom_mutual_info_classif(X,y):
                return mutual_info_classif(X, y, discrete_features=L)

            preprocessor = Pipeline (
                steps = [
                    ("feature_eng_steps", preprocessor1),
                    ("feature_selection_steps", SelectKBest(score_func=custom_mutual_info_classif,k=10))
                ]
            ).set_output(transform='pandas')

            # ///////////////////////////////////////////////////////////////
            # numerical_columns = ["age", "sex" , "cp", "trestbps" , "chol", "fbs", "restecg", "thalach","exang","oldpeak","slope","ca","thal"]
            # num_pipeline= Pipeline(
            #     steps=[
            #     ("imputer",SimpleImputer(strategy="median")),
            #     ("scaler",StandardScaler())

            #     ]
            # )

            # logging.info(f"Numerical columns: {numerical_columns}")

            # preprocessor=ColumnTransformer(
            #     [
            #     ("num_pipeline",num_pipeline,numerical_columns)

            #     ]


            # )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object(train_path)

            target_column_name="num"
            numeric_feat = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
            categorical_feat = ['sex', 'restecg', 'thal', 'slope', 'cp', 'exang', 'ca','fbs']
            

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)