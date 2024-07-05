import os
import pickle
import sys

import dill
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            print(f"Applying grid search for {type(model).__name__}")
            logging.info(f"Applying grid search for {type(model).__name__}")

            gs = GridSearchCV(model, para, cv=5, scoring="recall")
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            train_model_score = cross_val_score(
                model, X_train, y_train, scoring="recall", cv=5
            ).mean()

            print(
                f"{type(model).__name__} got the overall score on train data {train_model_score}"
            )

            test_model_score = cross_val_score(
                model, X_test, y_test, scoring="recall", cv=5
            ).mean()

            print(
                f"{type(model).__name__} got the overall score on test data {test_model_score}"
            )

            report[list(models.keys())[i]] = test_model_score
            print(report)
        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
