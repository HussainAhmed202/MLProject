import os
import pickle
import sys

import dill
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score

from src.exception import CustomException
from src.logger import logging


def performance_metric_score(y_true, y_pred):
    """Calculate the average specificity and sensitvity
    from confusion matrix."""

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificty = tn / (tn + fp)
    sensitivity = tp / (tn + fn)  # recall
    return (specificty + sensitivity) / 2


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

        # Create a custom scorer for specificity
        model_scorer = make_scorer(performance_metric_score)

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            print(f"Applying grid search for {type(model).__name__}")
            logging.info(f"Applying grid search for {type(model).__name__}")

            gs = GridSearchCV(model, para, cv=5, scoring=model_scorer)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # # train_model_score = accuracy_score(y_train, y_train_pred)
            # train_model_score = cross_val_score(
            #     model, X_train, y_train, scoring="accuracy", cv=5
            # ).mean()
            train_model_score = cross_val_score(
                model, X_train, y_train, scoring=model_scorer, cv=5
            ).mean()

            print(
                f"{type(model).__name__} got the overall score on train data {train_model_score}"
            )

            # # test_model_score = accuracy_score(y_test, y_test_pred)
            test_model_score = cross_val_score(
                model, X_test, y_test, scoring=model_scorer, cv=5
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
