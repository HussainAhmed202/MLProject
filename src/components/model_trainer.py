import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            models = {
                "SVM": SVC(random_state=42, probability=True),
                "KNN": KNeighborsClassifier(),
                "LR": LogisticRegression(random_state=42),
                "DT": DecisionTreeClassifier(random_state=42),
                "RF": RandomForestClassifier(random_state=42),
                "GaussainNB": GaussianNB(),
            }
            params = {
                "DT": {
                    "max_depth": [None, 5, 10, 15],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "criterion": ["gini", "entropy"],
                },
                "RF": {
                    "n_estimators": [10, 50, 100],
                    "max_depth": [None, 2, 5],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2, 3],
                    "criterion": ["gini"],
                },
                "SVM": {
                    "C": [0.0011, 0.005, 0.01, 0.05, 0.1, 1, 10, 20],
                    "kernel": ["linear", "rbf"],
                    "gamma": ["scale", "auto", 0.1, 0.5, 1, 5],
                    "degree": [1, 2, 3],
                },
                "KNN": {
                    "n_neighbors": list(range(1, 15)),
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                    "p": [1, 2],  # 1: Manhattan distance, 2: Euclidean distance
                },
                "LR": {
                    "C": [0.001, 0.01, 0.1, 1, 10, 100],
                    "max_iter": [100, 200, 300, 400, 500],
                },
                "GaussainNB": {"var_smoothing": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]},
            }

            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
            )

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))
            print(best_model_score)

            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            print(best_model)

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            # predicted = best_model.predict(X_test)
            # accuracy = accuracy_score(y_test, predicted)
            accuracy = cross_val_score(best_model, X_train, y_train, cv=5).mean()
            return accuracy

        except Exception as e:
            raise CustomException(e, sys)
