import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from src.algorithms import KNNTransformer, LRTransformer
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    trained_model_file_path2 = os.path.join("artifacts", "model2.pkl")


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
                "KNN": KNeighborsClassifier(),
                "LR": LogisticRegression(random_state=42),
            }
            params = {
                "KNN": {
                    "n_neighbors": list(range(1, 20, 2)),
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                    "metric": ["minkowski", "euclidean", "manhattan"],
                    "leaf_size": list(range(1, 50, 5)),
                },
                "LR": {
                    "C": [0.001, 0.01, 0.1, 1, 10, 100],
                    "max_iter": [100, 200, 300, 400, 500],
                },
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

            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_name == "LR":
                lr = LRTransformer()
                best_custom_model = lr.fit(X_train, y_train)
            else:
                knn = KNNTransformer(n_neigbours=5)
                best_custom_model = knn.fit(X_train, y_train)

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path2,
                obj=best_custom_model,
            )

            return best_model, best_custom_model

        except Exception as e:
            raise CustomException(e, sys)
