import datetime

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, url_for
from flask_cors import cross_origin
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.pipelines.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)


def load_model():
    """Loads the model from the specified model path."""

    MODEL_PATH = "./models/model.pkl"
    model = joblib.load(open(MODEL_PATH, "rb"))
    print("Model Loaded")
    return model


def preprocessor(input_lst: list) -> np.ndarray:
    """Prepares the input user data for the model"""

    SCALAR_PATH = "prep.pkl"

    # convert into 2D numpy array for scaling
    input_lst = np.array(input_lst).reshape(1, -1)

    scalar = joblib.load(open(SCALAR_PATH, "rb"))
    print("Scaler Loaded")

    return scalar.transform(input_lst)


@app.route("/", methods=["GET"])
@cross_origin()
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "GET":
        return render_template("predictor.html")
    else:
        data = CustomData(
            age=request.form.get("age"),
            sex=request.form.get("sex"),
            cp=request.form.get("cp"),
            trestbps=request.form.get("trestbps"),
            chol=request.form.get("chol"),
            fbs=request.form.get("fbs"),
            restecg=request.form.get("restecg"),
            thalach=request.form.get("thalach"),
            exang=request.form.get("exang"),
            oldpeak=float(request.form.get("oldpeak")),
            slope=request.form.get("slope"),
            ca=float(request.form.get("ca")),
            thal=float(request.form.get("thal")),
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template("predictor.html", results=results[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
