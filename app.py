import datetime
import sys

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, url_for
from flask_cors import cross_origin

from src.pipelines.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)


@app.route("/", methods=["GET"])
@cross_origin()
def home():
    return render_template("index.html")


@app.route("/predict2", methods=["GET", "POST"])
@cross_origin()
def predict2():
    if request.method == "POST":
        patient_id = request.form.get("patient_id")
        patient_name = request.form.get("patient_name")
        data = CustomData(
            age=int(request.form.get("age")),
            sex=int(request.form.get("sex")),
            cp=int(request.form.get("cp")),
            trestbps=int(request.form.get("trestbps")),
            chol=int(request.form.get("chol")),
            fbs=int(request.form.get("fbs")),
            restecg=int(request.form.get("restecg")),
            thalach=int(request.form.get("thalach")),
            exang=int(request.form.get("exang")),
            oldpeak=float(request.form.get("oldpeak")),
            slope=int(request.form.get("slope")),
            ca=float(request.form.get("ca")),
            thal=float(request.form.get("thal")),
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.custom_predict(pred_df)
        print("after Prediction")

        # Check the prediction result and redirect accordingly
        has_disease = (
            results[0] == 1
        )  # Assuming '1' means the patient has heart disease
        if has_disease:
            return redirect(
                url_for(
                    "heart_disease", patient_id=patient_id, patient_name=patient_name
                )
            )
        else:
            return redirect(
                url_for(
                    "no_heart_disease", patient_id=patient_id, patient_name=patient_name
                )
            )
    return render_template("predictor2.html")


@app.route("/predict", methods=["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        patient_id = request.form.get("patient_id")
        patient_name = request.form.get("patient_name")
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

        # Check the prediction result and redirect accordingly
        has_disease = (
            results[0] == 1
        )  # Assuming '1' means the patient has heart disease
        if has_disease:
            return redirect(
                url_for(
                    "heart_disease", patient_id=patient_id, patient_name=patient_name
                )
            )
        else:
            return redirect(
                url_for(
                    "no_heart_disease", patient_id=patient_id, patient_name=patient_name
                )
            )
    return render_template("predictor.html")


@app.route("/heart_disease")
def heart_disease():
    patient_id = request.args.get("patient_id")
    patient_name = request.args.get("patient_name")
    return render_template(
        "heart_disease.html", patient_id=patient_id, patient_name=patient_name
    )


@app.route("/no_heart_disease")
def no_heart_disease():
    patient_id = request.args.get("patient_id")
    patient_name = request.args.get("patient_name")
    return render_template(
        "no_heart_disease.html", patient_id=patient_id, patient_name=patient_name
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
