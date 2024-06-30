import datetime

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, url_for
from flask_cors import cross_origin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from src.pipelines.predict_pipeline import CustomData,PredictPipeline

app = Flask(__name__, template_folder="template")


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
    if request.method=='GET':
        return render_template('predictor.html')
    # if request.method == "GET":
    #     # # DATE
    #     # date = request.form["date"]
    #     # day = float(pd.to_datetime(date, format="%Y-%m-%d").day)
    #     # month = float(pd.to_datetime(date, format="%Y-%m-%d").month)
    #     # # MinTemp
    #     # minTemp = float(request.form["mintemp"])
    #     # # MaxTemp
    #     # maxTemp = float(request.form["maxtemp"])
    #     # # Rainfall
    #     # rainfall = float(request.form["rainfall"])
    #     # # Evaporation
    #     # evaporation = float(request.form["evaporation"])
    #     # # Sunshine
    #     # sunshine = float(request.form["sunshine"])
    #     # # Wind Gust Speed
    #     # windGustSpeed = float(request.form["windgustspeed"])
    #     # # Wind Speed 9am
    #     # windSpeed9am = float(request.form["windspeed9am"])
    #     # # Wind Speed 3pm
    #     # windSpeed3pm = float(request.form["windspeed3pm"])
    #     # # Humidity 9am
    #     # humidity9am = float(request.form["humidity9am"])
    #     # # Humidity 3pm
    #     # humidity3pm = float(request.form["humidity3pm"])
    #     # # Pressure 9am
    #     # pressure9am = float(request.form["pressure9am"])
    #     # # Pressure 3pm
    #     # pressure3pm = float(request.form["pressure3pm"])
    #     # # Temperature 9am
    #     # temp9am = float(request.form["temp9am"])
    #     # # Temperature 3pm
    #     # temp3pm = float(request.form["temp3pm"])
    #     # # Cloud 9am
    #     # cloud9am = float(request.form["cloud9am"])
    #     # # Cloud 3pm
    #     # cloud3pm = float(request.form["cloud3pm"])
    #     # # Cloud 3pm
    #     # location = int(request.form["location"])
    #     # # Wind Dir 9am
    #     # winddDir9am = int(request.form["winddir9am"])
    #     # # Wind Dir 3pm
    #     # winddDir3pm = int(request.form["winddir3pm"])
    #     # # Wind Gust Dir
    #     # windGustDir = int(request.form["windgustdir"])
    #     # # Rain Today
    #     # rainToday = int(request.form["raintoday"])

    #     # input_lst = [
    #     #     location,
    #     #     minTemp,
    #     #     maxTemp,
    #     #     rainfall,
    #     #     evaporation,
    #     #     sunshine,
    #     #     windGustDir,
    #     #     windGustSpeed,
    #     #     winddDir9am,
    #     #     winddDir3pm,
    #     #     windSpeed9am,
    #     #     windSpeed3pm,
    #     #     humidity9am,
    #     #     humidity3pm,
    #     #     pressure9am,
    #     #     pressure3pm,
    #     #     cloud9am,
    #     #     cloud3pm,
    #     #     temp9am,
    #     #     temp3pm,
    #     #     rainToday,
    #     #     month,
    #     #     day,
    #     # ]
    #     print(input_lst)

    #     input_lst = preprocessor(input_lst)

    #     model = load_model()
    #     pred = model.predict(input_lst)
    #     if pred[0][0] > 0.5:
    #         return render_template("rain.html")
    #     else:
    #         return render_template("sunny.html")
    
    else:
        data=CustomData(
            # gender=request.form.get('gender'),
            # race_ethnicity=request.form.get('ethnicity'),
            # parental_level_of_education=request.form.get('parental_level_of_education'),
            # lunch=request.form.get('lunch'),
            # test_preparation_course=request.form.get('test_preparation_course'),
            # reading_score=float(request.form.get('writing_score')),
            # writing_score=float(request.form.get('reading_score'))
            # 
            age = request.form.get('age'),
            sex = request.form.get('sex'),
            cp = request.form.get('cp'),
            trestbps = request.form.get('trestbps'),
            chol = request.form.get('chol'),
            fbs = request.form.get('fbs'),
            restecg = request.form.get('restecg'),
            thalach = request.form.get('thalach'),
            exang = request.form.get('exang'),
            oldpeak = float(request.form.get('oldpeak')),
            slope = request.form.get('slope'),
            ca = float(request.form.get('ca')),
            thal = float(request.form.get('thal'))
            # 

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('predictor.html',results=results[0])
    

    
    # return render_template("predictor.html")
if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)