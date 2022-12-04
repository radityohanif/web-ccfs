import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from flask import Flask, render_template, jsonify, make_response, request, url_for

app = Flask(__name__)

@app.get("/") 
def index():
    return render_template("index.html")

@app.post("/api") 
def machine_learning_api():
    # Train Model
    bmi_data = pd.read_csv("bmi_and_life_expectancy.csv")
    bmi_model = LinearRegression()
    bmi_model.fit(bmi_data[["BMI"]], bmi_data[["Life expectancy"]])

    # Get data from request API
    bodyRequest = request.json

    # Request Validation
    if bodyRequest == None:
        result = {
            "message" : "Body Request tidak boleh kosong",
            "status_code" : 400
        }
        return make_response(jsonify(result), 400)
    
    elif "bmi" not in bodyRequest:
        result = {
            "message" : "Body Request tidak menyertakan nilai bmi",
            "status_code" : 400
        }
        return make_response(jsonify(result), 400)

    
    # Model Prediction Process
    prediction_result = bmi_model.predict([[
        bodyRequest["bmi"]
    ]])
    
    # Set Result 
    result = {
        "life_expectancy" : prediction_result[0][0],
        "message" : "Berikut hasil prediksi umur kamu berdasarkan nilai BMI sebesar " + str(bodyRequest["bmi"]),
        "status_code" : 200
    }
    return make_response(jsonify(result), 200)
    
if __name__ == "__main__":
    app.run(debug=True, port=8080)