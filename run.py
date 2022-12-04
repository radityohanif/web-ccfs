import random
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from flask import Flask, render_template, jsonify, make_response, request, url_for

app = Flask(__name__)

@app.get("/") 
def index():
    return render_template("index.html")

def get_life_expectancy(bmi_input):
    # Train Model
    bmi_data = pd.read_csv("bmi_and_life_expectancy.csv")
    bmi_model = LinearRegression()
    bmi_model.fit(bmi_data[["BMI"]], bmi_data[["Life expectancy"]])
    
    # Model Prediction Process
    prediction_result = bmi_model.predict([[
        bmi_input
    ]])

    return prediction_result[0][0]


@app.post("/testing/api") 
def testing_machine_learning_api():
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
    
    # Set Result 
    result = {
        "life_expectancy" : get_life_expectancy(bmi_input=bodyRequest["bmi"]),
        "message" : "Berikut hasil prediksi umur kamu berdasarkan nilai BMI sebesar " + str(bodyRequest["bmi"]),
        "status_code" : 200
    }
    return make_response(jsonify(result), 200)

@app.get("/api") 
def machine_learning_api():
    random_bmi_value = random.randrange(12, 40)
    result = {
        "life_expectancy" : get_life_expectancy(bmi_input=random_bmi_value),
        "message" : "Berikut hasil prediksi umur kamu berdasarkan nilai BMI sebesar " + str(random_bmi_value),
        "status_code" : 200
    }
    return make_response(jsonify(result), 200)

if __name__ == "__main__":
    app.run(debug=True, port=8082)