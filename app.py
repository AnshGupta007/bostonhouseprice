import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json()   # ✅ get full JSON

    print("Raw JSON:", data)

    # If your JSON looks like: { "data": { ...features... } }
    input_data = data['data']   # get inner dictionary

    print("Feature values:", input_data)

    values = np.array(list(input_data.values())).reshape(1, -1)

    print("Array shape:", values)

    new_data = scalar.transform(values)
    output = regmodel.predict(new_data)

    return jsonify({"prediction": float(output[0])})

@app.route('/predict', methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = regmodel.predict(final_input)[0]
    return render_template("home.html", prediction_text="The predicted price of the house is {}".format(output))
if __name__ == "__main__":
    app.run(debug=True)