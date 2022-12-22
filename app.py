import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_application = Flask(__name__)
flask_application._static_folder = 'templates'
model = pickle.load(open("model.pkl", "rb"))

@flask_application.route("/")
def Home():
    return render_template("index.html")

@flask_application.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = "The Most Suitable Crop Is {}".format(prediction))

if __name__ == "__main__":
    flask_application.run(debug=True)