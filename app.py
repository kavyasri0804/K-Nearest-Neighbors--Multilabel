import flask
from flask import Flask, request, render_template
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
with open("iris_naive_bayes_model.pkl", "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)


@app.route("/")
def main_page():

    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:

        sepal_length = float(
            request.form["sepal_length"]
        )

        sepal_width = float(
            request.form["sepal_width"]
        )

        petal_length = float(
            request.form["petal_length"]
        )

        petal_width = float(
            request.form["petal_width"]
        )




        features = np.array([[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]])

        prediction = model.predict(features)[0]



        iris_mapping = {
            0: "Iris Setosa",
            1: "Iris Versicolor",
            2: "Iris Virginica"
        }

        prediction = iris_mapping.get(
            int(prediction),
            str(prediction)
        )


        # ----------------------------------------------------
        # SEND RESULT TO HTML
        # ----------------------------------------------------

        return render_template(
            "index.html",
            prediction_text=prediction
        )


    except Exception as e:

        return render_template(
            "index.html",
            prediction_text=f"Prediction Error: {str(e)}"
        )
if __name__ == "__main__":
    app.run(debug=True)
