from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    age = float(request.form["age"])
    bmi = float(request.form["bmi"])
    children = int(request.form["children"])
    smoker = int(request.form["smoker"])

    data = np.array([[age, bmi, children, smoker]])
    prediction = model.predict(data)[0]

    return render_template(
        "index.html",
        result=f"Predicted Medical Cost: ₹ {round(prediction, 2)}"
    )

if __name__ == "__main__":
    app.run(debug=True)
