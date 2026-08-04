from flask import Flask, render_template, jsonify

app = Flask(__name__)

@app.route("/")
def dashboard():
    return render_template("index.html")


@app.route("/live")
def live():
    return render_template("live.html")


@app.route("/offline")
def offline():
    return render_template("offline.html")


@app.route("/dataset")
def dataset():
    return render_template("dataset.html")


@app.route("/models")
def models():
    return render_template("models.html")
@app.route("/api/live")
def api_live():

    data = [
        {
            "time": "14:30:10",
            "src": "192.168.1.72",
            "dst": "142.250.183.78",
            "prediction": "Benign",
            "confidence": "98.75%"
        }
    ]

    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True)