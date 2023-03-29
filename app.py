from flask import Flask, render_template, request

app = Flask(__name__, template_folder='templates', static_folder='static')
app.debug = True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/visualize')
def visualize():
    return render_template('visualize.html')