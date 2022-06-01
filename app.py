from flask import Flask, request, jsonify
import numpy as np
import pickle

model = pickle.load(open('covid_prediction_model.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return "Hello world"


@app.route('/predict', methods=['POST'])
def predict():
    fever = request.form.get('fever')
    bodypain = request.form.get('bodypain')
    age = request.form.get('age')
    runnynose = request.form.get('runnynose')
    diffbreath = request.form.get('diffbreath')

    input_query = np.array([[fever, bodypain, age, runnynose, diffbreath]])

    result = model.predict(input_query)[0]

    return jsonify({'Suspected': str(result)})


if __name__ == '__main__':
    app.run()
    # app.run(host='localhost', port=8080)
