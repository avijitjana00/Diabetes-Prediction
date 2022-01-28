from flask import Flask, render_template, request
import pickle
import numpy as np

Pkl_Filename = 'dp.pkl'
classifier = pickle.load(open(Pkl_Filename, 'rb'))


app = Flask(__name__)

@app.route('/')
def front_page():
    return render_template('Input_Page.html')

@app.route('/predict', methods= ['POST'])
def predict():
    if request.method == 'POST':
        pg = request.form['pregnancies']
        gl = request.form['glucose']
        bp = request.form['bloodpressure']
        st = request.form['skinthickness']
        isu = request.form['insulin']
        bmi = request.form['bmi']
        dpf = request.form['dpf']
        age = request.form['age']

        store = np.array([[pg, gl, bp, st, isu, bmi, dpf, age]])

        pred = classifier.predict(store)

        return render_template('Output_Page.html', prediction = pred)

if __name__ == '__main__':
    app.run(debug=True)

