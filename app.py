from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Model
filename = 'model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['Age'])
        obesity = int(request.form['Obesity'])
        gender = int(request.form['Gender'])
        polyuria = int(request.form['Polyuria'])
        polydipsia = int(request.form['Polydipsia'])
        s_w_l = int(request.form['sudden_weight_loss'])
        weakness = int(request.form['weakness'])
        polyphagia = int(request.form['Polyphagia'])
        g_t = int(request.form['Genital_thrush'])
        v_b = int(request.form['visual_blurring'])
        itching = int(request.form['Itching'])
        irritability = int(request.form['Irritability'])
        d_h = int(request.form['delayed_healing'])
        p_p = int(request.form['partial_paresis'])
        m_s = int(request.form['muscle_stiffness'])
        alopecia = int(request.form['Alopecia'])


        input_data = np.array([[age, gender, polyuria, polydipsia, s_w_l, weakness, polyphagia, g_t, v_b, itching, irritability, d_h, p_p, m_s, alopecia, obesity]])

        my_prediction = classifier.predict(input_data)

        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)