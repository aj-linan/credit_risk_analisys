from flask import render_template, request, Flask
import numpy as np
import pickle

app = Flask(__name__)

def load_object(filename):
    with open(''+filename ,'rb') as f:
        loaded = pickle.load(f)
    return loaded

model = load_object(r'C:\Users\AlbertJ\Documents\Data_scientist\Data_glacier\week_4\credit_risk_analisys\model.pkl')
#http://127.0.0.1:8000/paramms?param1=Albert

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods =['POST'])
def predict():
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    if prediction == 1:
        output = "There is a risk of default"
    else: 
        output = "There is no risk of default"

    return render_template('index.html', predict_text = output)

if __name__ == '__main__':
    app.run(debug = True,port=8000)