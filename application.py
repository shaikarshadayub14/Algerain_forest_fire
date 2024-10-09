from flask import Flask,request,app,jsonify,render_template,url_for

import numpy as np
import pandas as pd
import pickle

application=Flask(__name__)
app=application
scaler=pickle.load(open('scaler.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['Post'])

def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=model.predict(final_input)[0]
    return render_template("home.html",prediction_text="The FWI prediction is {}".format(output))

if __name__=='__main__':
    app.run(debug=True)