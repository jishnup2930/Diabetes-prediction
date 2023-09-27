from flask import Flask,render_template,request,redirect
from Diabetes import model,scaler
import numpy as np

app = Flask(__name__)

@app.route('/',methods=['GET'])
def home():
    return render_template('home.html')
@app.route('/predict',methods=['GET'])
def prediction():
    return render_template('predict.html')
@app.route('/result',methods=['POST'])
def result():
    if request.method == 'POST':
        try:
            preg = float(request.form['f1'])
            glue = float(request.form['f2'])
            bp = float(request.form['f3'])
            st = float(request.form['f4'])
            ins = float(request.form['f5'])
            bmi = float(request.form['f6'])
            dpf = float(request.form['f7'])
            age = float(request.form['f8'])

            #prediction process
            out=np.array([[preg,glue,bp,st,ins,bmi,dpf,age]])
            print(out)
            scaled_data=scaler.transform(out)
            print(scaled_data)
            pred=model.predict(scaled_data)
            outcome = " "
            if pred==1:
                outcome='Positive' +' -Your health is in trouble'
            else:
                outcome='Negative' + '-You are healthy'
            return render_template('predict.html',output=outcome)
        except Exception as e:
            print('The exception is ',e)
            return 'Something is wrong'
    else:
        return render_template('predict.html')
if __name__=='__main__':
    app.run(debug=True)
