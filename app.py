from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler



app = Flask(__name__, template_folder='templates')


@app.route("/")
def home():
    return render_template("index.html")


model_path = 'c:\\Users\\raman\\Data Science\\EAP\\naivebayes_pickle.pkl'
model = pickle.load(open(model_path, 'rb'))

# -----------------------------------------------------
# with open('naivebayes_pickle.pkl' , 'rb') as model:
#     lr = pickle.load(model)

# @app.route('/',methods=['GET'])
# def Home():
#     return render_template('index.html')
# -----------------------------------------------------

standard_to = StandardScaler()

@app.route("/predict", methods = ['POST'])
def predict():
    
    if(request.method == 'POST'):

        Age = int(request.form['Age'])
        Distance_from_home = int(request.form['Distance_from_home'])
        EnvironmentSatisfaction = int(request.form['EnvironmentSatisfaction'])
        JobInvolvement = int(request.form['JobInvolvement'])
        JobLevel = int(request.form['JobLevel'])
        JobRole = int(request.form['JobRole'])
        JobSatisfaction = int(request.form['JobSatisfaction'])
        MaritalStatus = int(request.form['MaritalStatus'])
        MonthlyIncome = int(request.form['MonthlyIncome'])
        OverTime = int(request.form['OverTime'])
        StockOptionLevel = int(request.form['StockOptionLevel'])
        TotalWorkingYears = int(request.form['TotalWorkingYears'])
        YearsAtCompany = int(request.form['YearsAtCompany'])
        YearsInCurrentRole = int(request.form['YearsInCurrentRole'])
        YearsWithCurrManager = int(request.form['YearsWithCurrManager'])

       
        result = model.predict([[Age,Distance_from_home , EnvironmentSatisfaction,JobInvolvement,JobLevel,JobRole,JobSatisfaction,MaritalStatus,MonthlyIncome,OverTime,StockOptionLevel,TotalWorkingYears,YearsAtCompany,YearsInCurrentRole,YearsWithCurrManager]])

        # prediction=model.predict([[Age,Distance_from_home , EnvironmentSatisfaction,JobInvolvement,JobLevel,JobRole,JobSatisfaction,MaritalStatus,MonthlyIncome,OverTime,StockOptionLevel,TotalWorkingYears,YearsAtCompany,YearsInCurrentRole,YearsWithCurrManager ]])
        
        return render_template('results.html', result = result)


if __name__=="__main__":
    app.run(debug=True)
