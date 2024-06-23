from flask import Flask, request, render_template,redirect,url_for
import joblib
from copy import deepcopy
import pandas as pd

app = Flask(__name__)

encoder = joblib.load('encoder.joblib')
scaler = joblib.load('scaler.joblib')
model = joblib.load('model.joblib')

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')




@app.route('/predict', methods=['GET', 'POST'])
def predict():

    if request.method == 'POST':
        # Extract features from request form data
        Gender = request.form.get('Gender')
        Has_a_car = request.form.get('Has_a_car')
        Has_a_property = request.form.get('Has_a_property')
        Income = request.form.get('Income')
        Employment_status = request.form.get('Employment_status')
        Education_level = request.form.get('Education_level')
        Marital_status = request.form.get('Marital_status')
        Dwelling = request.form.get('Dwelling')
        Age = request.form.get('Age')
        Employment_length= request.form.get('Employment_length')
        Family_member_count = request.form.get('Family_member_count')


        data={
            'Gender':[Gender],'Has_a_car':[Has_a_car],'Has_a_property':[Has_a_property],
            'Income':[Income],'Employment_status':[Employment_status],'Education_level':[Education_level],
            'Marital_status':[Marital_status],'Dwelling':[Dwelling],'Age':[Age],
            'Employment_length':[Employment_length],'Family_member_count':[Family_member_count]

        }
        df=pd.DataFrame(data)

        features = ['Gender', 'Has_a_car', 'Has_a_property', 'Employment_status', 'Education_level', 'Marital_status',
                    'Dwelling']

        x_data_test = encoder.transform(df[features]).toarray()
        columns_data = encoder.get_feature_names_out()
        X_DT = pd.DataFrame(x_data_test, columns=columns_data)


        X_t = deepcopy(df)
        X_t.drop(columns=features, inplace=True)
        X_t = X_t.reset_index()

        X_test1 = pd.concat([X_DT, X_t], axis=1)
        X_test1.drop('index', axis=1, inplace=True)

        X_test1[['Age', 'Income', 'Employment_length']] = scaler.transform(X_test1[['Age', 'Income', 'Employment_length']])
        X_test1.drop(columns=['Gender_F', 'Has_a_car_N', 'Has_a_property_N'], inplace=True)

        # Make prediction
        prediction = model.predict(X_test1)[0]

        if prediction==0:
            pred='NO RISK'
        else:
            pred='THERE IS HIGH RISK'




    return redirect(url_for('result',pred=pred))

@app.route('/result', methods=['GET', 'POST'])
def result():

    return render_template('result.html', prediction=request.args.get('pred'))

if __name__ == '__main__':
    app.run(debug=True)