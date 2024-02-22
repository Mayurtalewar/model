from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('data.csv')

# Define LabelEncoder
le_interest = LabelEncoder()
le_branch = LabelEncoder()

# Fit LabelEncoders on all possible values
le_interest.fit(data['Interest'])
le_branch.fit(data['Branch'])

# Preprocess the data
data['Interest'] = le_interest.transform(data['Interest'])
data['Branch'] = le_branch.transform(data['Branch'])

# Split features and target variable
X = data[['Interest', 'Branch', 'Members']]
y = data['Prediction']

# Train the machine learning model
model = RandomForestClassifier()
model.fit(X, y)

# Define routes
@app.route('/')
def index():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the form
    interest = request.form['interest']
    branch = request.form['branch']
    members = int(request.form['members'])

    # Preprocess input data if necessary
    interest_encoded = le_interest.transform([interest])[0]
    branch_encoded = le_branch.transform([branch])[0]

    # Use machine learning model for prediction
    prediction = model.predict([[interest_encoded, branch_encoded, members]])[0]

    # Render the results template with the prediction
    return render_template('results.html', prediction=prediction)

# if __name__ == '__main__':
#     app.run(debug=True)