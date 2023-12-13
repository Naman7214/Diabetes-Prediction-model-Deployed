from flask import Flask, request, render_template
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the trained model
model = load_model('diabetes_model.h5')

# Load the MinMaxScaler for 'BMI'
scaler = MinMaxScaler()

# Load the training data used for scaling
X_train = pd.read_csv("my_dataframe.csv")
scaler.fit(X_train[['BMI']])

# Create a function to preprocess and format input data
def preprocess_input(data):
    # Convert binary categorical features from "Yes"/"No" to 1/0
    binary_features = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 'HeartDiseaseorAttack',
                        'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk']
    for feature in binary_features:
        data[feature] = 1 if data[feature] == 'Yes' else 0

    data['Sex'] = 1 if data['Sex'].lower() == 'male' else 0

    # Convert 'GenHlth' to binary values
    gen_hlth_mapping = {'excellent': 1, 'very good': 2, 'good': 3, 'fair': 4, 'poor': 5}
    data['GenHlth'] = gen_hlth_mapping.get(data['GenHlth'].lower(), 0)

    # Convert other features to numeric (handle errors for non-numeric inputs)
    numeric_features = ['BMI', 'MentHlth', 'PhysHlth', 'Age']
    for feature in numeric_features:
        data[feature] = pd.to_numeric(data[feature], errors='coerce')

    # Create a DataFrame from the dictionary
    data_df = pd.DataFrame([data])

    # Handle missing values if needed
    data_df.fillna(0, inplace=True)

    # Ensure the order and structure of features match the training data
    return data_df


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect data from the user input in the backend
    user_input = request.form.to_dict()

    # Preprocess and format the input data
    input_data = preprocess_input(user_input)

    # Make predictions using the model
    predictions = model.predict(input_data)

    predictions = (predictions >= 0.5).astype(int)

    # Return the results to the user interface
    return render_template('result.html', prediction=predictions[0][0])

if __name__ == '__main__':
    app.run(debug=True)
