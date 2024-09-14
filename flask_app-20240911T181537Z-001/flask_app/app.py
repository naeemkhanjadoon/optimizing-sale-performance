from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS
import pandas as pd


app = Flask(__name__)
CORS(app)

# Load the model (replace 'path_to_model.pkl' with the actual model path)
model_path = './random_forest_model.pkl'
model = joblib.load(model_path)

@app.route('/', methods=['POST'])
def index():
    return("<h1>API is Working</h1>")

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the request
    data = request.json
    # Convert the data into the appropriate format for the model
    # input_data = np.array([[
    #     float(data['Administrative_Duration']),
    #     float(data['Informational_Duration']),
    #     float(data['ProductRelated_Duration']),
    #     float(data['BounceRates']),
    #     float(data['ExitRates']),
    #     float(data['PageValues']),
    #     float(data['SpecialDay']),
    #     data['Month'],
    #     int(data['OperatingSystems']),
    #     int(data['Browser']),
    #     int(data['Region']),
    #     int(data['TrafficType']),
    #     data['VisitorType'],
    #     data['Weekend']
    # ]])

    # Create a DataFrame from the user input dictionary
    user_df = pd.DataFrame.from_dict([data])

    # Map VisitorType to binary columns
    visitor_type_mapping = {
        'New_Visitor': 1,
        'Other': 0,
        'Returning_Visitor': 0
    }
    user_df['VisitorType_New_Visitor'] = user_df['VisitorType'].map(lambda x: 1 if x == 'New_Visitor' else 0)
    user_df['VisitorType_Other'] = user_df['VisitorType'].map(lambda x: 1 if x == 'Other' else 0)
    user_df['VisitorType_Returning_Visitor'] = user_df['VisitorType'].map(lambda x: 1 if x == 'Returning_Visitor' else 0)

    # Drop the original VisitorType column
    user_df.drop('VisitorType', axis=1, inplace=True)

    # Convert the DataFrame to a format suitable for the model
    input_data = user_df.values

    # Make a prediction
    prediction = model.predict(input_data)

    # Return the prediction
    if prediction[0] == 0:
        prediction_result = "The Customer is not likely to buy products"
    else:
        prediction_result = "The Customer is likely to buy products"
    
    return jsonify({'prediction': prediction_result})

    # # Make a prediction
    # prediction = model.predict(input_data)

    # # Return the prediction
    # return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
