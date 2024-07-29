from flask import Flask, request, jsonify, render_template
import requests
import json

app = Flask(__name__)

# Replace with your Azure ML endpoint and key
AZURE_ML_ENDPOINT = "https://classification-model-hxcoe.eastus.inference.ml.azure.com/score"
AZURE_ML_KEY = "4eR2d6w1ulwI3aeMTOje2S75uufferCU"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Formatting data as per Azure ML example
    request_data = {
        "input_data": {
            "columns": [
                "sepal length (cm)",
                "sepal width (cm)",
                "petal length (cm)",
                "petal width (cm)"
            ],
            "index": [0],  # Assuming a single prediction, index can be a single value
            "data": [[
                data['sepal_length'],
                data['sepal_width'],
                data['petal_length'],
                data['petal_width']
            ]]
        }
    }

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {AZURE_ML_KEY}',
        'azureml-model-deployment': 'classification-model-1'
    }

    response = requests.post(AZURE_ML_ENDPOINT, headers=headers, data=json.dumps(request_data))
    
    # Print the response for debugging
    print(f"Response status code: {response.status_code}")
    print(f"Response content: {response.content}")
    
    prediction = response.json()
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True)