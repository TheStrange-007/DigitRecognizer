from flask import Flask, render_template, request, jsonify
import cv2
import base64
import io
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the Keras model
model = load_model("analysis\digits_classifier_cnn.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_image', methods=['POST'])
def predict_image():
    # Get the image data from the request
    data = request.get_json()
    canvas_data_url = data["image"]
    canvas_data = base64.b64decode(canvas_data_url.split(',')[1])
    nparr = np.frombuffer(canvas_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (28, 28))
    cv2.imwrite("digit.png", img)

    # Normalize the image
    img = img / 255.0

    # Reshape the image to match the input shape of the model
    img = img.reshape(1, 28, 28, 1)

    # Make the prediction
    probabilities = np.round(model.predict(img)[0] * 100)

    return f"{probabilities.tolist()}"

if __name__ == '__main__':
    app.run(debug=True)
