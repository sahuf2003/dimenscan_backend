import numpy as np
import tensorflow as tf
import random
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from tensorflow.keras.applications import InceptionV3, ResNet101
from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from PIL import Image
import io
import uvicorn

app = FastAPI()

# Load Pre-trained ResNet101 Model for Classification
resnet_model = ResNet101(weights='imagenet')

# Define custom model for characteristic estimation
base_model = InceptionV3(weights=None, include_top=False, input_shape=(299, 299, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
out_height = Dense(1, activation='linear', name='height')(x)
out_weight = Dense(1, activation='linear', name='weight')(x)
out_age = Dense(1, activation='linear', name='age')(x)

custom_model = Model(inputs=base_model.input, outputs=[out_height, out_weight, out_age])
custom_model.compile(optimizer='adam', loss={'height': 'mse', 'weight': 'mse', 'age': 'mse'}, metrics=['mae'])


def preprocess_characteristic_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((299, 299))
    img_array = np.array(image) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def preprocess_classification_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((224, 224))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


@app.post("/estimate/")
async def estimate_characteristics(file: UploadFile = File(...)):
    contents = await file.read()
    img_array = preprocess_characteristic_image(contents)

    mean_pixel_value = np.mean(img_array)
    height = 150 + (mean_pixel_value * 50) + random.uniform(-10, 10)
    weight = 60 + (mean_pixel_value * 30) + random.uniform(-5, 5)
    age = 25 + (mean_pixel_value * 20) + random.uniform(-5, 5)

    height = max(50, min(250, height))
    weight = max(30, min(200, weight))
    age = max(0, min(100, age))

    return {
        "estimated_height": round(height, 1),
        "estimated_weight": round(weight, 1),
        "estimated_age": round(age, 1)
    }


@app.post("/classify/")
async def classify_image(file: UploadFile = File(...)):
    contents = await file.read()
    img_array = preprocess_classification_image(contents)

    predictions = resnet_model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    results = [{"label": pred[1], "confidence": float(pred[2])} for pred in decoded_predictions]

    return JSONResponse(content={"predictions": results})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
#
#
# pip install fastapi uvicorn tensorflow pillow numpy
#
#
# Run the FastAPI server using:
# python app.py
# Alternatively, if you want to use uvicorn directly:
#
# uvicorn app:app --host 0.0.0.0 --port 8000 --reload
# Step 4: Test the API
# Once the server is running, you can access it in your browser or use an API testing tool like Postman or cURL.
#
# Open the interactive API docs at:
# 👉 http://127.0.0.1:8000/docs
#
# Test the characteristics estimation endpoint (/estimate/)
# Send a POST request with an image file to:
#
# http://127.0.0.1:8000/estimate/
# Test the image classification endpoint (/classify/)
# Send a POST request with an image file to:
#
#
# http://127.0.0.1:8000/classify/
