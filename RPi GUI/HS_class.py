import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image
import os

def load_labels(label_path):
    with open(label_path, 'r') as file:
        labels = file.read().splitlines()
    return labels

def preprocess_image(image_path, input_size):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(input_size)
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)
    return img

def classify_image(model_path, image_path, label_path):
    # Load the TFLite model and allocate tensors
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess the image
    input_size = input_details[0]['shape'][1:3]
    image = preprocess_image(image_path, input_size)

    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], image)

    # Run the interpreter
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
   
    # Get the predicted label
    predicted_index = np.argmax(output_data)
    labels = load_labels(label_path)
    predicted_label = labels[predicted_index]
   
    return predicted_label
    
model_path = './Heart Sound2/model_unquant.tflite'
#image_path = './Heart Sound2/Normal/btbwar-Heart Sound 20-001.jpg'
image_path = os.getenv('IMAGE_PATH')
label_path = './Heart Sound/labels.txt'
prediction = classify_image(model_path, image_path, label_path)
print('Predicted label:', prediction)
