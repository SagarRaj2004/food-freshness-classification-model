import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys
import os

def predict_food(img_path):
    # 1. Load the model we just trained
    model_path = '../models/food_classifier.keras'
    
    if not os.path.exists(model_path):
        print("Error: Trained model file not found in models folder!")
        return

    model = tf.keras.models.load_model(model_path)
    
    # 2. Process the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 3. Get Prediction
    preds = model.predict(img_array)
    
    # CRITICAL: These must match your folder names in 'dataset/train'
    labels = ['Fresh', 'Rotten'] 
    
    result = labels[np.argmax(preds)]
    confidence = np.max(preds) * 100
    
    print(f"\n--- Prediction Results ---")
    print(f"Status: {result}")
    print(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict_food(sys.argv[1])
    else:
        print("Usage: python predict.py your_image_path.jpg")