import tensorflow as tf
from tensorflow.keras import layers, models
import os
from PIL import Image

# 1. Path Setup
base_dir = 'dataset'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# --- UPDATED: SCRIPT TO SAFELY FIND AND DELETE BROKEN IMAGES ---
def verify_images(directory):
    print(f"Checking images in {directory}...")
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                try:
                    # 'with' ensures the file is CLOSED before we try to delete it
                    with Image.open(img_path) as img:
                        img.verify() 
                except Exception:
                    print(f"Found broken image, attempting to remove: {img_path}")
                    try:
                        os.remove(img_path)
                        print("Successfully removed.")
                    except Exception as e:
                        print(f"Could not remove {img_path}: {e}")

verify_images(base_dir)

# 2. Modern Data Loading
print("Loading dataset...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical'
)
# ... (rest of the code stays the same)
# 3. Build Model
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False 

model = models.Sequential([
    base_model,
    layers.Rescaling(1./255), # Added rescaling here instead of generator
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(2, activation='softmax') 
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Start Training
print("Starting Training...")
model.fit(train_ds, validation_data=val_ds, epochs=5)

# 5. Save (Using the recommended .keras format)
if not os.path.exists('models'): os.makedirs('models')
model.save('models/food_classifier.keras')
print("Success! Model saved in models folder as food_classifier.keras")