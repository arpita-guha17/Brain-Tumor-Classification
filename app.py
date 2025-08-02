import gradio as gr
import numpy as np
import logging
from tensorflow.keras.models import load_model
from PIL import Image

# -------- Load Model --------
try:
    model = load_model("model_03_base.h5")
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error("Error loading model: %s", e)
    raise e

# -------- Preprocessing Function --------
def preprocess_image(img):
    img = img.resize((240, 240))  # Match input shape from model
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# -------- Prediction Function --------
def predict_image(img):
    try:
        logging.info("Received image for prediction.")
        processed = preprocess_image(img)
        pred = model.predict(processed)[0]  # Softmax gives 2 probabilities
        class_index = np.argmax(pred)

        result = "Tumor Detected ✅" if class_index == 1 else "No Tumor ❌"
        logging.info(f"Prediction: {pred} → {result}")
        return result
    except Exception as e:
        logging.error("Prediction error: %s", e)
        return "Error processing the image."

# -------- Gradio UI --------
app = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload Brain MRI"),
    outputs=gr.Textbox(label="Prediction"),
    title="🧠 Brain Tumor Classifier",
    description="Upload an MRI brain image to check if it's tumorous or not.",
    theme="default"
)

# -------- Launch App --------
if __name__ == "__main__":
    app.launch()