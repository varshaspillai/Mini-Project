import os
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "resnet50_pcam.keras")
IMG_SIZE = 224

# IMPORTANT:
# This matches the common /255 preprocessing used in many PCam classifiers.
# If your Colab preprocessing used a different transformation, change this
# function to exactly match the training notebook.
MODEL_PREPROCESSING = "rescale_0_1"

_model = None

def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "resnet50_pcam.keras was not found. Copy the downloaded model "
                "into the same folder as app.py."
            )
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model

def preprocess_image(path):
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32)
    if MODEL_PREPROCESSING == "rescale_0_1":
        arr /= 255.0
    return arr

def find_feature_model(model):
    """
    Finds a suitable 4-D feature-producing layer.
    For the usual ResNet50 transfer-learning structure, the nested ResNet50
    model itself is an excellent Grad-CAM feature target.
    """
    # Prefer a nested model whose output is 4-D.
    for layer in reversed(model.layers):
        try:
            shape = tuple(layer.output.shape)
            if len(shape) == 4:
                return layer
        except Exception:
            pass

    # Fall back to a Conv2D layer exposed by the model.
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer

    raise RuntimeError("Could not find a 4-D feature layer for Grad-CAM.")

def make_gradcam(model, image_batch, target_index=None):
    feature_layer = find_feature_model(model)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[feature_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        feature_maps, predictions = grad_model(image_batch, training=False)

        pred_tensor = predictions
        if len(pred_tensor.shape) == 2 and pred_tensor.shape[-1] == 1:
            score = pred_tensor[:, 0]
        else:
            if target_index is None:
                target_index = int(tf.argmax(pred_tensor[0]))
            score = pred_tensor[:, target_index]

    grads = tape.gradient(score, feature_maps)
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))

    feature_maps = feature_maps[0]
    pooled_grads = pooled_grads[0]
    heatmap = tf.reduce_sum(feature_maps * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.maximum(tf.reduce_max(heatmap), tf.keras.backend.epsilon())

    return heatmap.numpy()

def save_gradcam(image_path, heatmap, output_path):
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError("Unable to read uploaded image.")

    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_uint8 = cv2.resize(
        heatmap_uint8,
        (original.shape[1], original.shape[0])
    )

    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.60, colored, 0.40, 0)
    cv2.imwrite(output_path, overlay)

def predict_image(image_path, heatmap_dir, filename):
    model = load_model()

    arr = preprocess_image(image_path)
    batch = np.expand_dims(arr, axis=0)

    prediction = model.predict(batch, verbose=0)
    prediction = np.asarray(prediction)

    # Supports sigmoid output shape (1,1) and binary softmax shape (1,2).
    if prediction.ndim == 2 and prediction.shape[1] == 1:
        positive_probability = float(prediction[0, 0])
    elif prediction.ndim == 2 and prediction.shape[1] >= 2:
        positive_probability = float(prediction[0, 1])
    else:
        positive_probability = float(np.squeeze(prediction))

    positive_probability = max(0.0, min(1.0, positive_probability))
    predicted_class = "Positive" if positive_probability >= 0.5 else "Negative"

    confidence = positive_probability if predicted_class == "Positive" else 1.0 - positive_probability

    heatmap = make_gradcam(model, batch)

    base = os.path.splitext(filename)[0]
    heatmap_filename = f"{base}_gradcam.jpg"
    heatmap_path = os.path.join(heatmap_dir, heatmap_filename)
    save_gradcam(image_path, heatmap, heatmap_path)

    return {
        "prediction": predicted_class,
        "confidence": round(confidence * 100, 2),
        "positive_probability": round(positive_probability * 100, 2),
        "heatmap_filename": heatmap_filename
    }
