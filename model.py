import os
import numpy as np
import tensorflow as tf


# ============================================================
# PATHOVISION AI MODEL
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "resnet50_pcam.keras"
)


print("Loading PathoVision AI model...")


model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)


print("PathoVision AI model loaded successfully.")
print("Model input:", model.input_shape)
print("Model output:", model.output_shape)


# ============================================================
# PREDICTION
# ============================================================

def predict_image(image_array):

    """
    Takes a preprocessed image with shape:

        (1, 224, 224, 3)

    and returns prediction information.
    """

    prediction = model.predict(
        image_array,
        verbose=0
    )

    probability = float(
        np.squeeze(prediction)
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # The exact meaning of 0 and 1 depends on the label
    # mapping used when your model was trained.
    #
    # We will verify this before final deployment.
    # --------------------------------------------------------

    if probability >= 0.5:

        predicted_class = "Abnormal / Tumor Suspected"

        confidence = probability * 100

    else:

        predicted_class = "Normal / Non-Tumor"

        confidence = (1 - probability) * 100


    return {

        "predicted_class": predicted_class,

        "probability": probability,

        "confidence": confidence
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("PATHOVISION AI MODEL TEST")
    print("=" * 60)

    print("Model loaded successfully.")

    print(
        "Input shape:",
        model.input_shape
    )

    print(
        "Output shape:",
        model.output_shape
    )

    print("=" * 60)
