from PIL import Image
import numpy as np


IMAGE_SIZE = (224, 224)


def preprocess_image(image_path):

    """
    Prepare uploaded pathology image
    for ResNet50.
    """

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        IMAGE_SIZE
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # ResNet-style normalization
    # --------------------------------------------------------

    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array
