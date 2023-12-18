import cv2
import numpy as np

from skimage import io


def load_image(image_path: str) -> np.ndarray:
    image = io.imread(image_path)
    return image.astype(np.float32)


def restore_mask(image: np.ndarray,
                 original_size: tuple[int, int]) -> np.ndarray:
    h, w = original_size
    image = (image > 0.5).astype(np.uint8)
    image = cv2.resize(image, (w, h))
    image = (image * 255.0).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def normalise_image(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    shape = image.shape

    image = cv2.resize(image, (256, 256))
    image = image.astype(np.float32)

    image[:, :, 0] = (image[:, :, 0] / 255.0)
    image[:, :, 1] = (image[:, :, 1] / 255.0)
    image[:, :, 2] = (image[:, :, 2] / 255.0)

    image = np.expand_dims(image, axis=0)
    return image, (shape[0], shape[1])
