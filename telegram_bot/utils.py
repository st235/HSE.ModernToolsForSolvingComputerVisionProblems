import cv2
import numpy as np
import time
import tvm

from skimage import io
from tvm.contrib import graph_executor
from typing import Callable


def perform_benchmarking(original_library: tvm.runtime.Module,
                         tuned_library: tvm.runtime.Module,
                         data: np.ndarray):
    _, original_inference_time = __measure_performance(lambda: __perform_segmentation_inference(original_library, data))
    data, tuned_inference_time = __measure_performance(lambda: __perform_segmentation_inference(tuned_library, data))
    return data, original_inference_time, tuned_inference_time


def load_image(image_path: str) -> np.ndarray:
    image = io.imread(image_path)
    return image.astype(np.float32)


def restore_mask(image: np.ndarray,
                 original_size: tuple[int, int],
                 acceptance_threshold: float = 0.5,
                 convert_to_rgb: bool = True) -> np.ndarray:
    assert 0 <= acceptance_threshold <= 1

    h, w = original_size
    image = (image > acceptance_threshold).astype(np.uint8)
    image = cv2.resize(image, (w, h))
    image = (image * 255.0).astype(np.uint8)
    if convert_to_rgb:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        return image


def normalise_image(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    shape = image.shape

    # Model input tensor shape.
    image = cv2.resize(image, (256, 256))
    image = image.astype(np.float32)

    image[:, :, 0] = (image[:, :, 0] / 255.0)
    image[:, :, 1] = (image[:, :, 1] / 255.0)
    image[:, :, 2] = (image[:, :, 2] / 255.0)

    image = np.expand_dims(image, axis=0)
    return image, (shape[0], shape[1])


def __perform_segmentation_inference(library: tvm.runtime.Module,
                                     data: np.ndarray):
    module = graph_executor.GraphModule(library["default"](tvm.cpu()))
    module.set_input('input_29', tvm.nd.array(data))
    module.run()
    return module.get_output(0).numpy()


def __measure_performance(executable: Callable) -> tuple:
    start_time = time.time_ns()
    result = executable()
    end_time = time.time_ns()
    return result, int((end_time - start_time) / 1000000)
