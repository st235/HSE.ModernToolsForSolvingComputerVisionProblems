import tvm
import numpy as np

from tvm.contrib import graph_executor
from utils import load_image, restore_mask, normalise_image


def segment_an_image(fname):
    res = {"info": "", "images": []}

    def store_image_data(img, name):
        res["images"].append((name + '.png', img))

    rgb_img = load_image(fname)
    store_image_data(rgb_img.astype(np.uint8), 'original_image')

    data, original_size = normalise_image(rgb_img)

    def infer_model_tvmc(data):
        loaded_lib = tvm.runtime.load_module('./library.so')
        module = graph_executor.GraphModule(loaded_lib["default"](tvm.cpu()))
        module.set_input('input_29', tvm.nd.array(data))
        module.run()
        res["info"] = "Inference time: NaN ms\n"
        return module.get_output(0).numpy()

    out = infer_model_tvmc(data)

    store_image_data(restore_mask(out[0, :, :, 0], original_size), 'background')
    store_image_data(restore_mask(out[0, :, :, 1], original_size), 'hair')
    store_image_data(restore_mask(out[0, :, :, 2], original_size), 'body-skin')
    store_image_data(restore_mask(out[0, :, :, 3], original_size), 'face-skin')
    store_image_data(restore_mask(out[0, :, :, 4], original_size), 'clothes')
    store_image_data(restore_mask(out[0, :, :, 5], original_size), 'accessories')

    return res
