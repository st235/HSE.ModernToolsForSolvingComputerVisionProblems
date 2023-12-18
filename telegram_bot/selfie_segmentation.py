import numpy as np

from tvm.driver import tvmc
from utils import load_image, restore_mask, load_tvm_parameters, normalise_image


def segment_an_image(fname):
    res = {"info": "", "images": []}

    def store_image_data(img, name):
        res["images"].append((name + '.png', img))

    rgb_img = load_image(fname)
    store_image_data(rgb_img.astype(np.uint8), 'original_image')

    data, original_size = normalise_image(rgb_img)
    params = load_tvm_parameters('parameters.npy')

    def infer_model_tvmc(data, package, params):
        package = tvmc.TVMCPackage(package_path=package)
        inputs = {"input_29": data, **params}
        num_iter = 10
        result = tvmc.run(package, device="cpu", inputs=inputs, number=num_iter)
        cost = np.array(result.times).mean()
        res["info"] = "Inference time: %g ms\n" % (cost * 1000)
        return result.outputs["output_0"]

    out = infer_model_tvmc(data, "package.tar", params)

    store_image_data(restore_mask(out[0, :, :, 0], original_size), 'background')
    store_image_data(restore_mask(out[0, :, :, 1], original_size), 'hair')
    store_image_data(restore_mask(out[0, :, :, 2], original_size), 'body-skin')
    store_image_data(restore_mask(out[0, :, :, 3], original_size), 'face-skin')
    store_image_data(restore_mask(out[0, :, :, 4], original_size), 'clothes')
    store_image_data(restore_mask(out[0, :, :, 5], original_size), 'accessories')

    return res
