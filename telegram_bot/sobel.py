from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np

import tvm
from tvm import relay
from tvm.driver import tvmc
from tvm.contrib import graph_executor


def apply_tvm_sobel(fname):
    res = {"info": "", "images": [] }
    
    def read_image(url):
        return io.imread(url)
    
    def store_image_data(img, name):
        res["images"].append((name + '.png', img))

    def rgb2gray(data):
        return color.rgb2gray(data)
    
    rgb_img = read_image(fname)
    store_image_data(rgb_img, 'original_image')

    data = rgb2gray(rgb_img)
    store_image_data(data, 'gray_image')

    def get_filter(dtype="float32"):
        sobelv = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=dtype)
        sobelh = sobelv.transpose()
        sobel = np.array((sobelh, sobelv))
        sobel = np.expand_dims(sobel, axis=3)
        sobel = sobel.transpose(1, 2, 3, 0)
        return sobel

    def expand_image_dims(data):
        assert(len(data.shape) == 2)
        return np.expand_dims(data, axis=(0, 3))

    data = expand_image_dims(data)
    params = { "weight": tvm.nd.array(get_filter()) }     


    def infer_model_tvmc(data, package, params):
        inputs = {"input": data, **params}
        num_iter = 10
        result = tvmc.run(package, device="cpu", inputs=inputs, number=num_iter)
        cost = result.times.mean
        res["info"] = "Inference time: %g ms\n" % (cost * 1000)
        out = result.outputs["output_0"]
        return out
    
    package = tvmc.TVMCPackage(package_path="package.tar")
    out = infer_model_tvmc(data, package, params)

    store_image_data(out[0, :, :, 0], 'horizontal_gradient_image')
    store_image_data(out[0, :, :, 1], 'vertical_gradient_image')


    def grad_amplitude(dx, dy):
        return np.sqrt(dx ** 2 + dy ** 2)
    
    union = grad_amplitude(out[0, :, :, 0], out[0, :, :, 1])
    store_image_data(union, 'gradient_amplitude_image')

    return res
