import cv2
import tvm
import numpy as np

from utils import load_image, restore_mask, normalise_image, perform_segmentation_inference, measure_performance

selfie_segmentation_library = tvm.runtime.load_module('./library.so')


def segment_an_image(file_name: str):
    res = {"info": "", "images": []}

    def store_image_data(image, name):
        res["images"].append((image, name))

    rgb_image = load_image(file_name)
    store_image_data(rgb_image.astype(np.uint8), 'original image')

    normalised_image, original_size = normalise_image(rgb_image)

    output_masks, time_ms = measure_performance(
        lambda: perform_segmentation_inference(selfie_segmentation_library, normalised_image))

    store_image_data(restore_mask(output_masks[0, :, :, 0], original_size), 'background')
    store_image_data(restore_mask(output_masks[0, :, :, 1], original_size), 'hair')
    store_image_data(restore_mask(output_masks[0, :, :, 2], original_size), 'body-skin')
    store_image_data(restore_mask(output_masks[0, :, :, 3], original_size), 'face-skin')
    store_image_data(restore_mask(output_masks[0, :, :, 4], original_size), 'clothes')
    store_image_data(restore_mask(output_masks[0, :, :, 5], original_size), 'accessories')

    res['info'] = f"Inference took {time_ms} ms."

    return res


def patch_hair(file_name: str,
               color: tuple[int, int, int]):
    res = {"info": "", "image": None}

    rgb_image = load_image(file_name).astype(np.uint8)
    normalised_image, original_size = normalise_image(rgb_image)

    output_masks, time_ms = measure_performance(
        lambda: perform_segmentation_inference(selfie_segmentation_library, normalised_image))

    hair_mask = restore_mask(output_masks[0, :, :, 1], original_size, convert_to_rgb=False)
    mask_height, mask_width = hair_mask.shape[0], hair_mask.shape[1]

    dyed_hairs = np.zeros((mask_height, mask_width, 3), np.uint8)
    dyed_hairs[:] = color

    dyed_hairs = cv2.bitwise_and(dyed_hairs, dyed_hairs, mask=hair_mask)
    dyed_hairs = cv2.blur(dyed_hairs, (9, 9))

    dyed_portrait = cv2.addWeighted(rgb_image, 0.9, dyed_hairs, 0.22, 0)

    res['info'] = f"Inference took {time_ms} ms."
    res['image'] = dyed_portrait

    return res
