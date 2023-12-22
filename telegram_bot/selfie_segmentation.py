import cv2
import tvm
import numpy as np

from utils import load_image, restore_mask, normalise_image, perform_benchmarking

original_segmentation_library = tvm.runtime.load_module('./original_library.so')
tuned_segmentation_library = tvm.runtime.load_module('./tuned_library.so')


def segment_an_image(file_name: str):
    res = {"info": "", "images": []}

    def store_image_data(image, name):
        res["images"].append((image, name))

    rgb_image = load_image(file_name)
    store_image_data(rgb_image.astype(np.uint8), 'original image')

    normalised_image, original_size = normalise_image(rgb_image)

    output_masks, original_time_ms, tuned_time_ms = perform_benchmarking(
        original_library=original_segmentation_library,
        tuned_library=tuned_segmentation_library,
        data=normalised_image
    )

    store_image_data(restore_mask(output_masks[0, :, :, 0], original_size), 'background')
    store_image_data(restore_mask(output_masks[0, :, :, 1], original_size), 'hair')
    store_image_data(restore_mask(output_masks[0, :, :, 2], original_size), 'body-skin')
    store_image_data(restore_mask(output_masks[0, :, :, 3], original_size), 'face-skin')
    store_image_data(restore_mask(output_masks[0, :, :, 4], original_size), 'clothes')
    store_image_data(restore_mask(output_masks[0, :, :, 5], original_size), 'accessories')

    res['info'] = f"Inference of default model took {original_time_ms} ms.\nInference of tuned model took {tuned_time_ms} ms."

    return res


def patch_hair(file_name: str,
               color: tuple[int, int, int]):
    res = {"info": "", "image": None}

    rgb_image = load_image(file_name).astype(np.uint8)
    normalised_image, original_size = normalise_image(rgb_image)

    output_masks, original_time_ms, tuned_time_ms = perform_benchmarking(
        original_library=original_segmentation_library,
        tuned_library=tuned_segmentation_library,
        data=normalised_image
    )

    hair_mask = restore_mask(output_masks[0, :, :, 1], original_size, convert_to_rgb=False)
    mask_height, mask_width = hair_mask.shape[0], hair_mask.shape[1]

    dyed_hairs = np.zeros((mask_height, mask_width, 3), np.uint8)
    dyed_hairs[:] = color

    dyed_hairs = cv2.bitwise_and(dyed_hairs, dyed_hairs, mask=hair_mask)
    dyed_hairs = cv2.blur(dyed_hairs, (9, 9))

    dyed_portrait = cv2.addWeighted(rgb_image, 0.9, dyed_hairs, 0.22, 0)

    res['info'] = f"Inference of default model took {original_time_ms} ms.\nInference of tuned model took {tuned_time_ms} ms."
    res['image'] = dyed_portrait

    return res


def patch_background(background_image_path: str,
                     foreground_image_path: str):
    res = {"info": "", "image": None}

    background_image = load_image(background_image_path).astype(np.uint8)
    foreground_image = load_image(foreground_image_path).astype(np.uint8)

    desired_width = int(foreground_image.shape[1])
    desired_height = int(
        background_image.shape[0]
        * (foreground_image.shape[1] / background_image.shape[1])
    )

    normalised_image, original_size = normalise_image(foreground_image)

    output_masks, original_time_ms, tuned_time_ms = perform_benchmarking(
        original_library=original_segmentation_library,
        tuned_library=tuned_segmentation_library,
        data=normalised_image
    )

    background_mask = restore_mask(output_masks[0, :, :, 0], original_size, convert_to_rgb=False)

    # If width scaling resulted in a smaller image
    # Let's additionally scale it to make height equal
    # to the original image's height
    if desired_height < foreground_image.shape[0]:
        desired_width = int(
            desired_width * (foreground_image.shape[0] / desired_height)
        )
        desired_height = int(foreground_image.shape[0])

    desired_dimensions = (desired_width, desired_height)

    resized_background = cv2.resize(
        background_image, desired_dimensions, interpolation=cv2.INTER_AREA
    )

    leftover_width = resized_background.shape[1] - foreground_image.shape[1]
    leftover_height = resized_background.shape[0] - foreground_image.shape[0]

    cropped_background = resized_background[
                         leftover_height: leftover_height + foreground_image.shape[0],
                         leftover_width: leftover_width + foreground_image.shape[1],
                         ]

    foreground = cv2.bitwise_and(foreground_image, foreground_image, mask=cv2.bitwise_not(background_mask))
    background = cv2.bitwise_and(cropped_background, cropped_background, mask=background_mask)

    res['info'] = f"Inference of default model took {original_time_ms} ms.\nInference of tuned model took {tuned_time_ms} ms."
    res['image'] = cv2.bitwise_or(foreground, background)

    return res
