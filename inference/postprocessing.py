import os.path

import numpy as np
from scipy.ndimage import binary_dilation
from skimage import measure


def process_confidence(scene_id, input_dir, mask_dict=None):
    data = []
    if mask_dict:
        fishing_preds = mask_dict["fishing_mask"]
        vessel_preds = mask_dict["vessel_mask"]
        length_preds = mask_dict["length_mask"]
        center_preds = mask_dict["center_mask"]
    else:
        try:
            fishing_preds = np.load(
                os.path.join(input_dir, scene_id, "fishing_preds.npy")
            )
            vessel_preds = np.load(
                os.path.join(input_dir, scene_id, "vessel_preds.npy")
            )
            length_preds = np.load(
                os.path.join(input_dir, scene_id, "length_preds.npy")
            )
            center_preds = np.load(
                os.path.join(input_dir, scene_id, "center_preds.npy")
            )
        except FileNotFoundError:
            return data

    centers = center_preds > 100
    centers = binary_dilation(centers, iterations=1)
    labeled_image = measure.label(centers)

    for rprop in measure.regionprops(labeled_image):
        y1, x1, y2, x2 = rprop.bbox
        y0, x0 = rprop.centroid
        prop_image = rprop.image
        c = center_preds[y1:y2, x1:x2][prop_image]
        vessel = vessel_preds[y1:y2, x1:x2][prop_image]
        fishing = fishing_preds[y1:y2, x1:x2][prop_image]
        length = length_preds[y1:y2, x1:x2][prop_image]
        mean_vessel = np.mean(vessel)
        mean_fishing = np.mean(fishing)
        mean_length = np.mean(length)
        mean_center = np.mean(c)
        is_vessel = np.count_nonzero(vessel > 100) > 8
        is_fishing = np.count_nonzero(fishing > 100) > 8
        data.append(
            [
                int(y0),
                int(x0),
                scene_id,
                is_vessel,
                is_fishing,
                length.max(),
                127,
                127,
                mean_vessel,
                mean_fishing,
                mean_length,
                mean_center,
            ]
        )
    return data
