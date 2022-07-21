import os.path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from utilss.funcs_tattoo import *

class AddTattoo():
    def __init__(self, seed=None, is_save=False, is_plot=False, opacity=0.9):
        self.seed = seed
        self.is_save = is_save
        self.is_plot = is_plot
        self.opacity = opacity

    def run(self, image_path, mask_path, tattoo_path):
        image, mask, tattoo = read_image(image_path, mask_path, tattoo_path)
        xmin, ymin, xmax, ymax, h_roi, w_roi = get_coord(mask)
        tattoo = resize(tattoo, w=w_roi)
        h_tattoo, w_tattoo = tattoo.shape[:2]
        roi = image[ymin:ymax, xmin: xmax]
        mask_roi = mask[ymin:ymax, xmin:xmax]

        if self.seed is not None:
            random.seed(25)

        x, y = random_xy(h_roi, w_roi)

        # x, y = 20, 20

        sub_roi = roi[y:y + h_tattoo, x:x + w_tattoo]
        sub_mask = mask_roi[y:y + h_tattoo, x:x + w_tattoo]

        tattoo_mask = get_mask(tattoo)
        final_mask = bitwise_and(sub_mask, tattoo_mask, mask=sub_mask)
        tattoo_in_skin = bitwise_and(sub_roi, tattoo, mask=final_mask, alpha=self.opacity)
        roi[y:y + h_tattoo, x:x + w_tattoo] = tattoo_in_skin

        image[ymin:ymax, xmin:xmax] = roi

        if self.is_save:
            save(image, tattoo_path)
        if self.is_plot:
            plot_image(image)

        return image
