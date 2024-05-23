import numpy as np
from PIL import Image

def avg_border_col(np_img):
    if isinstance(np_img, Image.Image):
        np_img=np.array(np_img)
    top = np_img[0]
    bottom = np_img[-1]
    left = np_img[1:-1, 0]
    right = np_img[1:-1, -1]
    border_pixels = np.concatenate((top, bottom, left, right))
    average_col = np.mean(border_pixels, axis=0)
    return average_col


def avg_border_col_agg(dataset):
    border_cols = []
    for img,_ in dataset:
        avg_col = avg_border_col(img)
        border_cols.append(avg_col)
    return tuple(np.mean(border_cols, axis=0))