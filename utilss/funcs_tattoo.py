import os.path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

def bitwise_and(src1, src2, mask, alpha=None):
    h, w = src1.shape[:2]
    for i in range(h):
        for j in range(w):
            if mask[i,j] == 255:
                if alpha is not None:
                    src1[i,j] = np.uint8(src1[i,j] * (1 - alpha) + src2[i,j] * alpha)
                else:
                    src1[i,j] = src2[i,j]
    return src1

def get_mask(image):
    image_g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_t = np.invert(image_g)
    _, thresh = cv2.threshold(image_t, 0, 255, cv2.THRESH_BINARY)
    return thresh

def get_coord(mask):
    ymin = min(np.where(mask == 255)[0])
    ymax = max(np.where(mask == 255)[0])
    xmin = min(np.where(mask == 255)[1])
    xmax = max(np.where(mask == 255)[1])
    return xmin, ymin, xmax, ymax, ymax - ymin, xmax - xmin

def resize(image, w=None, h=None):
    h_image, w_image, _ = image.shape
    if w is not None:
        ratio = w_image / w
        new_h = int(h_image / ratio)
        return cv2.resize(image, (w, new_h), interpolation=cv2.INTER_CUBIC)
    if h is not None:
        ratio = h_image / h
        new_w = int(w_image / ratio)
        return cv2.resize(image, (new_w, h), interpolation=cv2.INTER_CUBIC)

def padding(image, new_size, x, y):
    bg = np.zeros((*new_size, 3))
    if len(image.shape) == 3:
        h, w, _ = image.shape
    else:
        h, w = image.shape
    bg[y:y+h, x:x+w] = image
    return np.uint8(bg)

def get_name(path_name):
    if '/' in path_name:
        return path_name.split('/')[-1].split('.')[0] + '.png'
    else:
        return path_name.split('.')[0] + '.png'

def save(image, path_name, root='src/results/'):
    path_name = os.path.join(root, get_name(path_name))
    cv2.imwrite(path_name, image )

def plot_image(img):
    height, width = img.shape[:2]
    plt.figure(figsize=(12, height / width * 12))

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')

    plt.imshow(img[..., ::-1])
    plt.show()

def random_xy(h, w):
    x = random.randint(0, int(w * 0.1))
    y = random.randint(int(h * 0.1), int(h * 0.7))
    return x, y

def check_type(img_path):
    if type(img_path) == str:
        if img_path.endswith(('.jpg', '.png', '.jpeg')):
            img = cv2.imread(img_path)
        else:
            raise Exception("Please input a image file")
    elif type(img_path) == np.ndarray:
        img = img_path
    return img

def read_image(image_path, mask_path, tattoo_path):
    image = check_type(image_path)
    mask = check_type(mask_path)
    tattoo = check_type(tattoo_path)
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return image, mask, tattoo

def transform_bbox(xmin, ymin, xmax, ymax):
    old_size = (xmax - xmin + ymax - ymin) / 2
    center_x = xmax - (xmax - xmin) / 2.0 - old_size * 0.05
    center_y = ymax - (ymax - ymin) / 2.0 + old_size * 0.05 # 0.03
    size = int(old_size * 1.3) # 1.25
    roi_box = [0] * 6
    roi_box[0] = center_x - size / 2
    roi_box[1] = center_y - size / 2
    roi_box[2] = roi_box[0] + size
    roi_box[3] = roi_box[1] + size
    roi_box[4] = roi_box[3] -roi_box[1]
    roi_box[5] = roi_box[2] - roi_box[0]
    return roi_box # x1, y1, x2, y2, h_roi, w_roi