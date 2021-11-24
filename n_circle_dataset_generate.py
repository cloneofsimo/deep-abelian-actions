import cv2
import numpy as np


def black_box_circle(
    img_size: int, radius: int, pos_x: int, pos_y: int, img=None, color=(0, 0, 0)
):

    if img is None:
        img = np.zeros((img_size, img_size, 3), np.uint8)

    cv2.circle(img, (pos_x, pos_y), radius, color, -1)

    return img


def create_1_circle_dataset(size: int, save_path: str, img_size: int, radius: int):
    # set seed
    np.random.seed(0)

    for idx in range(size):
        pos_x = np.random.randint(0, img_size)
        pos_y = np.random.randint(0, img_size)
        img = black_box_circle(img_size, radius, pos_x, pos_y)
        cv2.imwrite(save_path + str(idx) + ".jpg", img)


def create_n_circle_dataset(
    size: int, save_path: str, img_size: int, radius: int, n: int
):
    # set seed
    np.random.seed(0)

    # n rainbow theme colors
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
    ]

    for idx in range(size):
        for cdx in range(n):
            pos_x = np.random.randint(0, img_size)
            pos_y = np.random.randint(0, img_size)
            if cdx == 0:
                img = black_box_circle(
                    img_size, radius, pos_x, pos_y, color=colors[cdx]
                )
            else:
                img = black_box_circle(
                    img_size, radius, pos_x, pos_y, color=colors[cdx], img=img
                )
        cv2.imwrite(save_path + str(idx) + ".jpg", img)


if __name__ == "__main__":

    img_size = 128
    radius = 10
    for idx in range(1, 4):
        save_path = f"./data/{idx}_circle_dataset/"
        create_n_circle_dataset(10000, save_path, img_size, radius, idx)
