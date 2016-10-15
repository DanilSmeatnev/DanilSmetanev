from __future__ import print_function
from sys import argv
import os.path
import cv2
import numpy as np


def box_flter(src_path, dst_path, w, h):
    src = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    assert (src.ndim == 2)

    W, H = src.shape
    assert(W > w and H > h)
    integrated_image = cv2.integral(src)


    dst = np.ndarray(shape=(W-w+1, H-h+1), dtype=float)
    for i in range(0, W - w + 1):
        for j in range(0, H - h + 1):
            dst[i][j] = integrated_image[i + w][j + h] - \
                        integrated_image[i + w][j] - \
                        integrated_image[i][j + h] + \
                        integrated_image[i][j]


    dst /= w * h
    dst[dst > 255.0] = 255
    dst = dst.astype(np.uint8)
    print (dst)
    cv2.imwrite(dst_path, dst)

if __name__ == '__main__':
    assert len(argv) == 5
    assert os.path.exists(argv[1])
    argv[3] = int(argv[3])
    argv[4] = int(argv[4])
    assert argv[3] > 0
    assert argv[4] > 0

    box_flter(*argv[1:])
