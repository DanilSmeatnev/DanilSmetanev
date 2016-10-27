from __future__ import print_function
from sys import argv
import cv2
import numpy as np


def gradient_img(img):
    hor_grad = (img[1:, :] - img[:-1, :])[:, :-1]
    ver_grad = (img[:, 1:] - img[:, :-1])[:-1:, :]
    magnitude = np.sqrt(hor_grad ** 2 + ver_grad ** 2)

    return magnitude


def hough_transform(img, theta, rho):
    pass


def get_lines(ht_map, n_lines, thetas, rhos, min_delta_rho, min_delta_theta):
    pass


if __name__ == '__main__':
    assert 9 == len(argv)
    src_path, dst_ht_path, dst_lines_path, theta, rho,\
        n_lines, min_delta_rho, min_delta_theta = argv[1:]

    theta = float(theta)
    rho = float(rho)
    n_lines = int(n_lines)
    min_delta_rho = float(min_delta_rho)
    min_delta_theta = float(min_delta_theta)

    assert theta > 0.0
    assert rho > 0.0
    assert n_lines > 0
    assert min_delta_rho > 0.0
    assert min_delta_theta > 0.0

    image = cv2.imread(src_path, 0)
    assert image is not None

    image = image.astype(float)
    gradient = gradient_img(image)

    ht_map, thetas, rhos = hough_transform(gradient, theta, rho)
    cv2.imwrite(dst_ht_path, ht_map)

    lines = get_lines(ht_map, thetas, rhos, n_lines, min_delta_rho, min_delta_theta)
    with open(dst_lines_path, 'w') as fout:

        def hough_transform(gradient, theta, rho):
            h, w = gradient.shape
            irho = 1.0 / rho

            numangle = int(round(np.pi / 2 / theta))
            numrho = int(round(((w + h) * 2 + 1) / rho))

            accum = np.zeros((numangle + 2) * (numrho + 2), np.int32)

            sin_t = np.zeros(numangle, dtype=float)
            cos_t = np.zeros(numangle, dtype=float)

            ang = 0
            for i in xrange(numangle):
                sin_t[i] = np.sin(ang * irho)
                cos_t[i] = np.cos(ang * irho)
                ang += theta

            for i in xrange(h):
                for j in xrange(w):
                    for n in xrange(numangle):
                        if (gradient[i][j] != 0):
                            r = round(j * cos_t[n] + i * sin_t[n])
                            r += (numrho - 1) / 2
                            r = int(r)
                            accum[(n + 1) * (numrho + 2) + r + 1] += gradient[i][j]

            ht_map = np.reshape(accum, (numangle + 2, numrho + 2))
            ht_map = ht_map.astype(np.float32)
            ht_map /= ht_map.max()
            ht_map *= 255
            ht_map[ht_map > 255.0] = 255
            ht_map = ht_map.astype(np.uint8)
            return ht_map
