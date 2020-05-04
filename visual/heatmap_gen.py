import time
import numpy as np
import cv2
import matplotlib.pyplot as plt


def CenterLabelHeatmap(img_width, img_height, c_x, c_y, sigma):
    X1 = np.linspace(start=1, stop=img_width, num=img_width)
    Y1 = np.linspace(start=1, stop=img_height, num=img_height)
    [X, Y] = np.meshgrid(X1, Y1)
    X = X - c_x
    Y = Y - c_y
    D2 = X * X + Y * Y
    E2 = 2.0 * sigma * sigma
    Exponent = D2 / E2
    heatmap = np.exp(-Exponent)
    return heatmap


# Compute gaussian kernel
def CenterGaussianHeatmap(img_height, img_width, c_x, c_y, variance):
    gaussian_map = np.zeros((img_height, img_width))
    for x_p in range(img_width):
        for y_p in range(img_height):
            dist_sq = (x_p - c_x) * (x_p - c_x) + \
                      (y_p - c_y) * (y_p - c_y)
            exponent = dist_sq / 2.0 / variance / variance
            gaussian_map[y_p, x_p] = np.exp(-exponent)
    return gaussian_map


image_file = 'visual/test.png'
img = cv2.imread(image_file)
img = img[:, :, ::-1]
print('> img.shape = ', img.shape)

height, width, _ = np.shape(img)
cy, cx = height / 2.0, width / 2.0

start = time.time()
heatmap1 = CenterLabelHeatmap(width, height, cx, cy, 21)
t1 = time.time() - start

start = time.time()
heatmap2 = CenterGaussianHeatmap(height, width, cx, cy, 21)
t2 = time.time() - start

print('> t1=', t1)
print('> t2=', t2)


plt.subplot(1, 2, 1)
plt.imshow(heatmap1)
plt.subplot(1, 2, 2)
plt.imshow(heatmap2)
plt.show()

print('End.')
