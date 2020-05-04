import numpy as np
import torchvision
import cv2
import PIL
import matplotlib.pyplot as plt

np_image = np.zeros((5, 5))
np_image[2, 2] = 1.0
image = PIL.Image.fromarray(np_image)
# > before resize
plt.imshow(np_image)
plt.show()
print('> [1] = ', np_image)

# > after resize x 2
#（等價於PIL中的resize)
pil_image = torchvision.transforms.functional.resize(image, (15, 15))
print('> [2] PIL = ', np.array(pil_image))

plt.imshow(np.array(pil_image))
plt.show()

cv_image = cv2.resize(np_image, (15, 15))
print('> [2] CV = ', np.array(cv_image))
plt.imshow(cv_image)
plt.show()

