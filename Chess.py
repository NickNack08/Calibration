import cv2
import numpy as np
from matplotlib import pyplot as plt

# Define camera matrix K
K = np.array([[1.79677330e+03, 0.00000000e+00, 6.03044271e+02],
 [0.00000000e+00, 1.79821147e+03, 4.92880943e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# Define distortion coefficients d
d = np.array([5.44787247e-02, 1.23043244e-01, -4.52559581e-04, 5.47011732e-03, -6.83110234e-01])

# Read an example image and acquire its size
img = cv2.imread('/Users/nickduggan/Desktop/IMAGING/Foosball.jpg')
h, w = img.shape[:2]

# Generate new camera matrix from parameters
newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(K, d, (w,h), 0)

# Generate look-up tables for remapping the camera image
mapx, mapy = cv2.initUndistortRectifyMap(K, d, None, newcameramatrix, (w, h), 5)

# Remap the original image to a new image
newimg = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

# Display old and new image
fig, (oldimg_ax, newimg_ax) = plt.subplots(1, 2)
oldimg_ax.imshow(img)
oldimg_ax.set_title('Original image')
newimg_ax.imshow(newimg)
newimg_ax.set_title('Unwarped image')
plt.show()

# moo
