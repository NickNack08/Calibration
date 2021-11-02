import cv2
import numpy as np
from matplotlib import pyplot as plt

# Define camera matrix K
K = np.array([[1.79533619e+03, 0.00000000e+00, 6.15361987e+02],
 [0.00000000e+00, 1.79762029e+03, 4.88841563e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# Define distortion coefficients d
d = np.array([-3.42497623e-01,  1.84956578e+00, -1.78145813e-03, -2.41032139e-03,
  -7.22531291e+00])

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
