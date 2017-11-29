import cv2
import numpy as np
import scipy.ndimage.filters
# from utils import *


img=cv2.imread("content1.png")
cv2.imshow('frame',img)

print(np.shape(img))
img=scipy.ndimage.filters.gaussian_filter(img, 1)
img2=scipy.misc.imresize(img, 25,interp='bilinear')
# img2=np.swapaxes(img2,0,2)
print(np.shape(img2))
cv2.imshow('frame2',img2)

cv2.waitKey(0);