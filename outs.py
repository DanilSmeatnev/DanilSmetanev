import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('image.jpg',0)
blur = cv2.GaussianBlur(img,(5,5),0)

hist = cv2.calcHist([blur],[0],None,[256],[0,256])
hist_norm = hist.ravel()/hist.max()
Q = hist_norm.cumsum()

bins = np.arange(256)

fn_min = np.inf
thresh = -1

for i in xrange(1,256):
    p1,p2 = np.hsplit(hist_norm,[i])
    q1,q2 = Q[i],Q[255]-Q[i]
    b1,b2 = np.hsplit(bins,[i])

    m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
    v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2

    fn = v1*q1 + v2*q2
    if fn < fn_min:
        fn_min = fn
        thresh = i

ret, otsu = cv2.threshold(blur,thresh,255,cv2.THRESH_BINARY)

cv2.imshow("image.jpg", otsu)
cv2.waitKey(0)








