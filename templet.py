import cv2 as cv
import numpy as np

img_rgb = cv.imread('images/test.png')
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

template1 = cv.imread('images/cactus1.png', 0)
template2 = cv.imread('images/cactus2.png', 0)
template3 = cv.imread('images/bird1.png', 0)
template4 = cv.imread('images/bird2.png', 0)

w1, h1 = template1.shape[::-1]
w2, h2 = template1.shape[::-1]
w3, h3 = template1.shape[::-1]
w4, h4 = template1.shape[::-1]

res1 = cv.matchTemplate(img_gray, template1, cv.TM_CCOEFF_NORMED)
res2 = cv.matchTemplate(img_gray, template2, cv.TM_CCOEFF_NORMED)
res3 = cv.matchTemplate(img_gray, template3, cv.TM_CCOEFF_NORMED)
res4 = cv.matchTemplate(img_gray, template4, cv.TM_CCOEFF_NORMED)

threshold = 0.8
loc1 = np.where(res1 >= threshold)
loc2 = np.where(res2 >= threshold)
loc3 = np.where(res1 >= threshold)
loc4 = np.where(res2 >= threshold)

for pt in zip(*loc1[::-1]):
    cv.rectangle(img_rgb, pt, (pt[0] + w1, pt[1] + h1), (0, 0, 255), 2)

for pt in zip(*loc2[::-1]):
    cv.rectangle(img_rgb, pt, (pt[0] + w2, pt[1] + h2), (0, 255, 0), 2)

for pt in zip(*loc1[::-1]):
    cv.rectangle(img_rgb, pt, (pt[0] + w3, pt[1] + h3), (0, 0, 255), 2)

for pt in zip(*loc2[::-1]):
    cv.rectangle(img_rgb, pt, (pt[0] + w4, pt[1] + h4), (0, 255, 0), 2)

cv.imshow('result', img_rgb)
cv.waitKey(0)