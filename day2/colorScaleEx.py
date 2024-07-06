import cv2
import numpy as np

r_bgr = np.array([[[0,0,255]]], dtype=np.uint8)
g_bgr = np.array([[[0,255,0]]], dtype=np.uint8)
b_bgr = np.array([[[255,0,0]]], dtype=np.uint8)
y_bgr = np.array([[[0,255,255]]], dtype=np.uint8)

r_hsv = cv2.cvtColor(r_bgr, cv2.COLOR_BGR2HSV)
g_hsv = cv2.cvtColor(g_bgr, cv2.COLOR_BGR2HSV)
b_hsv = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2HSV)
y_hsv = cv2.cvtColor(y_bgr, cv2.COLOR_BGR2HSV)

print('hsv red:',r_hsv)
print('hsv green:',g_hsv)
print('hsv blue:',b_hsv)
print('hsv yellow:',y_hsv)