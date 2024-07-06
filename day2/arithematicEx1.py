import cv2
import numpy as np

a = np.uint8([[200, 50]])
b = np.uint8([[100, 100]])

add1 = a + b
print(add1)

add2 = cv2.add(a, b)
print(add2)
print()

sub1 = a - b
print(sub1)

sub2 = cv2.subtract(a, b)
print(sub2)
print()

mult1 = a * 2
print(mult1)

mult2 = cv2.multiply(a, 2)
print(mult2)