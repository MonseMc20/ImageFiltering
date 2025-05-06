import easyocr
import numpy as np

import cv2 as cv
from PIL import ImageFilter
from PIL import Image

#read the image
img = cv.imread('C:\\Users\\monse\\Downloads\\placa_4.jpg')

"""
----------------- Filter black color ----------------------------------
"""

#convert the BGR image to HSV colour space
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

#set the lower and upper bounds for the green hue
lower_black= np.array([0,0,0])
upper_black = np.array([0,0,0])

#create a mask for yellow colour using inRange function
mask = cv.inRange(hsv, lower_black, upper_black)

#perform bitwise and on the original image arrays using the mask
res = cv.bitwise_and(img, img, mask=mask)

#Save the image
cv.imwrite('mask.png', mask)

"""
----------------- Gaussian filter ----------------------------------
"""
# Load the image
image = cv.imread('mask.png')

# Apply Gaussian blur
blurred_image = cv.GaussianBlur(image, (25, 25), 0)

# Save the blurred image 
cv.imwrite('blurred_placa.jpg', blurred_image)

"""
----------------- Borders ----------------------------------
"""

img = cv.imread('blurred_placa.jpg')  

sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
sobelxy = cv.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

"""
----------------- Applying new blur ----------------------------------
"""

# Apply Gaussian blur
sobelxy = cv.GaussianBlur(image, (25, 25), 0)

# Save the blurred image 
cv.imwrite('borders_placa.jpg', blurred_image)

"""
----------------- Text detection ----------------------------------
"""

#spanish language
reader = easyocr.Reader(['es'], gpu=False)
image = cv.imread("borders_placa.jpg")

result = reader.readtext(image, paragraph=False)

for res in result:
     print('res:', res)
     pt0 = res[0][0]
     pt1 = res[0][1]
     pt2 = res[0][2]
     pt3 = res[0][3]

     cv.rectangle(image, pt0, (pt1[0], pt1[1] - 23), (166, 56, 242), -1)
     cv.putText(image, res[1], (pt0[0], pt0[1] -3), 2, 0.8, (255, 255, 255), 1)

     cv.rectangle(image, pt0, pt2, (166, 56, 242), 2)
     cv.circle(image, pt0, 2, (255, 0, 0), 2)
     cv.circle(image, pt1, 2, (0, 255, 0), 2)
     cv.circle(image, pt2, 2, (0, 0, 255), 2)
     cv.circle(image, pt3, 2, (0, 255, 255), 2)

     cv.imwrite('Placa1.png', image)
