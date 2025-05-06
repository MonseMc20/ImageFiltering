import easyocr
import numpy as np

import cv2 as cv
from PIL import ImageFilter
from PIL import Image

#read the image
img = cv.imread('C:\\Users\\monse\\Downloads\\placa_4.jpg')

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

# Load the image
image = cv.imread('mask.png')

# Apply Gaussian blur
blurred_image = cv.GaussianBlur(image, (25, 25), 0)

# Save the blurred image 
cv.imwrite('blurred_placa.jpg', blurred_image)

img = cv.imread('blurred_placa.jpg')  

sobel_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
sobel_y = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
sobel_xy = cv.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

# Apply Gaussian blur
sobel_xy = cv.GaussianBlur(image, (25, 25), 0)

# Save the blurred image 
cv.imwrite('borders_placa.jpg', blurred_image)

#spanish language
reader = easyocr.Reader(['es'], gpu=False)
image = cv.imread("borders_placa.jpg")

result = reader.readtext(image, paragraph=False)

for res in result:
     print('res:', res)
     pt_0 = res[0][0]
     pt_1 = res[0][1]
     pt_2 = res[0][2]
     pt_3 = res[0][3]

     cv.rectangle(image, pt_0, (pt_1[0], pt_1[1] - 23), (166, 56, 242), -1)
     cv.putText(image, res[1], (pt_0[0], pt_0[1] -3), 2, 0.8, (255, 255, 255), 1)

     cv.rectangle(image, pt_0, pt_2, (166, 56, 242), 2)
     cv.circle(image, pt_0, 2, (255, 0, 0), 2)
     cv.circle(image, pt_1, 2, (0, 255, 0), 2)
     cv.circle(image, pt_2, 2, (0, 0, 255), 2)
     cv.circle(image, pt_3, 2, (0, 255, 255), 2)

     cv.imwrite('placa_1.png', image)
