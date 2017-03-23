# Testing k-means algorithm for image color clustering

import numpy as np
import cv2
from matplotlib import pyplot as plt
import Tkinter, tkFileDialog
import sys

# # Simple 1-D k-means clustering

# x = np.random.randint(25, 100, 25)
# y = np.random.randint(175, 250, 25)
# z = np.concatenate((x, y))
# z = z.reshape(50, 1)
# z = np.float32(z)

# # plt.hist(z, 256, [0,255]), plt.show()

# # Define the criteria, and flags for clustering
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# flags = cv2.KMEANS_RANDOM_CENTERS

# # Apply the k-means clustering
# compactness, labels, centers = cv2.kmeans(z, 2, None, criteria, 10, flags)

# A = z[labels == 0]
# B = z[labels == 1]

# # Plot
# plt.hist(A, 256, [0,256], color = 'r')
# plt.hist(B, 256, [0,256], color = 'b')
# plt.hist(centers, 32, [0,256], color = 'y')
# plt.show()

# Simple 2-D k-means clustering
# x = np.random.randint(25, 50, (25,2))
# y = np.random.randint(60, 85, (25,2))
# z = np.vstack((x, y))
# z = np.float32(z)

# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# flags = cv2.KMEANS_RANDOM_CENTERS

# # Apply the k-means clustering
# compactness, labels, centers = cv2.kmeans(z, 2, None, criteria, 10, flags)

# A = z[labels.ravel()==0]
# B = z[labels.ravel()==1]

# plt.scatter(A[:,0],A[:,1])
# plt.scatter(B[:,0],B[:,1],c = 'r')
# plt.scatter(centers[:,0],centers[:,1],s = 80,c = 'y', marker = 's')
# plt.xlabel('Height'),plt.ylabel('Weight')
# plt.show()

# k-means clustering for color images
root = Tkinter.Tk()
filename = tkFileDialog.askopenfilename(parent=root, initialdir="/home/nitin/OpenCV_Projects/standard_test_images/",
                                    title='Please select a file')
# img = cv2.imread('/home/nitin/OpenCV_Projects/standard_test_images/lena_color_512.tif')
img = cv2.imread(filename)
y,x,z = img.shape

if (z!=3): sys.exit("Please select a BGR image")

R = img[:,:,2].reshape(x*y,1)
G = img[:,:,1].reshape(x*y,1)
B = img[:,:,0].reshape(x*y,1)

mat = np.concatenate((B,G,R),axis=1)
mat = np.float32(mat)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS

# Apply the k-means clustering
compactness, labels, centers = cv2.kmeans(mat, 4, None, criteria, 10, flags)

centers = np.uint8(centers)

mat[labels.ravel()==0]=centers[0,:]
mat[labels.ravel()==1]=centers[1,:]
mat[labels.ravel()==2]=centers[2,:]

clImg = mat.reshape((img.shape))
clImg = np.uint8(clImg)

# print(img[256, 256, :]) 
# print(clImg[256, 256, :])

cv2.imshow('Original Image',img)
cv2.imshow('Clustered Image', clImg)
cv2.waitKey(0)
cv2.destroyAllWindows()