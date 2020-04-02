# import numpy as np
# import cv2 as cv
# path1='Data/book1.png'

# img = cv.imread(path1)
# gray= cv.cvtColor(img, cv.COLOR_BGR2GRAY)
 
# akaze = cv.AKAZE_create()
# kp, descriptor = akaze.detectAndCompute(gray, None)
 
# img=cv.drawKeypoints(gray, kp, img)
# cv.imwrite('keypoints.jpg', img)



import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
 
img1 = cv.imread('Data/book1.png', cv.IMREAD_GRAYSCALE)  # referenceImage
img2 = cv.imread('Data/book2.png', cv.IMREAD_GRAYSCALE)  # sensedImage
 
# Initiate AKAZE detector
akaze = cv.AKAZE_create()
# Find the keypoints and descriptors with SIFT
kp1, des1 = akaze.detectAndCompute(img1, None)
kp2, des2 = akaze.detectAndCompute(img2, None)
 
# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
 
# Apply ratio test
good_matches = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good_matches.append([m])
         
# Draw matches
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good_matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# cv.imwrite('matches.jpg', img3)
# plt.imshow(img3)


ref_matched_kpts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1,1,2)
sensed_matched_kpts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1,1,2)
 
# Compute homography
H, status = cv.findHomography(ref_matched_kpts, sensed_matched_kpts, cv.RANSAC,5.0)
 
# Warp image
warped_image = cv.warpPerspective(img2, H, (img1.shape[1]+img2.shape[1], img1.shape[0]))
             
cv.imwrite('warped.jpg', warped_image)