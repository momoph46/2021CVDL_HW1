import cv2
import numpy as np
import matplotlib.pyplot as plt


def createKeyPoint():
    img1 = cv2.imread("Q4_Image/Shark1.jpg")
    img2 = cv2.imread("Q4_Image/Shark2.jpg")
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1,descriptors_2)

    matches = sorted(matches, key = lambda x:x.distance,reverse=1)
    img1_idx=[]
    img2_idx=[]

    for i in range(0,200):
        img1_idx.append(keypoints_1[matches[i].queryIdx])
        img2_idx.append(keypoints_2[matches[i].trainIdx])

    img1 = cv2.drawKeypoints(gray_img1, img1_idx[:200], img1)
    img2 = cv2.drawKeypoints(gray_img2, img2_idx[:200], img2)

    cv2.imshow("img1", img1)
    cv2.imshow("img2", img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def matchedKeyPoint():
    img2 = cv2.imread("Q4_Image/Shark1.jpg")
    img1 = cv2.imread("Q4_Image/Shark2.jpg")

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()

    matches = bf.match(des1,des2)
    #print(type(matches))

    
    matches = sorted(matches, key = lambda x:x.distance,reverse=1)

    good = [[0, 0] for i in range(len(matches))]
    
    draw_params = dict(
        flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    

    #img3 = cv.drawMatches(img1, img1_idx[:200], img2, img2_idx[:200], matches)
    
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:200],None,**draw_params)
    #cv2.namedWindow('img',cv2.WINDOW_AUTOSIZE)
    cv2.imshow('img',img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#createKeyPoint()
#matchedKeyPoint()
