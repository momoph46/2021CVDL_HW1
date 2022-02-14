import cv2
import numpy as np

imL = cv2.imread('Q3_Image/imL.png')
imR = cv2.imread('Q3_Image/imR.png')


def stereo():
    gray_imgL = cv2.cvtColor(imL, cv2.COLOR_BGR2GRAY)
    gray_imgR = cv2.cvtColor(imR, cv2.COLOR_BGR2GRAY)
    stereo = cv2.StereoBM_create(numDisparities=256, blockSize=21)
    disparity = stereo.compute(gray_imgL, gray_imgR)
    cv2.imshow('gray',disparity)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def mouse_handler(event, x, y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:

        img_left=imL
        imr = imR.copy()
        cv2.circle(imr, (x-int(data[y][x]/16),y), 12, (0,0,255), 20, 16)

        img_final = np.concatenate((img_left, imr),1)
        cv2.imshow("Image",img_final)


def get_points():

    imL_g = cv2.cvtColor(imL, cv2.COLOR_BGR2GRAY)
    imR_g = cv2.cvtColor(imR, cv2.COLOR_BGR2GRAY)
    stereo = cv2.StereoBM_create(256, 21)
    data = stereo.compute(imL_g, imR_g)

    cv2.namedWindow("Image", 0)

    img_final = np.concatenate((imL, imR),1)
    cv2.imshow("Image", img_final)
    cv2.setMouseCallback("Image", mouse_handler, data)

    cv2.waitKey()
    cv2.destroyAllWindows()

#stereo()
#get_points()
