import cv2
import numpy as np


def shift_line(line,shift):
    for i in range(3):
        line[0][i]+=shift[i]
        line[1][i]+=shift[i]
    return line

def augmented_reality(t):
    textfile=cv2.FileStorage(f'Q2_image/Q2_lib/alphabet_lib_onboard.txt',cv2.FILE_STORAGE_READ)
    t=t.upper()
    objp = np.zeros((11*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
    objlist=[]
    imglist=[]

    imgshape=()
    for i in range(1,6):
        img = cv2.imread(f'Q2_Image/{i}.bmp')

        imgshape=img.shape[::-2]
        ret, corners=cv2.findChessboardCorners(img,(11,8),None)
        if ret==True:
            objlist.append(objp)
            imglist.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objlist, imglist, imgshape, None, None)
    for i in range(1,6):
        img = cv2.imread(f'Q2_Image/{i}.bmp')
        for j in range(len(t)):
            temp=textfile.getNode(t[j]).mat()
            for line in temp:
                line = shift_line(line, [7-j%3*3, 5-int(j/3)*3, 0])
                line = np.float32(line).reshape(-1,3)
                img_line, jac = cv2.projectPoints(line, rvecs[i-1], tvecs[i-1], mtx, dist)
                pt1=tuple(map(int,img_line[0].ravel()))
                pt2=tuple(map(int,img_line[1].ravel()))
                img = cv2.line(img, pt1, pt2, (0, 0, 255), 5)
        img = cv2.resize(img, (720, 720))
        cv2.imshow('12x9 chessborad', img)
        cv2.waitKey(500)
    cv2.destroyAllWindows()

#augmented_reality('opencv')
def show_vertical(t):
    textfile=cv2.FileStorage(f'Q2_image/Q2_lib/alphabet_lib_vertical.txt',cv2.FILE_STORAGE_READ)
    t=t.upper()
    objp = np.zeros((11*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
    objlist=[]
    imglist=[]

    imgshape=()
    for i in range(1,6):
        img = cv2.imread(f'Q2_Image/{i}.bmp')

        imgshape=img.shape[::-2]
        ret, corners=cv2.findChessboardCorners(img,(11,8),None)
        if ret==True:
            objlist.append(objp)
            imglist.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objlist, imglist, imgshape, None, None)
    for i in range(1,6):
        img = cv2.imread(f'Q2_Image/{i}.bmp')
        for j in range(len(t)):
            temp=textfile.getNode(t[j]).mat()
            for line in temp:
                line = shift_line(line, [7-j%3*3, 5-int(j/3)*3, 0])
                line = np.float32(line).reshape(-1,3)
                img_line, jac = cv2.projectPoints(line, rvecs[i-1], tvecs[i-1], mtx, dist)
                pt1=tuple(map(int,img_line[0].ravel()))
                pt2=tuple(map(int,img_line[1].ravel()))
                img = cv2.line(img, pt1, pt2, (0, 0, 255), 5)
        img = cv2.resize(img, (720, 720))
        cv2.imshow('12x9 chessborad', img)
        cv2.waitKey(500)
    cv2.destroyAllWindows()

#show_vertical('OPENCV')