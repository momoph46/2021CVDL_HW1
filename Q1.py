import cv2
import numpy as np


def findCorners():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for i in range(1, 16):
        img = cv2.imread(f"Q1_image/{i}.bmp")
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray_img, (11, 8), None)
        if ret == True:
            #corners2 = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(img, (11, 8), corners, ret)
            img = cv2.resize(img, (960, 480))
            cv2.imshow('12x9 chessborad', img)
            cv2.waitKey(500)
    cv2.destroyAllWindows()

def find_instrinsic():
    oplist=[]
    iplist=[]
    objp = np.zeros((11*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
    imgshape=()
    for i in range(1, 16):
        img = cv2.imread(f"Q1_image/{i}.bmp")
        imgshape=img.shape[::-2]
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray_img, (11, 8),None)
        if ret :
            oplist.append(objp)
            iplist.append(corners)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(oplist, iplist, imgshape, None, None)
    print("Intrinsic:")
    print(mtx)

def find_extrinsic(idx):
    objp = np.zeros((11*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
    
    img = cv2.imread(f"Q1_image/{idx}.bmp")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray_img, (11, 8), None)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp], [corners], gray_img.shape[::-1], None, None) 
    rvecs = cv2.Rodrigues(rvecs[0])
    print(rvecs[0])

def find_distorsion():
    objp = np.zeros((11*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
    oplist=[]
    iplist=[]
    imgshape=()
    for i in range(1, 16):
        img = cv2.imread(f"Q1_image/{i}.bmp")
        imgshape=img.shape[::-2]
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray_img, (11, 8), None)
        if ret :
            oplist.append(objp)
            iplist.append(corners)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(oplist, iplist, imgshape, None, None)
    print("Distortion:")
    print(dist)

def Show_Result():
    objp = np.zeros((11 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
    imgshape=()
    objpoints = []
    imgpoints = []
    for i in range(1,16):
        img=cv2.imread(f"Q1_Image/{i}.bmp")
        imgshape=img.shape[::-2]
        ret, corners = cv2.findChessboardCorners(img, (11,8))
        if ret :
            objpoints.append(objp)
            imgpoints.append(corners)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imgshape, None, None)

    for i in range(1,16):
        img = cv2.imread(f"Q1_Image/{i}.bmp")
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        dst = np.append(dst, img, axis=1)
        dst = cv2.resize(dst, (960, 480))
        cv2.imshow('12x9 chessborad', dst)
        cv2.waitKey(500)

    cv2.destroyAllWindows()
#findCorners()
#find_instrinsic()
#find_extrinsic(5)
#find_distorsion()
#Show_Result()