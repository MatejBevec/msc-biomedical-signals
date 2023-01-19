import os
import shutil
import math
import time
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import cv2 as cv


# CANNY EDGE DETECTOR

class CannyDetector():
    
    def __init__(self, blursize=5, sigma=1.2, sobelsize=3, weakthr=0.1, strongthr=0.25):
        self.blursize = blursize
        self.sigma = sigma
        self.sobelsize = sobelsize
        self.weakthr = weakthr
        self.strongthr = strongthr
    
    def _blur(self, img):
        sz = (self.blursize, self.blursize)
        return cv.GaussianBlur(img, sz, self.sigma)
    
    def _sobel(self, img):
        sobx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=self.sobelsize)
        soby = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=self.sobelsize)
        
        mag = np.sqrt(sobx**2 + soby**2)
        dr = np.arctan2(sobx, soby)
        dr = (np.round(dr/math.pi * 4 + 4.0) ).astype(int)
        return mag, dr
    
    def _threshold(self, mag, dr):
        
        dy = list(reversed([1, 1, 0, -1, -1, -1, 0, 1, 1]))
        dx= list(reversed([0, -1, -1, -1, 0, 1, 1, 1, 0]))
        
        edges = np.zeros_like(mag)
        
        for y in range(1, mag.shape[0]-1):
            for x in range(1, mag.shape[1]-1):
                i = dr[y, x]
                q = mag[y + dy[i], x + dx[i]]
                r =  mag[y - dy[i], x - dx[i]]
                if mag[y, x] >= q and mag[y, x] >= r:
                    edges[y, x] = mag[y, x]
                    
        return edges
    
    
    def _link(self, edges):
        strong = edges > self.strongthr
        weak = edges > self.weakthr
        
        ker = np.ones((3,3))
        for y in range(strong.shape[0]):
            for x in range(strong.shape[1]):
                is_weak = weak[y, x]
                strongnbh = strong[y-1:y+2, x-1:x+2]
                is_strong = np.sum(strongnbh) * is_weak
                if is_strong > 0:
                    strong[y, x] = True
        return strong
    
    def __call__(self, img):
        img = self._blur(img)
        mag, dr = self._sobel(img)
        edges = self._threshold(mag, dr)
        linked = self._link(edges)
        return linked, edges


def showimg(img, title=""):
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.title(title)


def loadimg(pth): 
    img_int = cv.cvtColor(cv.imread(pth), cv.COLOR_BGR2GRAY)
    img = img_int.astype(float) / np.max(img_int)
    return img



if __name__ == "__main__":

    img = loadimg(sys.argv[1])

    canny = CannyDetector(blursize=9, sigma=2.5, sobelsize=3, weakthr=0.04, strongthr=0.08)

    linked, edges = canny(img)

    plt.subplot(1, 3, 1); showimg(img)
    plt.subplot(1, 3, 2); showimg(edges)
    plt.subplot(1, 3, 3); showimg(linked)
    plt.show()

    linked = np.uint8(linked*255)
    cv.imwrite(sys.argv[1].rsplit(".")[-2] + "_edges.png", linked)
    



        
        