import cv2
import numpy as np
from skimage.morphology import skeletonize, thin


"""
Hough transform:

input:
img: PIL image
morphology: morphological transform

output:
lineP: set of lines
"""
def hough_transform(img, morphology=True):
    # PIL to numpy
    img_rgb = np.array(img)
    img_gray = np.array(img.convert("L"))
    
    # edge map
    edges = cv2.Canny(img_gray,50,200,None,apertureSize = 3)
    
    # morphological transform
    if morphology:
        # cloing for inpaint
        kernel = np.ones((3,3),np.uint8)
        closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        closing[closing>0] = 1
        
        # extract skeleton
        skeleton = skeletonize(closing)
        post_process = np.zeros(skeleton.shape)
        post_process[skeleton==True] = 255
        post_process = np.uint8(post_process)
        kernel = np.ones((2,2),np.uint8)
        
        # dilation from skeleton
        post_process_dilation = cv2.dilate(post_process,kernel,iterations = 1)
        
        # probabilistic hough transform
        linesP = cv2.HoughLinesP(post_process_dilation, 1, np.pi / 180, 50, None, 50, 10)
    else:
        # probabilistic hough transform
        linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)

    return linesP 
    
    