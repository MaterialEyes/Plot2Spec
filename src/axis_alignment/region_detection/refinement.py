import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from .hough_transform import hough_transform



def find_best_lines(hlines, vlines, bbox, max_dist):
    x1,y1,x2,y2 = bbox
    if len(hlines) == 0:
        hline = [x1,y2, x2,y2]
    else:
        dist = [abs((line[1]+line[3])/2-y2) for line in hlines]
        if dist[np.argmin(dist)] > max_dist:
            hline = [x1,y2, x2,y2]
        else:
            hline = hlines[np.argmin(dist)]
    if len(vlines) == 0:
        vline = [x1,y1, x1,y2]
    else:
        dist = [abs((line[0]+line[2])/2-x1) for line in vlines]
        if dist[np.argmin(dist)] > max_dist:
            vline = [x1,y1, x1,y2]
        else:
            vline = vlines[np.argmin(dist)]
    return hline, vline


def refine(img, bbox, 
           len_threshold=0.5, 
           angle_threshold=0.98, 
           max_dist = 30, 
           morphology=True):
    
    linesP = hough_transform(img, morphology)
    x1,y1,x2,y2 = bbox
    
    hlines, vlines = [], []
    h_vec, v_vec = np.array([1,0]).reshape(1,-1), np.array([0,1]).reshape(1,-1)
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        dx = abs(l[2]-l[0])
        dy = abs(l[3]-l[1])
        vec = np.array([dx,dy]).reshape(1,-1)
        if dx > len_threshold*(x2-x1) and abs(cosine_similarity(h_vec, vec)) > angle_threshold:
            hlines.append(l)
        if dy > len_threshold*(y2-y1) and abs(cosine_similarity(v_vec, vec)) > angle_threshold:
            vlines.append(l)
            
    hline, vline = find_best_lines(hlines, vlines, bbox, max_dist)
    re_y2 = (hline[1]+hline[3])/2
    re_x1 = (vline[0]+vline[2])/2
    return re_x1, y1, x2, re_y2
    