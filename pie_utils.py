# This file contains some utilities for the main pie processing of the images: 

import numpy as np
import cv2

def get_neighbor(x, y, w, h) -> np.ndarray:
    ''' Returns the neighbor index array of given point (x, y) '''
    res = np.array([[x-1, y], [x+1, y], [x, y-1], [x, y+1]])
    valid = np.array([True] * 4)
    if x <= 0:
        valid[0] = False
    elif x >= w-1:
        valid[1] = False
    if y <= 0:
        valid[2] = False
    elif y >= h-1:
        valid[3] = False
    return res[valid]

def get_bounder(omega: np.ndarray) -> np.ndarray:
    ''' Get the bounder over a region `omega`. '''
    bounder = np.zeros(omega.shape[0:2])
    for h in range(omega.shape[0]):
        for w in range(omega.shape[1]):
            if omega[h, w] == 1:
                continue
            neighbors = get_neighbor(w, h, omega.shape[1], omega.shape[0])
            for neighbor in neighbors:
                if omega[neighbor[1], neighbor[0]] == 1:
                    bounder[h, w] = 1
                    break
    return bounder

# testing
print(get_neighbor(10, 10, 20, 20)) # middle
print(get_neighbor(0, 0, 20, 20))   # left top
print(get_neighbor(19, 0, 20, 20))  # right top
print(get_neighbor(0, 19, 20, 20))  # left bottom
print(get_neighbor(19, 19, 20, 20)) # right bottom
print(get_neighbor(10, 0, 20, 20))  # top edge
print(get_neighbor(10, 19, 20, 20)) # bottom edge
print(get_neighbor(0, 10, 20, 20))  # left edge
print(get_neighbor(19, 10, 20, 20)) # right edge
