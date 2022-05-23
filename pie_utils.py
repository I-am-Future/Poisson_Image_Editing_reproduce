# This file contains some utilities for the main pie processing of the images: 

import numpy as np
from typing import Tuple

def get_neighbor(h, w, max_h, max_w) -> set:
    ''' Returns the neighbor index array of given point (h, w) '''
    res = [(h-1, w), (h+1, w), (h, w-1), (h, w+1)]
    valid = [True] * 4
    retval = set()
    if h <= 0:          valid[0] = False
    elif h >= max_w-1:  valid[1] = False
    if w <= 0:          valid[2] = False
    elif w >= max_h-1:  valid[3] = False
    for i in range(4):
        if valid[i]:
            retval.add(res[i])
    return retval

def get_bounder(omega: np.ndarray) -> np.ndarray:
    ''' Get the bounder over a region `omega`. '''
    bounder = np.zeros(omega.shape[0:2])
    bounder_set = set()
    for h in range(omega.shape[0]):
        for w in range(omega.shape[1]):
            if omega[h, w] == 1:
                continue
            neighbors = get_neighbor(h, w, omega.shape[1], omega.shape[0])
            for neighbor in neighbors:
                if omega[neighbor[0], neighbor[1]] == 1:
                    bounder[h, w] = 1
                    bounder_set.add((h, w))
                    break
    return bounder, bounder_set

def omegamask_to_set(mask: np.ndarray) -> Tuple[set, dict]:
    ''' Return a set contains all index pairs of the mask. '''
    result = set()
    pos2idx = dict()
    idx2pos = dict()
    count = 0
    for h in range(mask.shape[0]):
        for w in range(mask.shape[1]):
            if mask[h, w] == 1:
                result.add((h, w))
                pos2idx[(h, w)] = count
                idx2pos[count] = (h, w)
                count += 1
    return result, pos2idx, idx2pos

def guide_vec(p_h: int, p_w: int, q_h: int, q_w: int, 
        bkg: np.ndarray, src: np.ndarray, offset: tuple) -> int:
    ''' A function that gives guide `v_{pq}` '''
    assert(abs(p_h-q_h) <= 1)
    assert(abs(p_w-q_w) <= 1)
    src_p_h = p_h - offset[0]
    src_p_w = p_w - offset[1]
    src_q_h = q_h - offset[0]
    src_q_w = q_w - offset[1]
    try:
        # test if src's index is legal (not at edge)
        int(src[src_p_h, src_p_w]) - int(src[src_q_h, src_q_w])
    except:
        # return f* 's gradient
        return int(bkg[p_h, p_w]) - int(bkg[q_h, q_w])
    else:
        # return with the larger norm one
        if (abs(int(bkg[p_h, p_w]) - int(bkg[q_h, q_w])) > 
                abs(int(src[src_p_h, src_p_w]) - int(src[src_q_h, src_q_w]))):
            return int(bkg[p_h, p_w]) - int(bkg[q_h, q_w])
        else:
            return int(src[src_p_h, src_p_w]) - int(src[src_q_h, src_q_w])

# testing
# print(get_neighbor(10, 10, 20, 20)) # middle
# print(get_neighbor(0, 0, 20, 20))   # left top
# print(get_neighbor(19, 0, 20, 20))  # right top
# print(get_neighbor(0, 19, 20, 20))  # left bottom
# print(get_neighbor(19, 19, 20, 20)) # right bottom
# print(get_neighbor(10, 0, 20, 20))  # top edge
# print(get_neighbor(10, 19, 20, 20)) # bottom edge
# print(get_neighbor(0, 10, 20, 20))  # left edge
# print(get_neighbor(19, 10, 20, 20)) # right edge
