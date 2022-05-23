# the main code for the Poisson Image Editing Algorithm

import cv2
import numpy as np
import pie_utils

def PIE_kernel(bkg: np.ndarray, bkg_mask: np.ndarray, 
        src: np.ndarray, src_mask: np.ndarray):
    ''' PIE algorithm, for 1 channel processing. '''
    ...

def PIE(bkg: np.ndarray, src: np.ndarray, src_mask: np.ndarray, offset: tuple):
    ''' PIE algorithm that make `src` on `bkg`, with `offset` in the `bkg` image '''
    assert( len(offset) == 2 )

    Omega = np.zeros(bkg.shape[0: 2])     # backgrounding mask, As known as `\Omega`. 
    src_mask = np.where(src_mask > 128, 1, 0)  # changing src_mask to binary form
    Omega[offset[1]: offset[1]+src_mask.shape[0], offset[0]: offset[0]+src_mask.shape[1]] = src_mask
    Omega_bounder = pie_utils.get_bounder(Omega)

    # print(np.max(Omega))
    # print(np.min(Omega))
    # cv2.imshow('temp', Omega)
    # cv2.waitKey(0)
    print(np.max(Omega_bounder))
    print(np.min(Omega_bounder))
    cv2.imshow('temp', Omega_bounder)
    cv2.waitKey(0)
    return ...

bkg_img = cv2.imread('imgs/1_bkg.jpg')
src_img = cv2.imread('imgs/1_src1.jpg')
src_mask = cv2.imread('imgs/1_src1_mask.jpg', 0)
print(src_img.shape)
print(src_mask.shape)
PIE(bkg_img, src_img, src_mask, (100, 100))
