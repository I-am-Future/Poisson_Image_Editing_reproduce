# the main code for the Poisson Image Editing Algorithm

import cv2
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
import pie_utils

def PIE_kernel(bkg: np.ndarray, 
        omega: np.ndarray, omega_set: set, pos2idx: dict, idx2pos: dict,
        omega_bounder: np.ndarray, omega_bounder_set: set, 
        src: np.ndarray, src_mask: np.ndarray, offset: tuple, f):
    ''' PIE algorithm, for 1 channel processing. '''
    A = np.zeros((len(omega_set), len(omega_set)))
    A = lil_matrix(A)
    b = np.zeros((len(omega_set), 1))
    for i, (p_h, p_w) in enumerate(omega_set):
        # each iteration is one `p`. pos at p_h, p_w, and matrix index at p_index
        p_neighbors = pie_utils.get_neighbor(p_h, p_w, omega.shape[0], omega.shape[1])
        ## first term in equation: 
        p_index = pos2idx[(p_h, p_w)]
        A[i, p_index] = len(p_neighbors)
        ## second term in equation:
        for (q_h, q_w) in omega_set.intersection(p_neighbors):
            q_index = pos2idx[(q_h, q_w)]
            A[i, q_index] = -1
        ## rhs
        rhs = 0
        for (q_h, q_w) in omega_bounder_set.intersection(p_neighbors):
            rhs += int(bkg[q_h, q_w])
        for (q_h, q_w) in p_neighbors:
            rhs += pie_utils.guide_vec(p_h, p_w, q_h, q_w, bkg, src, offset)
        b[i, 0] = rhs
    print('construct finished!')
    # x = np.linalg.solve(A, b)
    x = spsolve(csc_matrix(A), b)
    print(x.shape)
    np.save(f, x)
    # x = np.load(f)
    # x = (x - x.min()) / (x.max() - x.min()) * 255
    x = np.maximum(x, 0)
    x = np.minimum(x, 255)
    for i in range(len(x)):
        pos = idx2pos[i]
        bkg[pos[0], pos[1]] = np.uint8(x[i])
    # cv2.imshow('output', bkg)
    # cv2.waitKey(0)
    return bkg

def PIE(bkg: np.ndarray, src: np.ndarray, src_mask: np.ndarray, offset: tuple, save_fname: str):
    ''' PIE algorithm that make `src` on `bkg`, with `offset` in the `bkg` image '''
    assert( len(offset) == 2 )

    ## Omega: the background mask (matrix form)
    Omega = np.zeros(bkg.shape[0: 2])     # backgrounding mask, As known as `\Omega`. 
    src_mask = np.where(src_mask > 128, 1, 0)  # changing src_mask to binary form
    Omega[offset[0]: offset[0]+src_mask.shape[0], 
            offset[1]: offset[1]+src_mask.shape[1]] = src_mask
    ## Omega_set: the background mask's index (set form)
    Omega_set, pos2idx, idx2pos = pie_utils.omegamask_to_set(Omega)
    ## Omega_bounder, Omega_bounder_set: the bounder (`\pratial \Omega`), 2 forms
    Omega_bounder, Omega_bounder_set = pie_utils.get_bounder(Omega)
    print(len(Omega_set))
    print(len(Omega_bounder_set))
    x = PIE_kernel(bkg[:, :, 0], Omega, Omega_set, pos2idx, idx2pos, 
            Omega_bounder, Omega_bounder_set, src[:, :, 0], src_mask, offset, 'x.npy')
    y = PIE_kernel(bkg[:, :, 1], Omega, Omega_set, pos2idx, idx2pos, 
            Omega_bounder, Omega_bounder_set, src[:, :, 1], src_mask, offset, 'y.npy')
    z = PIE_kernel(bkg[:, :, 2], Omega, Omega_set, pos2idx, idx2pos, 
            Omega_bounder, Omega_bounder_set, src[:, :, 2], src_mask, offset, 'z.npy')
    img = np.stack((x, y, z), axis = 2)
    print(img.shape)

    # print(np.sum(Omega_bounder))
    # print(len(Omega_bounder_set))
    # print(np.max(Omega))
    # print(np.min(Omega))
    # cv2.imshow('temp', Omega)
    # cv2.waitKey(0)
    print(np.max(Omega_bounder))
    print(np.min(Omega_bounder))
    cv2.imshow('temp', img)
    # cv2.imshow('temp', Omega_bounder)
    cv2.waitKey(0)
    cv2.imwrite(save_fname, img)
    return ...

bkg_img = cv2.imread('imgs/1_bkg.jpg')
src_img = cv2.imread('imgs/1_src1.jpg')
src_mask = cv2.imread('imgs/1_src1_mask.jpg', 0)
print(src_img.shape)
print(src_mask.shape)
PIE(bkg_img, src_img, src_mask, (400, 100), 'imgs/1_finala.jpg')

bkg_img = cv2.imread('imgs/1_finala.jpg')
src_img = cv2.imread('imgs/1_src2.jpg')
src_mask = cv2.imread('imgs/1_src2_mask.jpg', 0)

PIE(bkg_img, src_img, src_mask, (100, 100), 'imgs/1_final.jpg')
