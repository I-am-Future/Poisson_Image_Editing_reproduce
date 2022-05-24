# the main code for the Poisson Image Editing Algorithm

import cv2
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
import pie_utils


def PIE_kernel(
        bkg: np.ndarray,        # the background image
        omega: np.ndarray,      # the overwrite region `Omega`
        omega_set: set,         # the region `Omega`, in set(<y, x>) form
        pos2idx: dict,          # dict to remember position-index mapping
        idx2pos: dict,          # dict to remember index-position mapping
        omega_bounder_set: set, # `Omega` boundary, in set(<y, x>) form
        src: np.ndarray,        # source (overwritten) image
        offset: tuple,          # relative offset between source image and background. 
        pickled: bool = False   # whether save the sol `x` or not
    ) -> None:
    ''' PIE algorithm kernel function, for 1 channel processing. '''
    bkg = bkg.copy()
    A = lil_matrix(np.zeros((len(omega_set), len(omega_set))))
    b = np.zeros((len(omega_set), 1))
    for i, (p_h, p_w) in enumerate(omega_set):
        # each iteration is one `p`. pos at p_h, p_w, and matrix index at p_index
        p_neighbors = pie_utils.get_neighbor(p_h, p_w, omega.shape[0], omega.shape[1])
        ## first term in equation: 
        A[i, pos2idx[(p_h, p_w)]] = len(p_neighbors)
        ## second term in equation:
        for (q_h, q_w) in omega_set.intersection(p_neighbors):
            A[i, pos2idx[(q_h, q_w)]] = -1
        ## rhs
        rhs = 0
        if not omega_set.issuperset(p_neighbors): 
            # can ignore this term when omega contain p_neighbors entirely, otherwise do it
            for (q_h, q_w) in omega_bounder_set.intersection(p_neighbors):
                rhs += int(bkg[q_h, q_w])
        for (q_h, q_w) in p_neighbors:
            rhs += pie_utils.guide_vec(p_h, p_w, q_h, q_w, bkg, src, offset)
        b[i, 0] = rhs
    print('Equation construction finished!')
    x = spsolve(csc_matrix(A), b)
    if pickled:
        np.save('x.npy', x)
    x = np.maximum(x, 0)
    x = np.minimum(x, 255)
    for i in range(len(x)):
        pos = idx2pos[i]
        bkg[pos[0], pos[1]] = np.uint8(x[i])
    print('Linear equation solved!')
    return bkg

def PIE(bkg: np.ndarray, src: np.ndarray, 
        src_mask: np.ndarray, offset: tuple, save_fname: str):
    ''' PIE algorithm that make `src` on `bkg`, with `offset` of `src` in the `bkg` image 
        The file will be saved in `save_fname.jpg`.
    '''
    assert( len(offset) == 2 )
    assert( len(bkg.shape) == 3 )
    assert( len(src.shape) == 3 )
    assert( len(src_mask.shape) == 2 )
    print('Input image size:', bkg.shape)

    ## Omega: the background mask (matrix form)
    Omega = np.zeros(bkg.shape[0: 2])     # backgrounding mask, As known as `\Omega`. 
    src_mask = np.where(src_mask > 128, 1, 0)  # changing src_mask to binary form
    Omega[offset[0]: offset[0]+src_mask.shape[0], 
            offset[1]: offset[1]+src_mask.shape[1]] = src_mask
    ## Omega_set: the background mask's index (set form)
    Omega_set, pos2idx, idx2pos = pie_utils.omegamask_to_set(Omega)
    ## Omega_bounder, Omega_bounder_set: the bounder (`\pratial \Omega`), 2 forms
    Omega_bounder, Omega_bounder_set = pie_utils.get_bounder(Omega)
    print('Omega set size:', len(Omega_set))
    print('Omega set perimeter:', len(Omega_bounder_set))
    b = PIE_kernel(bkg[:, :, 0], Omega, Omega_set, pos2idx, idx2pos, 
            Omega_bounder_set, src[:, :, 0], offset)
    g = PIE_kernel(bkg[:, :, 1], Omega, Omega_set, pos2idx, idx2pos, 
            Omega_bounder_set, src[:, :, 1], offset)
    r = PIE_kernel(bkg[:, :, 2], Omega, Omega_set, pos2idx, idx2pos, 
            Omega_bounder_set, src[:, :, 2], offset)
    img = np.stack((b, g, r), axis = 2)

    cv2.imshow('temp view', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(save_fname, img)


