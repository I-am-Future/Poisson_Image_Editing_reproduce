# This file contains some utilities for pre-processing the images: 
#  - background image: 
#    = re-scaling
#  - source image: 
#    = mask generating
#    = re-scaling

import cv2
import numpy as np


def rescale(in_fname: str, out_fname: str, new_size: tuple, domain_axis: int) -> None:
    ''' A utility function for rescaling the image, encapsulated from `cv2.resize`. 
        @param in_fname: <str>, the input filename
        @param out_fname: <str>, the input filename
        @param new_size: <tuple(height, width)>, the new size (pixels) of the image
        @param domain_axis: For free rescaling, set to -1. For equal-ratio rescaling, set to 0/1.  \
            For example, if you want new_size depends on axis=0, set it to 0.  \
            Then axis1 will be automatically adjusted by axis0.
    '''
    assert( len(new_size) == 2 )
    img = cv2.imread(in_fname)
    if domain_axis == -1:
        out_img = cv2.resize(img, new_size)
    elif domain_axis == 0:
        fy = new_size[0] / img.shape[0]
        out_img = cv2.resize(img, None, fx=fy, fy=fy)
    elif domain_axis == 1:
        fx = new_size[1] / img.shape[1]
        out_img = cv2.resize(img, None, fx=fx, fy=fx)
    cv2.imwrite(out_fname, out_img)


def generate_mask(in_fname: str) -> None:
    ''' A utility function to generate a mask of a given image. '''

    def callback(event, x, y, flags, param):
        ''' Callback function of the window. '''
        if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            cv2.circle(mask, (x, y), 8, (255, 255, 255), -1)
            cv2.circle(img, (x, y), 8, (255, 255, 255), -1)
            # print(x, y)
            cv2.imshow('mask gen', img)
        # elif event == cv2.EVENT_LBUTTONUP:
        #     cv2.destroyWindow('mask gen')
        #     cv2.namedWindow('mask gen')
        #     cv2.setMouseCallback('mask gen',callback)

    img = cv2.imread(in_fname)
    h, w, _ = img.shape
    mask = np.zeros((h, w, 3), dtype=np.uint8)

    cv2.namedWindow('mask gen')
    cv2.setMouseCallback('mask gen',callback)
    cv2.imshow('mask gen', img)

    while(True):
        if cv2.waitKey(20)&0xFF==ord('s'):
            cv2.destroyAllWindows()
            cv2.imwrite(in_fname.split('.')[0] + '_mask.jpg', mask)
            break
        elif cv2.waitKey(20)&0xFF==ord('q'):
            cv2.destroyAllWindows()
            break

# USE samples:
# rescale('imgs/1_original_src1.jpg', 'imgs/1_src1.jpg', (100, 400), 1)
generate_mask('imgs/1_src1.jpg')
