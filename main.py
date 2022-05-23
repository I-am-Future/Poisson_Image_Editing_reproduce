import cv2
from preprocess_utils import rescale, generate_mask
from pie_functions import PIE

# rescale('imgs/1_original_src1.jpg', 'imgs/1_src1.jpg', (100, 400), 1)
# generate_mask('imgs/1_src1.jpg')

bkg_img = cv2.imread('imgs/1_bkg.jpg')
src_img = cv2.imread('imgs/1_src1.jpg')
src_mask = cv2.imread('imgs/1_src1_mask.jpg', 0)

PIE(bkg_img, src_img, src_mask, (400, 100), 'imgs/1_finala.jpg')

bkg_img = cv2.imread('imgs/1_finala.jpg')
src_img = cv2.imread('imgs/1_src2.jpg')
src_mask = cv2.imread('imgs/1_src2_mask.jpg', 0)

PIE(bkg_img, src_img, src_mask, (100, 100), 'imgs/1_final.jpg')
