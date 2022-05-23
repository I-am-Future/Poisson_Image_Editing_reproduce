# Poisson_Image_Editing

A reproduce repository to the classic computer graphics paper: "Poisson Image Editing" (https://doi.org/10.1145/882262.882269)

## Demo

<img src="https://future-cos01-1312070282.cos.ap-guangzhou.myqcloud.com/%5Cphotos%5C1_final.jpg" alt="demo" style="zoom: 67%;" />

## Program Requirements

+ Programming Language:
`python 3.x`
+ Package Required:
`numpy`
`python-opencv` (`cv2`)
`scipy`

​		All packages above can be installed with `pip install <pachage name>`. 

## Usage

+ Pre-processing

​	In the file `preprocess_utils.py` there are some utility functions. 

1. `rescale()` can help you rescaling the image, and 

2. `generate_mask()` can help you generating a mask over the source image. Use mouse to draw the mask (it will appear when you leave the mouse left button), and use `s` to save the mask, use `q` to quit. 

+ Main processing

​	See `main.py` for the program: Pass the *background image* (array, with size $w \times h \times 3$), and *source image* (array, with size $w' \times h' \times 3$), and the *source mask* (array, with size $w'' \times h''$), with the *output file name* to the main function `PIE()`, and let it calculates. The process may last several to tens of seconds depending on your image size. 

​	All implementation are in `pie_functions.py`. 

