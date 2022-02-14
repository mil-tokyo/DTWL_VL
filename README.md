# SDTW-VL
Code for "Generation of Variable-Length Time Series from Text using Dynamic Time Warping-Based Method".
The codes for implementation stated in the paper, and the code for creating dataset.


We refer the code at https://github.com/vincent-leguen/DILATE, which is code for the NeurIPS 2019 paper "Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models" (Le Guen, Vincent and Thome, Nicolas), for "DILATE_VL_pad" and "DILATE_VL_active".

We refer to https://github.com/jadore801120/attention-is-all-you-need-pytorch, for the structure of the model. 


Our proposed loss is implemented in DTWL_VL/SDTWVL.py

Our environment is below.

torch:  1.6.0+cu101

numpy:  1.17.2 

Python: 3.7.4


You can start training by entering the folder ("MSEVariant", "DILATE_VL_active", "DILATE_VL_pad", "DTWL_VL") and
running "python Models.py" .
