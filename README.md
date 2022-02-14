# Generation of Variable-Length Time Series from Text using Dynamic Time Warping-Based Method

### Ayaka Ideno, Yusuke Mukuta, and Tatsuya Harada

In our [paper](https://dl.acm.org/doi/10.1145/3469877.3495644), we propose loss "DTW-like method for variable-length data (DTWL_VL)".

This repository has the code for "Generation of Variable-Length Time Series from Text using Dynamic Time Warping-Based Method".
The codes for implementation stated in the paper, and the code for creating dataset.



##  Environments we use



Our environment is below.

torch:  1.6.0+cu101

numpy:  1.17.2 

Python: 3.7.4


##  Training Example
You can start training by entering the folder ("MSEVariant", "DILATE_VL_active", "DILATE_VL_pad", "DTWL_VL") and
running "python Models.py".

Our proposed loss is implemented in DTWL_VL/SDTWVL.py .


## Acknowledgements 

We refer the code at https://github.com/vincent-leguen/DILATE, which is code for the NeurIPS 2019 paper "Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models" (Le Guen, Vincent and Thome, Nicolas), for "DILATE_VL_pad" and "DILATE_VL_active".

We refer to https://github.com/jadore801120/attention-is-all-you-need-pytorch, for the structure of the model. 

The license file of https://github.com/jadore801120/attention-is-all-you-need-pytorch is in the folder of each settings("MSEVariant", "DILATE_VL_active", "DILATE_VL_pad", "DTWL_VL"), and license file of https://github.com/vincent-leguen/DILATE is in the folder named "loss" in the folder which includes the code related to the repository("DILATE_VL_active", "DILATE_VL_pad").
