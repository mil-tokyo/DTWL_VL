# Generation of Variable-Length Time Series from Text using Dynamic Time Warping-Based Method

### Ayaka Ideno, Yusuke Mukuta, and Tatsuya Harada

In our [paper](https://dl.acm.org/doi/10.1145/3469877.3495644), we propose loss "DTW-like method for variable-length data (DTWL-VL)".

This repository has the code for "Generation of Variable-Length Time Series from Text using Dynamic Time Warping-Based Method".
The codes for implementation stated in the paper, and the code for creating dataset.



##  Environments we use



Our environment is below.

torch:  1.6.0

numpy:  1.17.2 

Python: 3.7.4

## Preparation 
The path 
"../glove2/glove.6B.50d.txt" 
in "withtime_batch_dataload_inside.py" is the path of GloVe(Jeffrey Pennington, Richard Socher, and Christopher Manning. 2014. GloVe: Global Vectors for Word Representation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, Doha, Qatar, 1532–1543. https://doi.org/10.3115/v1/D14-1162) embedding file.

You need to download it from https://nlp.stanford.edu/data/glove.6B.zip (Jeffrey Pennington and Richard Socher and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation.) and put it at the place indicated the relative path, or edit the path ""../glove2/glove.6B.50d.txt"" written in "withtime_batch_dataload_inside.py" for GloVe file for yourself.


Also, you need to create the data for this experiment by running the code "textmake_withtime.py" in "DataCreation_40".
You need to put the folder where the same folder the folder for the experiment exists, or edit the path "../DatasetCreationCode_40" in "Models.py"
 for every experiment data.

##  Training Example
You can start training by entering the folder ("MSEVariant", "DILATE_VL_active", "DILATE_VL_pad", "DTWL_VL") and
running `python Models.py`.


Our proposed loss is implemented in DTWL_VL/SDTWVL.py .


## Reference and Acknowledgements 

We refer the code at https://github.com/vincent-leguen/DILATE, which is code for the NeurIPS 2019 paper "Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models" (Le Guen, Vincent and Thome, Nicolas), for "DILATE_VL_pad" and "DILATE_VL_active".

We refer to https://github.com/jadore801120/attention-is-all-you-need-pytorch, for the structure of the model. 

The license file of https://github.com/jadore801120/attention-is-all-you-need-pytorch is in the folder of each settings("MSEVariant", "DILATE_VL_active", "DILATE_VL_pad", "DTWL_VL"), and license file of https://github.com/vincent-leguen/DILATE is in the folder named "loss", which includes the code related to the repository(In "DILATE_VL_active", "DILATE_VL_pad").

The code for the model architecture (Written in "Models.py", "transformer/Layers.py", "transformer/Modules.py", "transformer/SubLayers.py" in all the training code folder) is based on the implementation of https://github.com/jadore801120/attention-is-all-you-need-pytorch, and
the code for DILATE loss ("loss/path_soft_dtw.py", "loss/dilate_loss.py", "loss/soft_dtw.py" in "DILATE_VL_active", "DILATE_VL_pad") is based on https://github.com/vincent-leguen/DILATE.
