# LRPCA

This repository is for our paper:

[1] HanQin Cai, Jialin Liu, and Wotao Yin. Learned Robust PCA: A Scalable Deep Unfolding Approach for High-Dimensional Outlier Detection. In *Advances in Neural Information Processing Systems*, 2021.

Problem description
===================
Given Y = X + S, where X is the underlying low-rank matrix, S is the ourlier, we want to recover X from Y. 

Files description
=================
1. "synthetic_data_exp" involves our codes for the synthetic-data experiments[^1].
[^1]: Other parts will be released soon.

First Time to Run
=================
* Enter "synthetic_data_exp" and run "testing_codes_matlab.m" directly.
* The test script will call a trained model stored in "synthetic_data_exp/trained_models"

Training the Model
==================
* Enter "synthetic_data_exp" and run "training_codes.py" directly.
* The training script will write the model into a ".mat" file that the test script can load.

Dependencies
============
* Testing codes: MATLAB (>= 2017b)
* Training codes: CUDA 11.0; pytorch 1.7.1

