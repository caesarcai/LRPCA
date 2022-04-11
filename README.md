# LRPCA

This repository is for our paper:

[1] HanQin Cai, Jialin Liu, and Wotao Yin. <a href=https://papers.nips.cc/paper/2021/hash/8d2355364e9a2ba1f82f975414937b43-Abstract.html>Learned Robust PCA: A Scalable Deep Unfolding Approach for High-Dimensional Outlier Detection</a>. In *Advances in Neural Information Processing Systems*, 34: 16977-16989, 2021.

###### To display math symbols properly, one may have to install a MathJax plugin. For example, [MathJax Plugin for Github](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima?hl=en).

#### Problem description

Given $Y = X + S$ where $X$ is the underlying low-rank matrix and $S$ is the sparse ourlier matrix, robust principal component analysis (RPCA) aims to recover $X$ and/or $S$ from the observed $Y$, depending on the application. 

#### Files description

* `synthetic_data_exp` involves our codes for the synthetic-data experiments[^1].
[^1]: Other parts will be released soon.

#### First Time to Run

* Enter `synthetic_data_exp` and run `testing_codes_matlab.m` directly.
* The test script will call a trained model stored in `synthetic_data_exp/trained_models`.

#### Training the Model

* Enter `synthetic_data_exp` and run `training_codes.py` directly.
* The training script will write the model into a `.mat` file that the test script can load.

#### Dependencies

* Testing codes: MATLAB (>= 2017b)
* Training codes: CUDA 11.0; pytorch 1.7.1

