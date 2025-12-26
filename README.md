**Boosting Semi-Supervised Medical Image Segmentation through Inter-Instance Information Complementarity**

Shuai Wu, Ruyi Liu, Hang Wei, Linrunjia Liu, Jie Wen, Qiguang Miao

**Introduction**

Official code for “Boosting Semi-Supervised Medical Image Segmentation through Inter-Instance Information Complementarity”. (IEEE TNNLS 2026)

**Requirements**

This repository is based on PyTorch 1.8.0, CUDA 11.1, and Python 3.6.13. All experiments in our paper were conducted on an NVIDIA GeForce RTX 3060 GPU under identical experimental settings.

**Usage**

\
To train a model, LA or BraTs

python ./code/ LA\_All\_train.py

python ./code/ LA\_CPAM\_train.py

python ./code/ BraTS\_All\_train.py

python ./code/ BraTS\_CPAM\_train.py

To test a model, LA or BraTs

`  `python ./code/test\_LA.py 

python ./code/test\_ BraTS.py 

**Acknowledgements**

Our code is largely based on <https://github.com/DeepMed-Lab-ECNU/BCP>.  Thanks for these authors for their valuable work, hope our work can also contribute to related research.
