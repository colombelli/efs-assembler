# efs-assembler

![](https://img.shields.io/badge/python-v3.8-blue)
![](https://img.shields.io/badge/R-v4.0-red)

## Introduction

The efs-assembler is a Python package integrated with R for performing ensemble feature selection experiments in binary classification problems. It is high flexible and allows the user to add different feature selection (this one includes support for R algorithms), aggregation and classification algorithms, offering options to perform stratified cross validation with downsampling and collect various performance metrics.

The package is backed by a scientific study about ensemble feature selection for biomarkers discovery in gene expression data. If you have scientific interests or want to use our package in formal reports, please cite us: xxxx

Currently, there are 4 types of experiments and because of the high flexibility, the input data structure requires special attention.

#### Feature Selection algorithms
* Gain Ratio - https://mi2-warsaw.github.io/FSelectorRcpp/reference/information_gain.html
* Symmetrical Uncertainty - https://mi2-warsaw.github.io/FSelectorRcpp/reference/information_gain.html
* ReliefF - https://pypi.org/project/ReliefF/
* GeoDE - http://www.maayanlab.net/CD/
* Wx (adapted) - https://github.com/deargen/DearWXpub

#### Aggregation algorithms
* Borda count
* Stability weightened aggregation 

For detailed information, check our article.

#### Classification algorithms

* Gradient Boost - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
* SVM - https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

#### Collected metrics

* Accuracy
* ROC-AUC
* PR-AUC
* Kuncheva Index (for method's stability measurement) - https://github.com/colombelli/kuncheva-index

## Installation 

Because of the package variety required by the provided feature selection algorithms, the installation steps still need to be further tested and investigated to make sure we cover more use cases and computer setups. Consider installing the current supported versions of Python (3.8) and R (4.0).

Install FSelectorRcpp:

    $ R
    > install.packages("FSelectorRcpp")
    
Install efs-assembler:  
    
    $ git clone https://github.com/colombelli/efs-assembler.git
    $ pip install -e efs-assembler
    
The python packages required by efs-assembler full funcionality are:
* pandas >= 1.0.5
* numpy >= 1.19.0
* rpy2 >= 3.3.4
* scikit-learn >= 0.23.1
* ReliefF >= 0.1.2
* tensorflow >= 2.2.0
* keras >= 2.4.3


