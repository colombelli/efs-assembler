# efs-assembler

![](https://img.shields.io/badge/python-v3.8-blue)
![](https://img.shields.io/badge/R-v4.0-red)

## Summary

- [Introduction](#introduction)
    - [Feature Selection algorithms](#feature-selection-algorithms)
    - [Aggregation algorithms](#aggregation-algorithms)
    - [Classification algorithms](#classification-algorithms)
    - [Collected metrics](#collected-metrics)
- [Installation](#installation)
- [Usage for running experiments](#usage-for-running-experiments)
    - [Example](#experiments-example)
- [Datasets expected format](#datasets-expected-format)
- [Results folder structure](#results-folder-structure)
- [Usage for running feature extraction](#usage-for-running-feature-extraction)

## Introduction

The efs-assembler is a Python package integrated with R for performing ensemble feature selection experiments in binary classification problems. It is high flexible and allows the user to add different algorithms of feature selection (including support for R language), aggregation and also classification, offering options to perform stratified cross validation with downsampling and collect various performance metrics.

The package is backed by a scientific study about ensemble feature selection for cancer biomarkers discovery in gene expression data. If you have scientific interests or want to use our package in formal reports, please cite us with: xxxx

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


## Usage for running experiments

The available experiments comprehend: Homogeneous Ensemble Feature Selection (hom), Heterogeneous Ensemble Feature Selection (het), Single Feature Selection (sin) and our proposed ensemble design, the Hybrid Ensemble Feature Selection (hyb). These types of experiment are explained in detail in our article. 

```python
from efsassembler import Experiments
exp = Experiments(experiments_list, "my/results/path/")
exp.run()
```

The expected type of input for the ```experiments_list``` object is a list of dictionaries, where each dictionary represents an experiment type. If multiple datasets are provided, multiple experiments of that type will be executed, each experiment using each of the provided datasets. The expected keys and values for each dictionary are:
* ```"type"```: either ```"hom"```, ```"het"```, ```"sin"``` or ```"hyb"```
* ```"thresholds"```: list of integer thresholds or percentages of features to consider, e.g., ```[3, 5, 10, 15]``` that would select the top 3 features, top 5 features, top 10 features and top 15 features; or ```[0.1, 0.2, 0.5]``` that would select the top 10% features, top 20% features and top 50% features
* ```"seed"```: an integer representing the seed for reproducibility, e.g., ```42```
* ```"folds"```: an integer representing the number of folds for the stratified cross-validation, e.g., ```10```
* ```"undersampling"```: a boolean indicating if the stratified cross-validation is to be performed with undersampling
* ```"balanced_final_selection"```: a boolean indicating if the final feature selection is to be applied in [balanced dataset folds](#balanced_folds)
* ```"classifier"```: either ```"gbm"``` or ```"svm"``` for provided classification algorithms
* ```"datasets"```: a list with dataset paths that are going to be exposed to the experiment, e.g., ```["data/set/one.csv", "data/set/two.rds"]```. The accepted file types for the datasets are .csv and .rds. Additional information about the expected dataset format is given on [this section](#datasets-expected-format).
* ```"rankers"```: a list with the feature selection algorithms to be used (even in "sin" and "hom" experiments a list is expected). The feature selection algorithms are represented by a tuple in the format ("file_name", "language", "rank_file_name_to_use_for_saving_their_result"), e.g., ```[("reliefF", "python", "rf"), ("geoDE", "python", "gd"), ("gain-ratio", "r", "gr"), ("symmetrical-uncertainty", "r", "su"), ("wx", "python", "wx")]```. Those are the current available algorithms. If more than one is given for a "sin" or "hom" experiment, the first algorithm will be used.

For **"hyb"**, **"het"** and **"hom"** experiments:
* ```"aggregators"```: a list with the aggregator algorithms to be used. Since only python algorithms are supported, only the file name of the algorithm is required, e.g., ```["stb_weightened_layer1", "borda"]```. If it is a "het" or "hom" experiment only the first aggregator will be considered and only "borda" is currently available for aggregating them. If more than one is given for a "het" or "hom" experiment, the first algorithm will be used.

For **"hyb"** and **"hom"** experiments:
* ```"bootstraps"```: an integer representing the number of bootstraps to be sampled, e.g., ```50```

#### <a name="experiments-example">Example</a>
```python
from efsassembler import Experiments

experiments_list = [
    {   "type": "hyb",
        "thresholds": [1,5,10,50],
        "seed": 42,
        "folds": 10,
        "undersampling": False,
        "balanced_final_selection": False,
        "classifier": "gbm",
        "datasets": ["my/dataset/one.csv", "my/dataset/two.csv"],
        "rankers": [("reliefF", "python", "rf"), ("geoDE", "python", "gd"), ("gain-ratio", "r", "gr")],
        "aggregators": ["stb_weightened_layer1", "borda"],
        "bootstraps": 50
    },
    
    {   "type": "het",
        "thresholds": [1,5,10,50],
        "seed": 42,
        "folds": 10,
        "undersampling": True,
        "balanced_final_selection": True,
        "classifier": "svm",
        "datasets": ["my/dataset/one.csv", "my/dataset/two.csv"],
        "rankers": [("reliefF", "python", "rf"), ("wx", "python", "wx"), ("gain-ratio", "r", "gr")],
        "aggregators": ["borda"]
    },
    
    {   "type": "hom",
        "thresholds": [1,5,10,50],
        "seed": 42,
        "folds": 10,
        "undersampling": False,
        "balanced_final_selection": True,
        "classifier": "gbm",
        "datasets": ["my/dataset/one.csv"],
        "rankers": [("gain-ratio", "r", "gr")],
        "aggregators": ["borda"],
        "bootstraps": 50
    },
    
    {   "type": "sin",
        "thresholds": [1,5,10,50],
        "seed": 42,
        "folds": 10,
        "undersampling": True,
        "balanced_final_selection": False,
        "classifier": "gbm",
        "datasets": ["my/dataset/one.csv"],
        "rankers": [("reliefF", "python", "rf")]
    }
]

exp = Experiments(experiments_list, "my/results/path/")
exp.run()
```

The above example will perform a hyb experiment and a het experiment on datasets *one.csv* and *two.csv* (4 experiments in total); a hom experiment and a sin experiment on dataset *one.csv* (2 experiments in total); thus, 6 experiments will be executed in total with ```exp.run()``` call. 

## Datasets expected format

The datasets are expected to:
* be a .csv or .rds file
* represent each sample on each row
* represent each feature on each column
* have the first column as the index name for each sample
* have the last column named exactly **class**
* each value on the class column must be either **1** for positive observations or **0** for negative/control observations
* have only numeric features without NaN/missing values

For example:
|   | feature1  | feature2  | feature3  | feature4  | feature5  | class  |
|---|---|---|---|---|---|---|
| sample1  | 2.357  | 10.124  | -1.733  | 5  | 1.553  | 0  |
| sample2  | 2.823  | 11.3274  | 0.001  | 2  | 1.287  | 1  |
| sample3  | 1.7343  | 11.8922  | -0.736  | 2  | 1.5981  | 1  |
| sample4  | 2.568  | 9.476  | -2.0012  | 6  | 1.9988  | 0  |
| sample5  | 1.871  | 11.046  | -0.8375  | 1  | 1.3094  | 1  |


## Results folder structure

Example for a het experiment using borda aggregation and a 5-fold stratified cross-validation:
```
.                                   # results/experiment root folder 
├── accuracies_results.csv          # accuracies for each threshold in each fold
├── info.txt                        # information about the experiment or simple feature extraction
├── final_confusion_matrices.pkl    # confusion matrices for each threshold in each fold
├── final_results.csv               # stabilites and classification metrics mean/std for each threshold
├── fold_sampling.pkl               # indexes used for each fold iteration
├── pr_aucs_results.csv             # pr_aucs for each threshold in each fold
├── roc_aucs_results.csv            # roc_aucs for each threshold in each fold
├── fold_1                          # files related to the first fold iteration of the stratified cv
        ├── gr.csv                  # gain-ratio feature importance ranking
        ├── rf.csv                  # relieff feature importance ranking
        ├── wx.csv                  # wx feature importance ranking
        └── relevance_rank.csv      # final (after aggregate) feature importance ranking
├── fold_2
        ├── gr.csv
        ├── rf.csv
        ├── wx.csv
        └── relevance_rank.csv
├── fold_3
        ├── gr.csv
        ├── rf.csv
        ├── wx.csv
        └── relevance_rank.csv
├── fold_4
        ├── gr.csv
        ├── rf.csv
        ├── wx.csv
        └── relevance_rank.csv
├── fold_5
        ├── gr.csv
        ├── rf.csv
        ├── wx.csv
        └── relevance_rank.csv
└── selection                       # fold related to the final selection (using all samples in the dataset)
        ├── folds.pkl               # generated folds for balancing minority and majority class (check our article)
        ├── relevance_rank.csv      # final aggregated ranking, the ranking that should be used as the final output for this method
        ├── 0
            └── relevance_rank.csv  # final aggregated rank for this portion of data samples
        ├── 1
            └── relevance_rank.csv 
        └── 2
            └── relevance_rank.csv 
```

<a name="balanced_folds">If</a> ```"balanced_final_selection"``` is set to *True* (or not provided), the final selection process will split the whole dataset into equally (except, maybe, for the last fold) stratified folds using all the samples of the minority class and a correspondent amount of the majority class. It will generate folds (each with the same examples for the minority class) until there's no majority class examples left. After the feature selection method concluding the ranking for each fold, they are aggregated in one final ranking, the ```relevance_rank.csv``` file inside the ```selection/``` folder. This should be used as the final true feature importance ranking generated by the selected method for the provided data. If the same argument is set to *False*, the feature selection technique will be applied in the whole dataset directly.

The numbers inside the ranking .csv files should be ignored as they are only residuals left after the aggreagtion process conclude (it used them as a reference for sorting the features). The features are ordered by descending of relevance, which means that the first feature of the ranking file is the most important and the last feature in the ranking is the least important.



## Usage for running feature extraction

If the user only wants to directly extract the features without the whole experiment procedures, the ```FeatureExtraction``` class can be used and it works essentially like the Experiments class.

```python
from efsassembler import FeatureExtraction
fe = FeatureExtraction(extraction_configs, "my/results/path/")
fe.run()
```

The expected type of input for the ```extraction_configs``` object is a list of dictionaries, where each dictionary represents a feature extraction (FE) configuration. If multiple datasets are provided, multiple FEs using that configuration will be executed, each FE using each of the provided datasets. The expected keys and values for each dictionary are:
```"type"```, ```"thresholds"```, ```"seed"```,```"datasets"```, ```"rankers"```, ```"aggregators"``` (if applied), ```"bootstraps"``` (if applied) and ```"balanced_selection"``` (True if the FE process is to be applied in [balanced dataset folds](#balanced_folds)). The values and meanings of these keys are the same as explained in the [Usage for running experiments](#usage-for-running-experiments) section.


