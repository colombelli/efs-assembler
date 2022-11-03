# efs-assembler

![](https://img.shields.io/badge/python-v3.6-blue)
![](https://img.shields.io/badge/R-v4.0-red)

## Summary

- [Introduction](#introduction)
    - [Feature selection algorithms](#feature-selection-algorithms)
    - [Aggregation algorithms](#aggregation-algorithms)
    - [Classification algorithms](#classification-algorithms)
    - [Collected metrics](#collected-metrics)
- [Installation](#installation)
- [Usage for running experiments](#usage-for-running-experiments)
    - [Example](#experiments-example)
- [Datasets expected format](#datasets-expected-format)
- [Results folder structure](#results-folder-structure)
- [Usage for running feature selection](#usage-for-running-feature-selection)
- [Adding new algorithms](#add_new_algs)
    - [Rules for new feature selection algorithms](#rules-for-new-feature-selection-algorithms) 
    - [Rules for new aggregation algorithms](#rules-for-new-aggregation-algorithms) 
    - [Rules for new classifier algorithms](#rules-for-new-classifier-algorithms) 
- [BibTeX entry](#bibtex-entry) 
- [Acknowledgement](#acknowledgement)

## Introduction

The efs-assembler is a Python package integrated with R for performing ensemble feature selection experiments in binary classification problems. It is high flexible and allows the user to add different algorithms of feature selection (including support for R language), aggregation and also classification, offering options to perform stratified cross validation with downsampling and collect various performance metrics.

The package is backed by a scientific study about ensemble feature selection for cancer biomarkers discovery in gene expression data. If you have scientific interests or want to use our package in formal reports, we kindly ask you to cite us in your publication: [Colombelli, F., Kowalski, T.W. and Recamonde-Mendoza, M., 2022. A hybrid ensemble feature selection design for candidate biomarkers discovery from transcriptome profiles. Knowledge-Based Systems, 254, p.109655.](#bibtex-entry)

This work was developed at the Institute of Informatics, Universidade Federal do Rio Grande do Sul and Bioinformatics Core, Hospital de Clínicas de Porto Alegre.

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

Because of the package variety required by the provided feature selection algorithms, the installation steps still need to be further tested and investigated to make sure we cover more use cases and computer setups. Consider installing the current supported versions of Python (3.6) and R (4.0).

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
* ```"classifier"```: either ```"random_forest"```, ```"gbc"```, or ```"svm"```, for indicating the classification algorithm to be used
* ```"datasets"```: a list with dataset paths that are going to be exposed to the experiment, e.g., ```["data/set/one.csv", "data/set/two.rds"]```. The accepted file types for the datasets are .csv and .rds. Additional information about the expected dataset format is given on [this section](#datasets-expected-format).
* ```"rankers"```: a list with the feature selection algorithms to be used (even in "sin" and "hom" experiments a list is expected). The feature selection algorithms are represented by a tuple in the format ("file_name", "language", "rank_file_name_to_use_for_saving_their_result"), e.g., ```[("reliefF", "python", "rf"), ("geoDE", "python", "gd"), ("gain-ratio", "r", "gr"), ("symmetrical-uncertainty", "r", "su"), ("wx", "python", "wx")]```. Those are the current available algorithms. If more than one is given for a "sin" or "hom" experiment, the first algorithm will be used.

For **"hyb"**, **"het"** and **"hom"** experiments:
* ```"aggregators"```: a list with the aggregator algorithms to be used. Since only python algorithms are supported, only the file name of the algorithm is required, e.g., ```["stb_weightened_layer1", "borda"]```. If it is a "het" or "hom" experiment only the first aggregator will be considered and only "borda" is currently available for aggregating them.

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
        "classifier": "gbc",
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
        "classifier": "gbc",
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
        "classifier": "gbc",
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

<a name="balanced_folds">If</a> ```"balanced_final_selection"``` is set to *True* (or not provided), the final selection process will split the whole dataset into equally (except, maybe, for the last fold) stratified folds using all the samples of the minority class and a correspondent amount of the majority class. It will generate folds (each with the same examples for the minority class) until there's no majority class examples left. After the feature selection method conclude the ranking for each fold, they are aggregated in one final ranking, the ```relevance_rank.csv``` file inside the ```selection/``` folder (or ```agg_rank_th<a threshold>.csv``` if FS method is threshold sensitive). This should be used as the final true feature importance ranking generated by the selected method for the provided data. If the same argument is set to *False*, the feature selection technique will be applied in the whole dataset directly.

The numbers inside the ranking .csv files should be ignored as they are only residuals left after the aggreagtion process conclude (it used them as a reference for sorting the features). The features are ordered by descending of relevance, which means that the first feature of the ranking file is the most important and the last feature in the ranking is the least important.



## Usage for running feature selection

If the user only wants to directly select the features without the whole experiment procedures (cross-validation, classification, stability measurements, etc), the ```FeatureSelection``` class can be used and it works essentially like the Experiments class.

```python
from efsassembler import FeatureSelection
fs = FeatureSelection(selection_configs, "my/results/path/")
fs.run()
```

The expected type of input for the ```selection_configs``` object is a list of dictionaries, where each dictionary represents a feature selection (FS) configuration. If multiple datasets are provided, multiple FSs using that configuration will be executed, each FS using each of the provided datasets. The expected keys and values for each dictionary are:
```"type"```, ```"thresholds"```, ```"seed"```,```"datasets"```, ```"rankers"```, ```"aggregators"``` (if applied), ```"bootstraps"``` (if applied) and ```"balanced_selection"``` (True if the FS process is to be applied in [balanced dataset folds](#balanced_folds)). The values and meanings of these keys are the same as explained in the [Usage for running experiments](#usage-for-running-experiments) section.



## <a name="add_new_algs">Adding new algorithms (feature selection, aggregation, classifier)</a>
 
 For the addition of new feature selection algorithms, new aggregation algorithms or new classifiers, the ScriptsManager class comes handy.
 The addition is as simple as initializing the a ScriptsManager object and call the appropriate add method.
 
 ```python
from efsassembler import ScriptsManager

sm  = ScriptsManager()

# For adding new feature selection algorithms:
sm.add_fs_algorithm("/path/to/my/selector.py")   # Or selector.r

# For adding new aggregation algorithms:
sm.add_aggregation_algorithm("/path/to/my/aggregator.py")

# For adding new classifier algorithms:
sm.add_classifier("/path/to/my/classifier.py")
```

If, for some reason, the user wants to remove any user added algorithms, remove methods should be called instead and the name of the script to remove should be given as the parameter for the remove method.

```python
from efsassembler import ScriptsManager

sm  = ScriptsManager()

# For removing user added feature selection algorithms:
sm.remove_fs_algorithm("selector.py")   # Or selector.r

# For removing user added aggregation algorithms:
sm.remove_aggregation_algorithm("aggregator.py")

# For removing user added classifiers:
sm.remove_fs_algorithm("classifier.py")
```


### <a name="rules-for-new-feature-selection-algorithms">Rules for new feature selection algorithms</a>

The first thing to noticed is that the new user-defined personalized feature selection algorithm to integrate the ensemble should be a ranker. It is supposed to rank all the given features according to its relevance to the binary classification problem from the most to the least relevant one. The <a name="ranking_format">output</a> of the algorithm should be a dataframe where the features are the index and there's only one column called exaclty `rank` where each value corresponds to the ranking of the feature (it could be any arbitrary value, but something like {1....n} where n is the total number of features, is more common).

The output, then, should look like:
|   | rank |
| ------------- | ------------- |
| feature_x  | 1  |
| feature_y  | 2  |
| feature_z  | 3  |
| feature_w  | 4  |

The script.py or script.r implementing the personalized feature selection algorithm should define a function called `select(arg)` from which it will receive the input and deliver the output. While the name of the parameter does not matter as long as one and only one argument is defined, the name of the function should be exaclty `select`.

The input received by the function is a fraction of the original dataset in a dataframe format. The `select(arg)` function should be able to understand this data structure and establish a criteria to deliver the desired output. The fraction of the original dataset only reduces some samples under the k-fold cross validation process, but its remaining samples have all the features, including the class column, looking exactly as detailed [here](#datasets-expected-format).


### <a name="rules-for-new-aggregation-algorithms">Rules for new aggregation algorithms</a>

The script implementing the new aggregation algorithm should have:
* A boolean variable called `heavy`
* A boolean variable called `threshold_sensitive`
* A function with only two arguments called `aggregate`, where the first argument is for `self` and the second, `selector`, is for the type of experiment (implemented by the `FSTechnique` superclass or any of its specialized classes, `Heterogeneous`, `Homogeneous`, `Hybrid`, `SingleFR`).

The `heavy` variable is only considered in the Hybrid ensemble experiments and aggregators requiring the heavy behavior are only used as first aggregation methods (see our paper for more information on that). If `heavy` is set to true, for each fold iteration, it forces the buildage of a dictionary containing all rankings from all feature ranker methods and keep it in **memory** until the next fold iteration. This is useful if the user wants, for example, to measure the stability of the first layer rankings of each feature ranker method. This dictionary is accessible by the attribute `selector.dm.bs_ranking` and it should be used to deliver the output: the aggregated rankings of the first aggregator method. Assuming _m_ bootstraps, the `selector.dm.bs_ranking` is a dictionary in which the keys represent the number of the bootstrap (from 0 to _m_-1) and the value is a list of rankings to aggregate. Each item in this list follows the data structure described [here](#ranking_format). The output in this case is a list of rankings (in the same format), one for each bootstrap.

Set `heavy` to false if the aggregator only needs the list of rankings as information to output a final aggregated ranking. In this case the algorithm should use the attribute `selector.rankings_to_aggregate` as input, which is a list where each element is a ranking following the same data structure as described [here](#ranking_format). The output should be a single ranking (same data structure), resulted from the aggregation of the received list of rankings. 

In the `efs-assembler/efsassembler/aggregators/` path there's an example of an aggregator using the heavy aggregation (stb_weightened_layer1.py) and another aggregator that does not require the heavy aggregation process (borda.py).

The `threshold_sensitive` variable is only used to indicate if the aggregation algorithm output changes depending on the selected threshold applied in the rankings. If it does, the algorithm can access this information through the attribute `selector.current_threshold`, which is an integer representing the cut-off point (in terms of index) to be applied in the rankings.


### <a name="rules-for-new-classifier-algorithms">Rules for new classifier algorithms</a>

The script containing the new personalized classifier algorithm should implement a class called `Classifier` with three methods anologous to scikit-learn predictor objects:
* `.fit(X, y)`: method for training the classifier
* `.predict_proba(X)`: method for predicting the probabilities of each class for each sample data
* `.predict(X)`: method for predicting the classes of each sample data

For detailed information on the functionality of those methods, see [Developing scikit-learn estimators](#https://scikit-learn.org/stable/developers/develop.html), or directly a [classifier example](#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).



## BibTeX entry

```
@article{colombelli2022hybrid,
title = {A hybrid ensemble feature selection design for candidate biomarkers discovery from transcriptome profiles},
journal = {Knowledge-Based Systems},
pages = {109655},
year = {2022},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2022.109655},
url = {https://www.sciencedirect.com/science/article/pii/S0950705122008383},
author = {Felipe Colombelli and Thayne Woycinck Kowalski and Mariana Recamonde-Mendoza},
keywords = {Feature selection, Ensemble learning, Biomarkers discovery, Microarray, Bioinformatics, High-dimensional data}
}
```

## Acknowledgement

This project was financed in part by the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior - Brasil (CAPES) - Finance Code 001, Conselho Nacional de Desenvolvimento Científico e Tecnológico (project CNPq/AWS 032/2019, process no. 440005/2020-5), and Fundação de Amparo à Pesquisa do Estado do Rio Grande do Sul (FAPERGS).
