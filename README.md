# Meta-Learning for encoder selection

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [Project Organization](#project-organization)
- [License](#license)

## Project Description
Welcome to our project repository! We are a group of dedicated students from the Karlsruhe Institute of Technology, utilizing our data science skills to tackle real-world, high-impact problems as part of our Data Science Lab course at KIT. This project is focused on the exploration of meta-learning for the task of encoder selection.

In this project we aim to predict a ranking based on features such as "dataset", "model", "tuning", "scoring", and "encoder". We approach this problem from four different angles:

1. Regression: This approach predicts a real number based on the features from above.
2. Pointwise (multi-class): This approach predicts a natural number or “missing” rank based on the same features.
3. Pairwise (multi-target binary): This approach predicts if one encoder performs better than another given the data. This is done for every pair of encoders and results in a Boolean outcome.
4. Listwise (multi-target multi-class): This final approach predicts the entire ranking at once resulting in an array of natural numbers.

The repository you are about to explore contains all the code, resources, and documentation produced and used during this project. Our objective is to evaluate the performance of each approach and to develop a comprehensive understanding of their strengths and weaknesses.

## Installation

###  (optional) Create virtual environment
You can optionally choose to run the project in a virtual environment. We would strongly recommend to do this if you have multiple python projects that you would like to run on your device. This short guide will show you how to get a virtual python enviroment running with pythons `venv` module.

First we need to create the virtual environment

#### Windows
```
python -m venv venv
```

#### Linux
```
python3 -m venv venv
```

Now a new folder has been created in your current terminals path that includes the necessary files for the virtual environment. The folder should be called venv. (Make sure it is named this way to have the folder ignored by the preconfigured .gitignore file) The environment now needs to be activated. 

#### Windows
```
venv\Scripts\activate
```

#### Linux
```
source venv/bin/activate
```

Your terminal should now show a `(venv)` next to the path. If that is the case the virtual environment has been activated. If you encounter a terminal permission error take a look at [this post](https://stackoverflow.com/questions/56199111/visual-studio-code-cmd-error-cannot-be-loaded-because-running-scripts-is-disabl) or use CMD instead of powershell.

### Installing the Requirements
This project uses a "requirements.txt" file which includes all dependecies required to run this project. The project is built upon and tested on a windows system. Below we also include the necessary command to run the installation on a linux system which should run fine as well.


#### Windows

```
py -m pip install -r requirements.txt
```

#### Linux

```
pip install -r requirements.txt
```

## Usage
For running the pipeline of this project you first have to make sure that the training set you want to use is in the required `data/raw` folder. By default a dataset named `dataset_train.csv` is expected. If you want to use a different dataset, you have to use the `--train_dataset` parameter when running the pipeline. You may also use the `--test_dataset` parameter if you have a test set that you want to get predictions from. If the pipeline is started with this parameter, the pipeline will save the predictions into the `data/processed` folder. In case that the training dataset doesn't include the target column a seperate target file is required. With the `--y_train_dataset` parameter you may include this file in the pipeline. The pipeline will then use this file as the target column for the training set. \
It is also possible to run the pipeline as a listwise Neural Network. When running the `main.py` file set the `--as_neural_network` flag. This will start the execution with the Neural Network.

Another important parameters is the `--pipeline_type` parameter. You can find more information about the implementation of these pipelines in the `PipelineFactory` class found in the __src/pipeline/__ folder. This parameter defines the type of pipeline that will be executed. The following pipeline types are available:
* "regre_baseline"
* "class_baseline"
* "linear_regression"
* "regre_preprocessed"
* "regre_test"
* "regre_no_search"
* "regre_bayes_search"
* "pointwise_regression_no_search"
* "pointwise_normalized_regression_no_search"
* "pointwise_classification_no_search"
* "pointwise_ordinal_regression_no_search"
* "pointwise_regression_grid_search"
* "pointwise_normalized_regression_bayes_search"
* "pointwise_classification_bayes_search"
* "pointwise_ordinal_regression_bayes_search"
* "pairwise_classification_no_search"
* "pairwise_classification_optuna_search"
* "listwise_multidimensional_regression_no_search"
* "listwise_multidimensional_regression_bayes_search"
* "listwise_dimensionwise_regression_no_search"
* "listwise_dimensionwise_regression_bayes_search"


The `main.py` file is the entry point for the project. To run the pipeline with the default dataset and no test set use the following command:
```
py main.py
```
The project will start running and display output in your command line. If you want to use a dataset different from the default set and with a test set for getting a `prediction.csv` file use the following command:
```
py main.py --dataset [your_dataset_name] --test_dataset [your_test_dataset_name] --y_train_dataset [your_target_dataset_name]
```
**Hint:** replace `[your_dataset_name]` and don't forget to include the .csv ending  

**Important:** only use `--y_train_dataset` if your training dataset doesn't include the target column. 

Parameter overview for main.py
```
--pipeline_type: The type of pipeline to run. Possible values are can be found in the list above. Eg. "regre_baseline"
--train_dataset: The name of the training dataset. Default is dataset_train.csv
--test_dataset: The name of the test dataset. Only if this parameter is set a prediction.csv file will be created.
--y_train_dataset: The name of the target dataset. Only use this parameter if your training dataset doesn't include the target column.
--target: The name of the target column. Default is cv_score
--as_neural_network: If this flag is set the pipeline will run as a listwise neural network.
--epochs: The number of epochs for the neural network. Default is 200
```

## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download, generate or manipulate data
    │   │   
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │                     predictions
    │   │
    │   ├── pipeline       <- Scripts that create or modify the data pipeline
    │   │
    │   ├── tests          <- Scripts to test other python modules in the project
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented 


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## License

This project is licensed under the MIT License. See the [LICENSE.md](LICENSE.md) file for details.
