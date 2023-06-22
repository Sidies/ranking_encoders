# Meta-Learning for encoder selection

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Project Description](#project-description)
- [Installation](#installation)
- [Project Organization](#project-organization)
- [Usage](#usage)
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
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## Usage
For running the pipeline of this project you first have to make sure that the dataset you want to use is in the required `data/raw` folder. By default a dataset named `dataset_train.csv` is expected. If you want to use a different dataset, you have to use the --dataset parameter when running the pipeline.

Open a terminal and navigate to the project directory and run the following command: 
```
py main.py
```
The project will start running and display output in your command line. If you want to use a dataset different from the default use the following command:
```
py main.py --dataset [your_dataset_name]
```
Hint: replace `[your_dataset_name]` and don't forget to include the .csv ending

## License

This project is licensed under the MIT License. See the [LICENSE.md](LICENSE.md) file for details.
