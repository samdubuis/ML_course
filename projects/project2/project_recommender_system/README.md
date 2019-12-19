# CS-433 Machine Learning Project 2 - Project Recommender

This project presents a machine learning model that aim to predict item recommendations to a database of users. 
Based on their ratings of several items on a scale of 1 to 5 (no additional information about the movies or users), 
we combine different prediction algorithms in order to guess such recommendations. 

## Content
In this repository you'll find a few folders.
* Datasets contains the necessary .csv files used for training and testing and is also the output folder where the prediction csv file will appear
* rapport which will contain the final report PDF that was written in Overleaf
* npy is the folder of output where we saved our necessary numpy arrays 

In the scripts folder, there are these python files :
* implementations.py : contains the required implementations by the project of linear/ridge/logistic regression, etc...
* def.py : external function we coded that would help us throughout the project, either for cross validation or display or anything
* costs.py : file containing the code for costs computation
* helpers.py and proj1_helpers.py : two files given at the beginning containing pre-made function useful for importing and displaying infos
* project1.ipynb : the Jupyter Notebook where most tests where made
* run.py : actual file that should be run and does the prediction

## Execution 
For this project we coded everything by using Python 3.

Get into the scripts/ folder and launch
```bash
python3 run.py
```

The input are in the data folder : train.csv and test.csv

The code execute itself. We do a ridge regression on the training dataset, and a cross validation, with parameters as such :
* K-Fold : 10
* Degree : 12 up to this number to train and find the best one
* Lambda : between 10^-15 and 10^-3 
* Seed : 5 for randomness

We have also previously seperated the data to train in 4 mini-sets following a specific variable, better explained in the report.

## Submission
The output pred.csv is in the data folder and was submitter on [here](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs-2019) where we achieved a score of 0.809


