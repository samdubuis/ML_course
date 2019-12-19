# CS-433 Machine Learning Project 2 - Project Recommender

This project presents a machine learning model that aim to predict item recommendations to a database of users. 
Based on their ratings of several items on a scale of 1 to 5 (no additional information about the movies or users), 
we combine different prediction algorithms in order to guess such recommendations. 

## Content
In this repository you'll find a few folders.
* Datasets contains the necessary .csv files used for training and testing and is also the output folder where the prediction csv file will appear
* rapport which will contain the final report PDF that was written in Overleaf
* npy is the folder of output where we saved our necessary numpy arrays 
* notebooks contains all jupyter notebooks we used to work on the project before moving to the run.py script

Finally there are two files displayed at the root of the project and which are the two necessary files to run the code.
* run.py is the main execution file, well commented which you can read to understand most of what was done
* helpers.py simple python file which is imported by the run.py script and contains helper functions

## Execution 
For this project we coded everything by using Python 3.

Get into the scripts/ folder and launch
```bash
python3 run.py
```

The input are in the Datasets folder : data_train.csv and sample_submission.csv
The idea is to get in the code and add more parameters to each algo, so that when it gridsearches it can maybe attain better rmse than we did by letting it run a lot longer and on more efficiement machines.

We develop our machine learning model in two distinct layers. In the first place, we train a series of prediction algorithms and optimize the parameters of each one in order to minimise the RMSE of the predicted ratings over the test set. This part uses the library Surprise which is a a Python scikit building and analyzing recommender systems. In the second place, we combine all the algorithms together with a ridge regression model and optimize weighting parameters to obtain the best linear combination. This part uses the library Scikit-learn, which is a free and open source Python library for machine learning.

## Submission
The output submission_run_script.csv is in the Datasets folder and was submitted on [here](https://www.aicrowd.com/challenges/epfl-ml-recommender-system-2019) where we achieved a score of 1.026


