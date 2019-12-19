################# IMPORTS #################

import datetime
time = datetime.datetime.now()
print("launched at : ", time)
import pandas as pd
import numpy as np

#!pip3 install surprise    # this command is used when the script is launched in a Google Colab notebook and Surprise needs to be installed there
from surprise import *
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split

from helpers import *

print("IMPORT DONE")
print("")

################# DATAPROCESSING FOR THE FIRST LAYER #################

df = pd.read_csv("Datasets/data_train.csv")

df[["user", "item"]] = df.Id.str.split("_", expand=True)

df.user = df.user.str.replace("r", "")
df.item = df.item.str.replace("c", "")

reader = Reader(rating_scale=(1,5)) 

'''
We split the pandas df into two sets of 70-30% randomly, the first one will be used to train once every algo, 
and the smaller one for the ridge regression of the second layer
'''
df_7, df_3 = train_test_split(df, train_size=0.7, random_state=1) 

tmp7 = Dataset.load_from_df(df_7[["user","item","Prediction"]], reader)
tmp3 = Dataset.load_from_df(df_3[["user","item","Prediction"]], reader)
data_train_7 = tmp7.build_full_trainset()
del df
print("DATA AND READER ARE READY")
print("")

df2 = pd.read_csv("Datasets/sample_submission.csv")

df2[["user", "item"]] = df2.Id.str.split("_", expand=True)

df2.user = df2.user.str.replace("r", "")
df2.item = df2.item.str.replace("c", "")

reader = Reader(rating_scale=(1,5)) 
data_test = Dataset.load_from_df(df2[["user","item","Prediction"]], reader)
test = data_test.construct_testset(data_test.raw_ratings)

################# FINAL RIDGE REGRESSION PART #################
print("FINAL RIDGE REGRESSION BEGIN")

##### LOADING
X = np.load("npy/X.npy")
pred_array = np.load("npy/pred_array.npy")
print("loaded")
##### EXPANSION + STANDARDIZATION
X=standardize(expansion(X,4))[0]
pred_array=standardize(expansion(pred_array,4))[0]
print("expansion + standardization")
##### ACTUAL RIDGE REGRESSION
y=df_3.Prediction.values # values to compare to based on yet still the 30% of original data
clf=RidgeCV(alphas=np.linspace(10**-15,1,500),cv=30)
clf=clf.fit(X,y)

print(clf.coef_)
print(clf.alpha_)

##### PREDICTION
pred=clf.predict(pred_array)
print("prediction done")
################# OUTPUT CREATION AND ROUNDING #################

final_array=np.rint(pred)
final_array[np.where(final_array>5)]=5
final_array[np.where(final_array<1)]=1

df2.Prediction = final_array
df2 = df2.drop(columns=["user", "item"])
df2.to_csv("Datasets/submission_run_script.csv", index=False)

print("EVERYTHING DONE")
