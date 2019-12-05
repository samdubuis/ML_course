import datetime
time = datetime.datetime.now()
print("launched at : ", time)
import pandas as pd
import numpy as np

#!pip3 install surprise
from surprise import Reader
from surprise import Dataset
from surprise import SVD               # importer ici les algo qu'on testera
from surprise import model_selection
from surprise import dump

from sklearn.model_selection import train_test_split 

print("IMPORT DONE")


df2 = pd.read_csv("Datasets/sample_submission.csv")

df2[["user", "item"]] = df2.Id.str.split("_", expand=True)

df2.user = df2.user.str.replace("r", "")
df2.item = df2.item.str.replace("c", "")


array_SVD = np.ones((df2.shape[0],1))
array_KNN = np.ones((df2.shape[0],1))
array_NMF = np.ones((df2.shape[0],1))
array_SlopeOne = np.ones((df2.shape[0],1))
array_CoClustering = np.ones((df2.shape[0],1))

print("PANDAS DONE, ARRAYS ARE READY")
print("")
print("LOADING OF PICKLES FOR EACH ALGO")

_, algo_svd = dump.load("dump/dump_SVD")
_, algo_knn = dump.load("dump/dump_KNN_basic")
_, algo_nmf = dump.load("dump/dump_NMF")
_, algo_slopeone = dump.load("dump/dump_SlopeOne")
_, algo_coclustering = dump.load("dump/dump_CoClustering")

print("ESTIMATION OF VALUES FOR EACH ALGO")
for i in df2.iterrows():
    if i[0]%100000==0:
        print(i[0])
    array_SVD[i[0]] = algo_svd.estimate(int(i[1][2])-1, int(i[1][3])-1)
    array_KNN[i[0]]= algo_knn.estimate(int(i[1][2])-1, int(i[1][3])-1)[0]
    array_NMF[i[0]] = algo_nmf.estimate(int(i[1][2])-1, int(i[1][3])-1)
    array_SlopeOne[i[0]]=algo_slopeone.estimate(int(i[1][2])-1, int(i[1][3])-1)
    array_CoClustering[i[0]] = algo_coclustering.estimate(int(i[1][2])-1, int(i[1][3])-1)
    
print("DONE")
print("")
print("BLENDING EACH ALGO INTO AN ARRAY")

tmp = np.concatenate((array_SVD, array_KNN, array_NMF, array_SlopeOne, array_CoClustering), axis=1 )
final_array = np.mean(tmp, axis=1)
final_array = np.rint(final_array)
final_array[final_array>5]=5
final_array[final_array<1]=1
final_array

df2.Prediction = final_array
df2 = df2.drop(columns=["user", "item"])
df2.to_csv("Datasets/submission_pipeline.csv", index=False)

print("EVERYTHING DONE")