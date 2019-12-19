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
We split the pandas df into two sets of 70-30% randomly, that will be used to train once every algo, and the smaller one the ridge regression
of the second layer
'''
df_7, df_3 = train_test_split(df, train_size=0.7, random_state=1) 

tmp7 = Dataset.load_from_df(df_7[["user","item","Prediction"]], reader)
tmp3 = Dataset.load_from_df(df_3[["user","item","Prediction"]], reader)
data_train_7 = tmp7.build_full_trainset()
del df
print("DATA AND READER ARE READY")
print("")


# #############################################################################
# ################# TRAINING OF EACH ALGO FOR THE FIRST LAYER #################
# #############################################################################

# print("TRAINING OF EACH ALGO")
# ################# PARAMETERS #################

# n_jobs = int(input("Value for how many processors to use : (-1 is all, -2 is all except one) "))
# cv=5 # number of fold for the CV of the gridsearch of the parameters
# f=open("results.txt", "a")

# '''
# Below are the training for all algorithms tested for this project
# Except SlopeOne which doesn't have any parameters and as such doesn't need a gridsearch done to it
# All algos are based and computed on the same principles 
# - gridsearch with CV of n folds, n described just previously
# - parameters are in a dict
# - parameters, algo, timing, and results are written in a results.txt file so that it can be looked up later on

# When the gridsearch is done, the algo is fitted with the best parameters, those that give the smallest rmse on the 70% data of the original dataset
# finally it is dumped to a local file so it can be reused later on and it frees memory
# '''

# ################# Baseline Only #################
# print("")
# print("BASELINE ONLY")

# baseline_only_param_grid = {
#     'bsl_options' : {
#         'method' : ['als', 'sgd']
#     }
# }
# print("BASELINE ONLY PARAMETERS : ", baseline_only_param_grid)
# f.write("\n")
# f.write("Baseline Only {}\n".format(time))
# f.write(str(baseline_only_param_grid))

# algorithm = BaselineOnly
# gs = model_selection.GridSearchCV(algorithm, baseline_only_param_grid, measures=['rmse'], cv=cv, n_jobs=n_jobs, joblib_verbose=100)  #enlever mae car non utilisé dans le projet pour sauver du temps

# print("BEGINNING OF FITTING GRIDSEARCH")
# print("")
# gs.fit(tmp7)
# print("FITTING GRIDSEARCH DONE")
# print("")
# print(gs.best_params)
# print(gs.best_score)
# f.write("Best param : {}\n" .format(gs.best_params))
# f.write("Best score : {}\n" .format(gs.best_score))
# f.write("\n")
# f.close()
# algo_baseline_only = gs.best_estimator['rmse']
# print("FITTING OF DATA ON BEST ALGO")
# algo_baseline_only.fit(data_train_7) # ici on va train notre algo sur le dataset complet, sans cv car les paramètres sont optimaux

# dump_name = "dump/dump_BaselineOnly"
# dump.dump(dump_name, algo=algo_baseline_only, verbose=1)
# del algo_baseline_only, gs


# ################# KNN Basic #################

# print("")
# print("KNN Basic")
# f=open("results.txt", "a")

# knn_basic_param_grid = {
#     'k' : [150],
#     'min_k': [3],
#     'sim_options' : {
#         'user_based' : [True]
#     }
# }  
# print("KNN BASIC PARAMETERS : ", knn_basic_param_grid)
# f.write("\n")
# f.write("KNN basic {}\n".format(time))
# f.write(str(knn_basic_param_grid))

# algorithm = KNNBasic
# gs = model_selection.GridSearchCV(algorithm, knn_basic_param_grid, measures=['rmse'], n_jobs=2, cv=cv, joblib_verbose=100)  #enlever mae car non utilisé dans le projet pour sauver du temps

# print("BEGINNING OF FITTING GRIDSEARCH")
# print("")
# gs.fit(tmp7)
# print("FITTING GRIDSEARCH DONE")
# print("")
# print(gs.best_params)
# print(gs.best_score)
# f.write("Best param : {}\n" .format(gs.best_params))
# f.write("Best score : {}\n" .format(gs.best_score))
# f.write("\n")
# f.close()
# algo_knn_basic = gs.best_estimator['rmse']
# print("FITTING OF DATA ON BEST ALGO")
# algo_knn_basic.fit(data_train_7) # ici on va train notre algo sur le dataset complet, sans cv car les paramètres sont optimaux

# dump_name = "dump/dump_KNN_basic"
# dump.dump(dump_name, algo=algo_knn_basic, verbose=1)
# del algo_knn_basic, gs


# ################# KNN Means #################

# print("")
# print("KNN Means")
# f=open("results.txt", "a")

# knn_means_param_grid = {
#     'k' : [70],
#     'min_k': [5],
#     'sim_options' : {
#         'user_based' : [False]
#     }
# } 
# print("KNN MEANS PARAMETERS : ", knn_means_param_grid)
# f.write("\n")
# f.write("KNN means {}\n".format(time))
# f.write(str(knn_means_param_grid))

# algorithm = KNNWithMeans
# gs = model_selection.GridSearchCV(algorithm, knn_means_param_grid, measures=['rmse'], n_jobs=2, cv=cv, joblib_verbose=100)  #enlever mae car non utilisé dans le projet pour sauver du temps

# print("BEGINNING OF FITTING GRIDSEARCH")
# print("")
# gs.fit(tmp7)
# print("FITTING GRIDSEARCH DONE")
# print("")
# print(gs.best_params)
# print(gs.best_score)
# f.write("Best param : {}\n" .format(gs.best_params))
# f.write("Best score : {}\n" .format(gs.best_score))
# f.write("\n")
# f.close()
# algo_knn_means = gs.best_estimator['rmse']
# print("FITTING OF DATA ON BEST ALGO")
# algo_knn_means.fit(data_train_7) # ici on va train notre algo sur le dataset complet, sans cv car les paramètres sont optimaux

# dump_name = "dump/dump_KNN_means"
# dump.dump(dump_name, algo=algo_knn_means, verbose=1)
# del algo_knn_means, gs


# ################# KNN Baseline #################

# print("")
# print("KNN Baseline")
# f=open("results.txt", "a")
# knn_baseline_param_grid = {
#     'k' : [40],
#     'min_k': [5],
#     'sim_options' : {
#         'user_based' : [False]
#     },
#     'bsl_options' : {
#         'method' : ['als']
#     }
# } 
# print("KNN BASELINE PARAMETERS : ", knn_baseline_param_grid)
# f.write("\n")
# f.write("KNN Baseline {}\n".format(time))
# f.write(str(knn_baseline_param_grid))

# algorithm = KNNBaseline
# gs = model_selection.GridSearchCV(algorithm, knn_baseline_param_grid, measures=['rmse'], n_jobs=2, cv=cv, joblib_verbose=100)  #enlever mae car non utilisé dans le projet pour sauver du temps

# print("BEGINNING OF FITTING GRIDSEARCH")
# print("")
# gs.fit(tmp7)
# print("FITTING GRIDSEARCH DONE")
# print("")
# print(gs.best_params)
# print(gs.best_score)
# f.write("Best param : {}\n" .format(gs.best_params))
# f.write("Best score : {}\n" .format(gs.best_score))
# f.write("\n")
# f.close()
# algo_knn_baseline = gs.best_estimator['rmse']
# print("FITTING OF DATA ON BEST ALGO")
# algo_knn_baseline.fit(data_train_7) # ici on va train notre algo sur le dataset complet, sans cv car les paramètres sont optimaux

# dump_name = "dump/dump_KNN_baseline"
# dump.dump(dump_name, algo=algo_knn_baseline, verbose=1)
# del algo_knn_baseline, gs

# ################# SVD #################

# print("")
# print("SVD")
# f=open("results.txt", "a")
# svd_param_grid = {
#     'n_factors' : [400],
#     'n_epochs': [300],
#     'lr_all': [0.004],
#     'reg_all': [0.1]
# }  
# print("SVD PARAMETERS : ", svd_param_grid)
# f.write("\n")
# f.write("SVD {}\n".format(time))
# f.write(str(svd_param_grid))

# algorithm = SVD
# gs = model_selection.GridSearchCV(algorithm, svd_param_grid, measures=['rmse'], cv=cv, n_jobs=n_jobs, joblib_verbose=100)  #enlever mae car non utilisé dans le projet pour sauver du temps

# print("BEGINNING OF FITTING GRIDSEARCH")
# print("")
# gs.fit(tmp7)
# print("FITTING GRIDSEARCH DONE")
# print("")
# print(gs.best_params)
# print(gs.best_score)
# f.write("Best param : {}\n" .format(gs.best_params))
# f.write("Best score : {}\n" .format(gs.best_score))
# f.write("\n")
# f.close()
# algo_svd = gs.best_estimator['rmse']
# print("FITTING OF DATA ON BEST ALGO")
# algo_svd.fit(data_train_7) # ici on va train notre algo sur le dataset complet, sans cv car les paramètres sont optimaux

# dump_name = "dump/dump_SVD"
# dump.dump(dump_name, algo=algo_svd, verbose=1)
# del algo_svd, gs

# ################# NMF #################

# print("")
# print("NMF")
# f=open("results.txt", "a")
# nmf_param_grid = {
#     'n_factors' : [300],
#     'n_epochs': [200]
# }  
# print("NMF PARAMETERS : ", nmf_param_grid)
# f.write("\n")
# f.write("NMF {}\n".format(time))
# f.write(str(nmf_param_grid))

# algorithm = NMF
# gs = model_selection.GridSearchCV(algorithm, nmf_param_grid, measures=['rmse'], cv=cv, n_jobs=n_jobs, joblib_verbose=100)  #enlever mae car non utilisé dans le projet pour sauver du temps

# print("BEGINNING OF FITTING GRIDSEARCH")
# print("")
# gs.fit(tmp7)
# print("FITTING GRIDSEARCH DONE")
# print("")
# print(gs.best_params)
# print(gs.best_score)
# f.write("Best param : {}\n" .format(gs.best_params))
# f.write("Best score : {}\n" .format(gs.best_score))
# f.write("\n")
# f.close()
# algo_nmf = gs.best_estimator['rmse']
# print("FITTING OF DATA ON BEST ALGO")
# algo_nmf.fit(data_train_7) # ici on va train notre algo sur le dataset complet, sans cv car les paramètres sont optimaux

# dump_name = "dump/dump_NMF"
# dump.dump(dump_name, algo=algo_nmf, verbose=1)
# del algo_nmf, gs

# ################# SlopeOne #################

# print("")
# print("SLOPEONE")

# algo_slopeone = SlopeOne()
# algo_slopeone.fit(data_train_7)
# dump.dump("dump/dump_slopeone", algo=algo_slopeone, verbose=1)
# del algo_slopeone

# ################# CoClustering #################


# print("")
# print("CoClustering")
# f=open("results.txt", "a")
# coclustering_param_grid = {
#     'n_cltr_u' : [1],
#     'n_cltr_i': [1],
#     'n_epochs' : [200]
# }  
# print("COCLUSTERING PARAMETERS : ", coclustering_param_grid)
# f.write("\n")
# f.write("COCLUSTERING {}\n".format(time))
# f.write(str(coclustering_param_grid))

# algorithm = CoClustering
# gs = model_selection.GridSearchCV(algorithm, coclustering_param_grid, measures=['rmse'], cv=cv, n_jobs=n_jobs, joblib_verbose=100)  #enlever mae car non utilisé dans le projet pour sauver du temps

# print("BEGINNING OF FITTING GRIDSEARCH")
# print("")
# gs.fit(tmp7)
# print("FITTING GRIDSEARCH DONE")
# print("")
# print(gs.best_params)
# print(gs.best_score)
# f.write("Best param : {}\n" .format(gs.best_params))
# f.write("Best score : {}\n" .format(gs.best_score))
# f.write("\n")
# f.close()
# algo_coclustering = gs.best_estimator['rmse']
# print("FITTING OF DATA ON BEST ALGO")
# algo_coclustering.fit(data_train_7) # ici on va train notre algo sur le dataset complet, sans cv car les paramètres sont optimaux

# dump_name = "dump/dump_CoClustering"
# dump.dump(dump_name, algo=algo_coclustering, verbose=1)
# del algo_coclustering, gs


# #-------------------------------------------------#-------------------------------------------------#-------------------------------------------------# 

# print("FITTING OF ALL ALGOS IS DONE")


################# DATA PROCESSING FOR TESTING THE ALGOS #################


df2 = pd.read_csv("Datasets/sample_submission.csv")

df2[["user", "item"]] = df2.Id.str.split("_", expand=True)

df2.user = df2.user.str.replace("r", "")
df2.item = df2.item.str.replace("c", "")

reader = Reader(rating_scale=(1,5)) 
data_test = Dataset.load_from_df(df2[["user","item","Prediction"]], reader)
test = data_test.construct_testset(data_test.raw_ratings)


################# LOADING + TESTING OF EACH ALGO #################

print("TESTING OF THE ALGOS")
_, algo_baseline_only = dump.load("dump/dump_BaselineOnly")
_, algo_knn_basic = dump.load("dump/dump_KNN_basic")
_, algo_knn_means  = dump.load("dump/dump_KNN_means")
_, algo_knn_baseline = dump.load("dump/dump_KNN_baseline")
_, algo_svd = dump.load("dump/dump_SVD")
_, algo_nmf = dump.load("dump/dump_NMF")
_, algo_slopeone = dump.load("dump/dump_slopeone")
_, algo_coclustering = dump.load("dump/dump_CoClustering")

'''
Below we test each algo on the test set we just created previously
As it returns us a "Prediction" we need to map these to a list where we only output what interest us here, which is the 
estimated value
TODO could be a bit more modularized 
'''
array_baseline_only = algo_baseline_only.test(test)
array_baseline_only=list(map(lambda x:x[3], array_baseline_only))
print("BASELINE TESTED", datetime.datetime.now())

array_knn_basic = algo_knn_basic.test(test)
array_knn_basic=list(map(lambda x:x[3], array_knn_basic))
print("KNN BASIC TESTED", datetime.datetime.now())

array_knn_means  = algo_knn_means.test(test)
array_knn_means=list(map(lambda x:x[3], array_knn_means))
print("KNN MEANS TESTED", datetime.datetime.now())

array_knn_baseline = algo_knn_baseline.test(test)
array_knn_baseline=list(map(lambda x:x[3], array_knn_baseline))
print("KNN BASELINE TESTED", datetime.datetime.now())

array_svd = algo_svd.test(test)
array_svd=list(map(lambda x:x[3], array_svd))
print("SVD TESTED", datetime.datetime.now())

array_nmf = algo_nmf.test(test)
array_nmf=list(map(lambda x:x[3], array_nmf))
print("NMF TESTED", datetime.datetime.now())

array_slopeone = algo_slopeone.test(test)
array_slopeone=list(map(lambda x:x[3], array_slopeone))
print("SLOPEONE TESTED", datetime.datetime.now())

array_coclustering = algo_coclustering.test(test)
array_coclustering=list(map(lambda x:x[3], array_coclustering))
print("COCLUSTERING TESTED", datetime.datetime.now())

print("DONE")
print("")

################# BLENDING ALL ALGOS INTO A MATRIX #################

print("BLENDING EACH ALGO INTO AN ARRAY")
pred_array=np.vstack(
    [array_baseline_only, 
    array_knn_basic, 
    array_knn_means,
    array_knn_baseline, 
    array_svd, 
    array_nmf,
    array_slopeone,
    array_coclustering]
    )
# pred_array[np.where(pred_array>5)]=5
# pred_array[np.where(pred_array<1)]=1

pred_array = pred_array.T
np.save("npy/pred_array", pred_array)


##################################################################################################
################# SECOND LAYER : RIDGE REGRESSION TO DEFINE WEIGHTS OF EACH ALGO #################
##################################################################################################

data_train_3 = tmp3.construct_testset(tmp3.raw_ratings) # second test set based on the 30% of the original data

'''
Below we retest each algo on the second test set based on the other 30 % of the original data that weren't used yet
As it returns us a "Prediction" we need to map these to a list where we only output what interest us here, which is the 
estimated value
TODO could be a bit more modularized 
'''
array_baseline_only = algo_baseline_only.test(data_train_3)
array_baseline_only=list(map(lambda x:x[3], array_baseline_only))
print("BASELINE TESTED", datetime.datetime.now())

array_knn_basic = algo_knn_basic.test(data_train_3)
array_knn_basic=list(map(lambda x:x[3], array_knn_basic))
print("KNN BASIC TESTED", datetime.datetime.now())

array_knn_means  = algo_knn_means.test(data_train_3)
array_knn_means=list(map(lambda x:x[3], array_knn_means))
print("KNN MEANS TESTED", datetime.datetime.now())

array_knn_baseline = algo_knn_baseline.test(data_train_3)
array_knn_baseline=list(map(lambda x:x[3], array_knn_baseline))
print("KNN BASELINE TESTED", datetime.datetime.now())

array_svd = algo_svd.test(data_train_3)
array_svd=list(map(lambda x:x[3], array_svd))
print("SVD TESTED", datetime.datetime.now())

array_nmf = algo_nmf.test(data_train_3)
array_nmf=list(map(lambda x:x[3], array_nmf))
print("NMF TESTED", datetime.datetime.now())

array_slopeone = algo_slopeone.test(data_train_3)
array_slopeone=list(map(lambda x:x[3], array_slopeone))
print("SLOPEONE TESTED", datetime.datetime.now())

array_coclustering = algo_coclustering.test(data_train_3)
array_coclustering=list(map(lambda x:x[3], array_coclustering))
print("COCLUSTERING TESTED", datetime.datetime.now())

################# BLENDING ALL ARRAYS INTO A MATRIX #################

X=np.vstack(
    [array_baseline_only, 
    array_knn_basic, 
    array_knn_means,
    array_knn_baseline, 
    array_svd, 
    array_nmf,
    array_slopeone,
    array_coclustering]
    )
# X[np.where(X>5)]=5
# X[np.where(X<1)]=1

X = X.T

np.save("npy/X", X)

################# FINAL RIDGE REGRESSION PART #################
print("FINAL RIDGE REGRESSION BEGIN")

##### LOADING
X = np.load("npy/X.npy")
pred_array = np.load("npy/pred_array.npy")

##### EXPANSION + STANDARDIZATION
X=standardize(expansion(X,4))[0]
pred_array=standardize(expansion(pred_array,4))[0]

##### ACTUAL RIDGE REGRESSION
y=df_3.Prediction.values # values to compare to based on yet still the 30% of original data
clf=RidgeCV(alphas=np.linspace(10**-8,1,100),cv=10)
clf=clf.fit(X,y)

print(clf.coef_)
print(clf.alpha_)

##### PREDICTION
pred=clf.predict(pred_array)

################# OUTPUT CREATION AND ROUNDING #################

final_array=np.rint(pred)
final_array[np.where(final_array>5)]=5
final_array[np.where(final_array<1)]=1

df2.Prediction = final_array
df2 = df2.drop(columns=["user", "item"])
df2.to_csv("Datasets/submission_run_script.csv", index=False)

print("EVERYTHING DONE")