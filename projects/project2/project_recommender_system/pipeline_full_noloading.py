import datetime
time = datetime.datetime.now()
print("launched at : ", time)
import pandas as pd
import numpy as np

#!pip3 install surprise
from surprise import Reader
from surprise import Dataset
from surprise import SVD, KNNBasic, NMF, SlopeOne, CoClustering              # importer ici les algo qu'on testera
from surprise import model_selection
from surprise import dump

from sklearn.model_selection import train_test_split 

print("IMPORT DONE")

############
df = pd.read_csv("Datasets/data_train.csv")

df[["user", "item"]] = df.Id.str.split("_", expand=True)

df.user = df.user.str.replace("r", "")
df.item = df.item.str.replace("c", "")

#########
df2 = pd.read_csv("Datasets/sample_submission.csv")

df2[["user", "item"]] = df2.Id.str.split("_", expand=True)

df2.user = df2.user.str.replace("r", "")
df2.item = df2.item.str.replace("c", "")


reader = Reader(rating_scale=(1,5)) 
data = Dataset.load_from_df(df[["user","item","Prediction"]], reader)


del df
print("PANDAS DONE, DATA AND READER ARE READY")
print("")

########

# ## Training de chaque algo
f=open("results.txt", "a")
print("TRAINING OF EACH ALGO")

n_epochs = list(map(int, input('Enter values for n_epochs FOR ALL ALGOS separated by , without space: ').split(',')))


print("PARAMETERS FOR SVD :")
n_factors = list(map(int, input('Enter values for n_factors separated by , without space: ').split(',')))
lr_all = list(map(float, input('Enter values for learning rate lr_all separated by , without space: ').split(',')))
reg_all = list(map(float, input('Enter values for regularization term rg_all separated by , without space: ').split(',')))
print("")

print("PARAMETERS FOR KNN BASIC :")
k = list(map(int, input('Enter values for k separated by , without space: ').split(',')))
min_k = list(map(int, input('Enter values for min_k separated by , without space: ').split(',')))
print("")

print("PARAMETERS FOR NMF :")
n_factors = list(map(int, input('Enter values for n_factors separated by , without space: ').split(',')))
print("")

print("PARAMETERS FOR COCLUSTERING :")
n_cltr_i = list(map(float, input('Enter values for n_cltr_i separated by , without space: ').split(',')))
n_cltr_u = list(map(float, input('Enter values for n_cltr_u separated by , without space: ').split(',')))
print("")

n_jobs = int(input("Value for how many processors to use : (-1 is all, -2 is all except one) "))
print("")
#-------------------------------------------------


print("SVD")

f.write("\n")
f.write("SVD {}\n".format(time))
f.write("n_factors : {}\n" .format(n_factors))
f.write("n_epochs : {}\n" .format(n_epochs))
f.write("learning rate lr_all : {}\n" .format(lr_all))
f.write("regularization term rg_all : {}\n" .format(reg_all))

param_grid = {
    'n_factors' : n_factors,
    'n_epochs': n_epochs,
    'lr_all': lr_all,
    'reg_all': reg_all
} 
cv=5
algorithm = SVD

gs = model_selection.GridSearchCV(algorithm, param_grid, measures=['rmse'], cv=cv, n_jobs=n_jobs, joblib_verbose=100)  #enlever mae car non utilisé dans le projet pour sauver du temps

tasks = 1
for i in param_grid:
    tasks *= len(param_grid.get(i))

tasks *= cv
print("Total number of tasks to compute : ",tasks)
print("")

print("BEGINNING OF FITTING GRIDSEARCH")
print("")

f.write("Full data \n")
gs.fit(data) # les calculs se font actuellement ici 

print("FITTING GRIDSEARCH DONE")
print("")
print(gs.best_params)
print(gs.best_score)

f.write("Best param : {}\n" .format(gs.best_params))
f.write("Best score : {}\n" .format(gs.best_score))
f.write("\n")

algo_svd = gs.best_estimator["rmse"] #choix de l'algo selon l'erreur, touuuut inclus

print("FITTING OF DATA ON BEST ALGO")
algo_svd.fit(data.build_full_trainset()) # ici on va train notre algo sur le dataset complet, sans cv car les paramètres sont optimaux

dump_name = "dump/dump_SVD"
dump.dump(dump_name, algo=algo_svd, verbose=1)

#-------------------------------------------------

# ### KNN
print("")
print("KNN Basic")

f.write("\n")
f.write("KNN basic {}\n".format(time))
f.write("k : {}\n" .format(k))
f.write("min_k : {}\n" .format(min_k))

param_grid = {
    'k' : k,
    'min_k': min_k
} 
cv=5
algorithm = KNNBasic

gs = model_selection.GridSearchCV(algorithm, param_grid, measures=['rmse'], cv=cv, n_jobs=n_jobs, joblib_verbose=100)  #enlever mae car non utilisé dans le projet pour sauver du temps

tasks = 1
for i in param_grid:
    tasks *= len(param_grid.get(i))

tasks *= cv
print("Total number of tasks to compute : ",tasks)
print("")

print("BEGINNING OF FITTING GRIDSEARCH")
print("")

f.write("Full data \n")
gs.fit(data) # les calculs se font actuellement ici 

print("FITTING GRIDSEARCH DONE")
print("")
print(gs.best_params)
print(gs.best_score)

f.write("Best param : {}\n" .format(gs.best_params))
f.write("Best score : {}\n" .format(gs.best_score))
f.write("\n")

algo_knn = gs.best_estimator["rmse"] #choix de l'algo selon l'erreur, touuuut inclus

print("FITTING OF DATA ON BEST ALGO")
algo_knn.fit(data.build_full_trainset()) # ici on va train notre algo sur le dataset complet, sans cv car les paramètres sont optimaux

dump_name = "dump/dump_KNN_basic"
dump.dump(dump_name, algo=algo_knn, verbose=1)


#-------------------------------------------------
# ### NFM
print("")
print("NMF")
f.write("\n")
f.write("NMF {}\n".format(time))
f.write("n_factors : {}\n" .format(n_factors))
f.write("n_epochs : {}\n" .format(n_epochs))

param_grid = {
    'n_factors' : n_factors,
    'n_epochs': n_epochs
} 
cv=5
algorithm = NMF


gs = model_selection.GridSearchCV(algorithm, param_grid, measures=['rmse'], cv=cv, n_jobs=n_jobs, joblib_verbose=100)  #enlever mae car non utilisé dans le projet pour sauver du temps

tasks = 1
for i in param_grid:
    tasks *= len(param_grid.get(i))

tasks *= cv
print("Total number of tasks to compute : ",tasks)
print("")

print("BEGINNING OF FITTING GRIDSEARCH")
print("")

f.write("Full data \n")
gs.fit(data) # les calculs se font actuellement ici 

print("FITTING GRIDSEARCH DONE")
print("")
print(gs.best_params)
print(gs.best_score)

f.write("Best param : {}\n" .format(gs.best_params))
f.write("Best score : {}\n" .format(gs.best_score))
f.write("\n")

algo_nmf = gs.best_estimator["rmse"] #choix de l'algo selon l'erreur, touuuut inclus

print("FITTING OF DATA ON BEST ALGO")
algo_nmf.fit(data.build_full_trainset()) # ici on va train notre algo sur le dataset complet, sans cv car les paramètres sont optimaux

dump_name = "dump/dump_NMF"
dump.dump(dump_name, algo=algo_nmf, verbose=1)

#-------------------------------------------------
# ### SlopeOne
print("")
print("SlopeOne")

algo_slopeone = SlopeOne()
algo_slopeone.fit(data.build_full_trainset())
dump.dump("dump/SlopeOne_fitted_dump", algo=algo_slopeone, verbose=1)

#-------------------------------------------------
# ### CoClustering

print("")
print("CoClustering")
f.write("\n")
f.write("CoClustering {}\n".format(time))
f.write("n_epochs : {}\n" .format(n_epochs))
f.write("n_cltr_i : {}\n" .format(n_cltr_i))
f.write("n_cltr_u : {}\n" .format(n_cltr_u))

param_grid = {
    'n_cltr_u' : n_cltr_u,
    'n_cltr_i' : n_cltr_i, 
    'n_epochs': n_epochs
} 
cv=5
algorithm = CoClustering


gs = model_selection.GridSearchCV(algorithm, param_grid, measures=['rmse'], cv=cv, n_jobs=n_jobs, joblib_verbose=100)  #enlever mae car non utilisé dans le projet pour sauver du temps

tasks = 1
for i in param_grid:
    tasks *= len(param_grid.get(i))

tasks *= cv
print("Total number of tasks to compute : ",tasks)
print("")

print("BEGINNING OF FITTING GRIDSEARCH")
print("")

f.write("Full data \n")
gs.fit(data) # les calculs se font actuellement ici 

print("FITTING GRIDSEARCH DONE")
print("")
print(gs.best_params)
print(gs.best_score)

f.write("Best param : {}\n" .format(gs.best_params))
f.write("Best score : {}\n" .format(gs.best_score))
f.write("\n")
f.close()

algo_coclustering = gs.best_estimator["rmse"] #choix de l'algo selon l'erreur, touuuut inclus

print("FITTING OF DATA ON BEST ALGO")
algo_coclustering.fit(data.build_full_trainset()) # ici on va train notre algo sur le dataset complet, sans cv car les paramètres sont optimaux

dump_name = "dump/dump_CoClustering"
dump.dump(dump_name, algo=algo_coclustering, verbose=1)

#-------------------------------------------------
del gs


array_SVD = np.ones((df2.shape[0],1))
array_KNN = np.ones((df2.shape[0],1))
array_NMF = np.ones((df2.shape[0],1))
array_SlopeOne = np.ones((df2.shape[0],1))
array_CoClustering = np.ones((df2.shape[0],1))


# ## Estimations

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
