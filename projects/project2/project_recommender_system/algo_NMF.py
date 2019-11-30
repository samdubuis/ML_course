import datetime
time = datetime.datetime.now()
print("launched at : ", time)
import pandas as pd
import numpy as np

#!pip3 install surprise
from surprise import Reader
from surprise import Dataset
from surprise import NMF              # importer ici les algo qu'on testera
from surprise import model_selection
from surprise import dump

from sklearn.model_selection import train_test_split 

print("IMPORT DONE")

############
df = pd.read_csv("Datasets/data_train.csv")

df[["user", "item"]] = df.Id.str.split("_", expand=True)

df.user = df.user.str.replace("r", "")
df.item = df.item.str.replace("c", "")


reader = Reader(rating_scale=(1,5)) 
data = Dataset.load_from_df(df[["user","item","Prediction"]], reader)

del df #freeing memory
print("PANDAS DONE, DATA AND READER ARE READY")
print("")

n_factors = list(map(int, input('Enter values for n_factors separated by , without space: ').split(',')))
n_epochs = list(map(int, input('Enter values for n_epochs separated by , without space: ').split(',')))


f=open("results.txt", "a")
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

n_jobs = int(input("Value for how many processors to use : (-1 is all, -2 is all except one) "))
print("")

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

algo = gs.best_estimator["rmse"] #choix de l'algo selon l'erreur, touuuut inclus

print("FITTING OF DATA ON BEST ALGO")
algo.fit(data.build_full_trainset()) # ici on va train notre algo sur le dataset complet, sans cv car les paramètres sont optimaux

dump_name = "dump/dump_NMF"
dump.dump(dump_name, algo=algo, verbose=1)
print("EVERYTHING DONE")