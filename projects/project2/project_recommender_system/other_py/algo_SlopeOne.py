import datetime
time = datetime.datetime.now()
print("launched at : ", time)
import pandas as pd
import numpy as np

#!pip3 install surprise
from surprise import Reader
from surprise import Dataset
from surprise import SlopeOne               # importer ici les algo qu'on testera
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


algo = SlopeOne()

print("FITTING OF DATA ON BEST ALGO")
algo.fit(data.build_full_trainset()) # ici on va train notre algo sur le dataset complet, sans cv car les paramètres sont optimaux

dump_name = "dump/dump_SlopeOne"
dump.dump(dump_name, algo=algo, verbose=1)
print("EVERYTHING DONE")