import pandas as pd
import numpy as np
from surprise import Dataset
from surprise import Reader

df = pd.read_csv("Datasets/data_train.csv")

df[["user", "item"]] = df.Id.str.split("_", expand=True)

df.user = df.user.str.replace("r", "")
df.item = df.item.str.replace("c", "")
print("Loaded")
reader = Reader(rating_scale=(1,5)) 
data = Dataset.load_from_df(df[["user","item","Prediction"]], reader)
print(data)
unique_user=df['user'].unique()

c=0
for u in unique_user:
    mean=df.loc[df.user==u,'Prediction'].mean()
    var=df.loc[df.user==u,'Prediction'].var()
    df.loc[df.user==u,'Prediction']=(df.loc[df.user==u,'Prediction']-mean)/var
    #if (c%500==0):
    print(c, mean, var)
    c+=1

mean_var_user=[]
for u in unique_user :
    mean=df.loc[df.user==u,'Prediction'].mean()
    var=df.loc[df.user==u,'Prediction'].var()
    mean_var_user.append((mean,var))

unique_items=df['items'].unique()
mean_var_items=[]
for i in unique_item :
    mean=df.loc[df.item==i,'Prediction'].mean()
    var=df.loc[df.item==i,'Prediction'].var()
    mean_var_items.append((mean,var))

for r in df.iterrows():
    print(df.loc[r])
