import datetime
time = datetime.datetime.now()
print("launched at : ", time)
import pandas as pd
import numpy as np

#!pip3 install surprise
from surprise import *
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x

print("IMPORT DONE")

n_jobs = int(input("Value for how many processors to use : (-1 is all, -2 is all except one) "))
print("")

############
df = pd.read_csv("Datasets/data_train.csv")

df[["user", "item"]] = df.Id.str.split("_", expand=True)

df.user = df.user.str.replace("r", "")
df.item = df.item.str.replace("c", "")

reader = Reader(rating_scale=(1,5)) 
df_7, df_3 = train_test_split(df, train_size=0.7)

data_train_7 = Dataset.load_from_df(df_7[["user","item","Prediction"]], reader)
data_train_3 = Dataset.load_from_df(df_3[["user","item","Prediction"]], reader)
data_train_7 = data_train_7.build_full_trainset()
data_train_3 = data_train_3.build_full_trainset()
del df
print("DATA AND READER ARE READY")
print("")

###################################################################

# ## Training de chaque algo
f=open("results.txt", "a")
print("TRAINING OF EACH ALGO")
#-------------------------------------------------#-------------------------------------------------#-------------------------------------------------

print("")
print("NORMAL PREDICTOR")

algo_normal_predictor = NormalPredictor()
algo_normal_predictor.fit(data_train_7)
dump.dump("dump/dump_BaselineOnly", algo=algo_normal_predictor, verbose=1)

#-------------------------------------------------#-------------------------------------------------#-------------------------------------------------
print("")
print("BASELINE ONLY")

baseline_only_param_grid = {
    'bsl_options' : {
        'methods' : ['asl', 'sgd']
    }
}
print("BASELINE ONLY PARAMETERS : ", baseline_only_param_grid)
f.write("\n")
f.write("Baseline Only {}\n".format(time))
f.write(baseline_only_param_grid)

algorithm = BaselineOnly
gs = model_selection.GridSearchCV(algorithm, baseline_only_param_grid, measures=['rmse'], n_jobs=n_jobs, joblib_verbose=100)  #enlever mae car non utilisé dans le projet pour sauver du temps

print("BEGINNING OF FITTING GRIDSEARCH")
print("")
gs.fit(data_train_7)
print("FITTING GRIDSEARCH DONE")
print("")
print(gs.best_params)
print(gs.best_score)
f.write("Best param : {}\n" .format(gs.best_params))
f.write("Best score : {}\n" .format(gs.best_score))
f.write("\n")
algo_baseline_only = gs.best_estimator['rmse']
print("FITTING OF DATA ON BEST ALGO")
algo_baseline_only.fit(data_train_7) # ici on va train notre algo sur le dataset complet, sans cv car les paramètres sont optimaux

dump_name = "dump/dump_BaselineOnly"
dump.dump(dump_name, algo=algo_baseline_only, verbose=1)

#-------------------------------------------------#-------------------------------------------------#-------------------------------------------------

# ### KNN
print("")
print("KNN Basic")

knn_basic_param_grid = {
    'k' : [10,40,70,100],
    'min_k': [1,3,5],
    'sim_options' : {
        'user_based' : [True, False]
    }
}  
print("KNN BASIC PARAMETERS : ", knn_basic_param_grid)
f.write("\n")
f.write("KNN basic {}\n".format(time))
f.write(knn_basic_param_grid)

algorithm = KNNBasic
gs = model_selection.GridSearchCV(algorithm, knn_basic_param_grid, measures=['rmse'], n_jobs=n_jobs, joblib_verbose=100)  #enlever mae car non utilisé dans le projet pour sauver du temps

print("BEGINNING OF FITTING GRIDSEARCH")
print("")
gs.fit(data_train_7)
print("FITTING GRIDSEARCH DONE")
print("")
print(gs.best_params)
print(gs.best_score)
f.write("Best param : {}\n" .format(gs.best_params))
f.write("Best score : {}\n" .format(gs.best_score))
f.write("\n")
algo_knn_basic = gs.best_estimator['rmse']
print("FITTING OF DATA ON BEST ALGO")
algo_knn_basic.fit(data_train_7) # ici on va train notre algo sur le dataset complet, sans cv car les paramètres sont optimaux

dump_name = "dump/dump_KNN_basic"
dump.dump(dump_name, algo=algo_knn_basic, verbose=1)

#-------------------------------------------------#-------------------------------------------------#-------------------------------------------------

# ### KNN
print("")
print("KNN Means")

knn_means_param_grid = {
    'k' : [10,40,70,100],
    'min_k': [1,3,5],
    'sim_options' : {
        'user_based' : [True, False]
    }
} 
print("KNN MEANS PARAMETERS : ", knn_means_param_grid)
f.write("\n")
f.write("KNN means {}\n".format(time))
f.write(knn_means_param_grid)

algorithm = KNNWithMeans
gs = model_selection.GridSearchCV(algorithm, knn_means_param_grid, measures=['rmse'], n_jobs=n_jobs, joblib_verbose=100)  #enlever mae car non utilisé dans le projet pour sauver du temps

print("BEGINNING OF FITTING GRIDSEARCH")
print("")
gs.fit(data_train_7)
print("FITTING GRIDSEARCH DONE")
print("")
print(gs.best_params)
print(gs.best_score)
f.write("Best param : {}\n" .format(gs.best_params))
f.write("Best score : {}\n" .format(gs.best_score))
f.write("\n")
algo_knn_means = gs.best_estimator['rmse']
print("FITTING OF DATA ON BEST ALGO")
algo_knn_means.fit(data_train_7) # ici on va train notre algo sur le dataset complet, sans cv car les paramètres sont optimaux

dump_name = "dump/dump_KNN_means"
dump.dump(dump_name, algo=algo_knn_means, verbose=1)


#-------------------------------------------------#-------------------------------------------------#-------------------------------------------------

# ### KNN
print("")
print("KNN Baseline")

knn_baseline_param_grid = {
    'k' : [10,40,70,100],
    'min_k': [1,3,5],
    'sim_options' : {
        'user_based' : [True, False]
    },
    'bsl_options' : {
        'methods' : ['asl', 'sgd']
    }
} 
print("KNN BASELINE PARAMETERS : ", knn_baseline_param_grid)
f.write("\n")
f.write("KNN Baseline {}\n".format(time))
f.write(knn_baseline_param_grid)

algorithm = KNNBaseline
gs = model_selection.GridSearchCV(algorithm, knn_baseline_param_grid, measures=['rmse'], n_jobs=n_jobs, joblib_verbose=100)  #enlever mae car non utilisé dans le projet pour sauver du temps

print("BEGINNING OF FITTING GRIDSEARCH")
print("")
gs.fit(data_train_7)
print("FITTING GRIDSEARCH DONE")
print("")
print(gs.best_params)
print(gs.best_score)
f.write("Best param : {}\n" .format(gs.best_params))
f.write("Best score : {}\n" .format(gs.best_score))
f.write("\n")
algo_knn_baseline = gs.best_estimator['rmse']
print("FITTING OF DATA ON BEST ALGO")
algo_knn_baseline.fit(data_train_7) # ici on va train notre algo sur le dataset complet, sans cv car les paramètres sont optimaux

dump_name = "dump/dump_KNN_baseline"
dump.dump(dump_name, algo=algo_knn_baseline, verbose=1)

#-------------------------------------------------#-------------------------------------------------#-------------------------------------------------

print("")
print("SVD")

svd_param_grid = {
    'n_factors' : [100,200],
    'n_epochs': [20,100,200],
    'lr_all': [0.001,0.002,0.003,0.004,0.005,0.01,0.02,0.03,0.04,0.05],
    'reg_all': [0.01,0.02,0.03,0.04,0.05,0.1,0.2,0.3,0.4,0.5]
}  
print("SVD PARAMETERS : ", svd_param_grid)
f.write("\n")
f.write("SVD {}\n".format(time))
f.write(svd_param_grid)

algorithm = SVD
gs = model_selection.GridSearchCV(algorithm, svd_param_grid, measures=['rmse'], n_jobs=n_jobs, joblib_verbose=100)  #enlever mae car non utilisé dans le projet pour sauver du temps

print("BEGINNING OF FITTING GRIDSEARCH")
print("")
gs.fit(data_train_7)
print("FITTING GRIDSEARCH DONE")
print("")
print(gs.best_params)
print(gs.best_score)
f.write("Best param : {}\n" .format(gs.best_params))
f.write("Best score : {}\n" .format(gs.best_score))
f.write("\n")
algo_svd = gs.best_estimator['rmse']
print("FITTING OF DATA ON BEST ALGO")
algo_svd.fit(data_train_7) # ici on va train notre algo sur le dataset complet, sans cv car les paramètres sont optimaux

dump_name = "dump/dump_SVD"
dump.dump(dump_name, algo=algo_svd, verbose=1)


#-------------------------------------------------#-------------------------------------------------#-------------------------------------------------

print("")
print("NMF")

nmf_param_grid = {
    'n_factors' : [50,100],
    'n_epochs': [20,100,200]
}  
print("NMF PARAMETERS : ", nmf_param_grid)
f.write("\n")
f.write("NMF {}\n".format(time))
f.write(nmf_param_grid)

algorithm = NMF
gs = model_selection.GridSearchCV(algorithm, nmf_param_grid, measures=['rmse'], n_jobs=n_jobs, joblib_verbose=100)  #enlever mae car non utilisé dans le projet pour sauver du temps

print("BEGINNING OF FITTING GRIDSEARCH")
print("")
gs.fit(data_train_7)
print("FITTING GRIDSEARCH DONE")
print("")
print(gs.best_params)
print(gs.best_score)
f.write("Best param : {}\n" .format(gs.best_params))
f.write("Best score : {}\n" .format(gs.best_score))
f.write("\n")
algo_nmf = gs.best_estimator['rmse']
print("FITTING OF DATA ON BEST ALGO")
algo_nmf.fit(data_train_7) # ici on va train notre algo sur le dataset complet, sans cv car les paramètres sont optimaux

dump_name = "dump/dump_NMF"
dump.dump(dump_name, algo=algo_nmf, verbose=1)

#-------------------------------------------------#-------------------------------------------------#-------------------------------------------------# ### SlopeOne
print("")
print("SLOPEONE")

algo_slopeone = SlopeOne()
algo_slopeone.fit(data_train_7)
dump.dump("dump/SlopeOne_fitted_dump", algo=algo_slopeone, verbose=1)

#-------------------------------------------------#-------------------------------------------------#-------------------------------------------------

print("")
print("CoClustering")

coclustering_param_grid = {
    'n_cltr_u' : [1,3,5,10,20],
    'n_cltr_i': [1,3,5,10,20]
}  
print("COCLUSTERING PARAMETERS : ", coclustering_param_grid)
f.write("\n")
f.write("COCLUSTERING {}\n".format(time))
f.write(coclustering_param_grid)

algorithm = CoClustering
gs = model_selection.GridSearchCV(algorithm, coclustering_param_grid, measures=['rmse'], n_jobs=n_jobs, joblib_verbose=100)  #enlever mae car non utilisé dans le projet pour sauver du temps

print("BEGINNING OF FITTING GRIDSEARCH")
print("")
gs.fit(data_train_7)
print("FITTING GRIDSEARCH DONE")
print("")
print(gs.best_params)
print(gs.best_score)
f.write("Best param : {}\n" .format(gs.best_params))
f.write("Best score : {}\n" .format(gs.best_score))
f.write("\n")
algo_coclustering = gs.best_estimator['rmse']
print("FITTING OF DATA ON BEST ALGO")
algo_coclustering.fit(data_train_7) # ici on va train notre algo sur le dataset complet, sans cv car les paramètres sont optimaux

dump_name = "dump/dump_CoClustering"
dump.dump(dump_name, algo=algo_coclustering, verbose=1)

#-------------------------------------------------#-------------------------------------------------#-------------------------------------------------# ### SlopeOne
f.close()
del gs
print("FITTING OF ALL ALGOS IS DONE")

df2 = pd.read_csv("Datasets/sample_submission.csv")

df2[["user", "item"]] = df2.Id.str.split("_", expand=True)

df2.user = df2.user.str.replace("r", "")
df2.item = df2.item.str.replace("c", "")

reader = Reader(rating_scale=(1,5)) 
data_test = Dataset.load_from_df(df2[["user","item","Prediction"]], reader)
test = data_test.build_full_trainset()
test = test.build_testset()


#-------------------------------------------------#-------------------------------------------------#-------------------------------------------------# ### SlopeOne
print("TESTING OF THE ALGOS")
array_normal_pred = algo_normal_predictor.test(test)
array_baseline_only = algo_baseline_only.test(test)
array_knn_basic = algo_knn_basic.test(test)
array_knn_means  = algo_knn_means.test(test)
array_knn_baseline = algo_knn_baseline.test(test)
array_svd = algo_svd.test(test)
array_nmf = algo_nmf.test(test)
array_coclustering = algo_coclustering.test(test)


print("DONE")
print("")
print("BLENDING EACH ALGO INTO AN ARRAY")
pred_array=np.vstack(
    [array_normal_pred[:,3], 
    array_baseline_only[:,3], 
    array_knn_basic[:,3], 
    array_knn_means[:,3],
    array_knn_baseline[:,3], 
    array_svd[:,3], 
    array_nmf[:,3],
    array_coclustering[:,3]]
    )
pred_array[np.where(pred_array>5)]=5
pred_array[np.where(pred_array<1)]=1


array_normal_pred = algo_normal_predictor.test(data_train_3)
array_baseline_only = algo_baseline_only.test(data_train_3)
array_knn_basic = algo_knn_basic.test(data_train_3)
array_knn_means  = algo_knn_means.test(data_train_3)
array_knn_baseline = algo_knn_baseline.test(data_train_3)
array_svd = algo_svd.test(data_train_3)
array_nmf = algo_nmf.test(data_train_3)
array_coclustering = algo_coclustering.test(data_train_3)

X=np.vstack(
    [array_normal_pred[:,3], 
    array_baseline_only[:,3], 
    array_knn_basic[:,3], 
    array_knn_means[:,3],
    array_knn_baseline[:,3], 
    array_svd[:,3], 
    array_nmf[:,3],
    array_coclustering[:,3]]
    )
X[np.where(X>5)]=5
X[np.where(X<1)]=1


y=np.array(df_3.Prediction.values)
clf=RidgeCV(alphas=np.linspace(10**-5,1,10),cv=10)
X=standardize(X)[0]
clf=clf.fit(X,y)

pred_array=standardize(pred_array)[0]
pred=clf.predict(pred_array)
final_array=np.rint(pred)
final_array[np.where(final_array>5)]=5
final_array[np.where(final_array<1)]=1


df2.Prediction = final_array
df2 = df2.drop(columns=["user", "item"])
df2.to_csv("Datasets/submission_pipeline.csv", index=False)

print("EVERYTHING DONE")
