'''
MT2017006 Aditi Baghel
MT2017012 Akshita
MT2017037 Deepali Shinde
MT2017128 Tarun Pandey
'''
import pandas as pd
import numpy as np
import pickle
from sklearn.mixture import GaussianMixture

def get_features(data):
	features=[]
	for i in range(0,len(data)):
		instance=data['feature'][i].tolist()
		features.append(instance)
	feature1=pd.DataFrame([subject for subject in features])
	return feature1

unique_labels=[]
lns=open("unique_labels.txt",'r')
for line in lns:
	unique_labels.append((line.split('\n'))[0])
	os.makedirs('models/'+line, exist_ok=True)

for phoneme in unique_labels:
	fname=str(phoneme)+"_train.hdf"
	df=pd.read_hdf("./features2/mfcc/"+fname)
	df_d=pd.read_hdf("./features2/mfcc_delta/"+fname)
	df_dd=pd.read_hdf("./features2/mfcc_delta_delta/"+fname)
	features=get_features(df)
	features_d=get_features(df_d)
	features_dd=get_features(df_dd)
	for mixtur in range(1,9):
		mixture=pow(2,mixtur)
		model_name="ec_mfcc"+str(mixture)+".pkl"
		model_name_w="wec_mfcc"+str(mixture)+".pkl"
		model=GaussianMixture(n_components=mixture, covariance_type='diag')
		model.fit(features.as_matrix(columns=None))
		model_w=GaussianMixture(n_components=mixture, covariance_type='diag')
		model_w.fit((features.drop(columns=[0],axis=1)).as_matrix(columns=None))
		pickle.dump(model,open("./models/"+str(phoneme)+"/"+model_name, 'wb'))
		pickle.dump(model_w, open("./models/"+str(phoneme)+"/"+model_name_w, 'wb'))


	model_delta_name="ec_delta.pkl"
	model_delta_name_w="wec_delta.pkl"
	model_dd_name="ec_delta_delta.pkl"
	model_dd_name_w="wec_delta_delta.pkl"
	model1=GaussianMixture(n_components=64, covariance_type='diag')
	model1.fit(features_d.as_matrix(columns=None))
	pickle.dump(model1, open("./models/"+str(phoneme)+"/"+model_delta_name, 'wb'))
	model1_w=GaussianMixture(n_components=64,covariance_type='diag')
	model1_w.fit((features_d.drop(columns=[0,13],axis=1)).as_matrix(columns=None))
	pickle.dump(model1_w, open("./models/"+str(phoneme)+"/"+model_delta_name_w, 'wb'))
	model2=GaussianMixture(n_components=64, covariance_type='diag')
	model2.fit(features_dd.as_matrix(columns=None))
	pickle.dump(model2, open("./models/"+str(phoneme)+"/"+model_dd_name, 'wb'))
	model2_w=GaussianMixture(n_components=64, covariance_type='diag')
	model2_w.fit((feature_dd.drop(columns=[0,13,26],axis=1)).as_matrix(columns=None))
	pickle.dump(model2_w, open("./models/"+str(phoneme)+"/"+model_dd_name_w, 'wb'))




#Training code
